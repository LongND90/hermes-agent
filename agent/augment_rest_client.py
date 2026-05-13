"""OpenAI-compatible shim that forwards Hermes requests to Augment REST API.

Calls ``POST {api_url}/chat-stream`` directly via HTTPS using a session token
loaded from ``~/.augment/session.json`` (or AUGMENT_API_TOKEN / AUGMENT_API_URL
env vars).  This is an alternative to the ACP subprocess path provided by
``auggie-acp`` — same backend service, different transport.

The wire schema is reverse-engineered from ``@augmentcode/auggie-sdk`` v0.1.15
(``dist/auggie/ai-sdk-provider.js``).  Streaming response is NDJSON; each line
is a JSON object with optional ``text`` deltas, ``nodes``, and ``stop_reason``.
"""

from __future__ import annotations

import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Iterable

import httpx

from agent.augment_rest_messages import (
    build_chat_request,
    map_stop_reason_to_finish,
    tools_to_definitions,
)

logger = logging.getLogger(__name__)

REST_MARKER_BASE_URL = "augment-rest://chat-stream"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_USER_AGENT = "Hermes-Agent/0.1 augment-rest"


def _coerce_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


def _iter_ndjson(response: httpx.Response) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from an httpx streaming response (NDJSON)."""
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.debug("augment-rest: skipping unparseable NDJSON line")
            continue


def _build_tool_call(node_tool_use: dict[str, Any], index: int) -> SimpleNamespace:
    call_id = str(node_tool_use.get("tool_use_id") or f"augment_call_{index+1}")
    name = str(node_tool_use.get("tool_name") or "")
    args_raw = node_tool_use.get("input_json") or "{}"
    args = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


def _parse_stream(
    response: httpx.Response,
) -> tuple[str, str, list[SimpleNamespace], dict[str, int], int | None]:
    """Parse the NDJSON stream into (text, reasoning_text, tool_calls, usage, stop_reason)."""
    accumulated_text = ""
    reasoning_text = ""
    tool_calls: list[SimpleNamespace] = []
    seen_tool_ids: set[str] = set()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    stop_reason: int | None = None

    for chunk in _iter_ndjson(response):
        text_delta = chunk.get("text")
        if isinstance(text_delta, str) and text_delta:
            accumulated_text += text_delta
        for node in chunk.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type")
            if node_type == 5 and node.get("tool_use"):
                tool_id = str(node["tool_use"].get("tool_use_id") or "")
                if tool_id and tool_id in seen_tool_ids:
                    continue
                if tool_id:
                    seen_tool_ids.add(tool_id)
                tool_calls.append(_build_tool_call(node["tool_use"], len(tool_calls)))
            elif node_type == 8 and node.get("thinking"):
                thinking_content = (
                    node["thinking"].get("content")
                    or node["thinking"].get("summary")
                    or ""
                )
                if thinking_content:
                    reasoning_text += thinking_content
            elif node_type == 10 and node.get("token_usage"):
                tu = node["token_usage"]
                usage["prompt_tokens"] = int(tu.get("input_tokens") or 0)
                usage["completion_tokens"] = int(tu.get("output_tokens") or 0)
                usage["cached_tokens"] = int(
                    tu.get("cache_read_input_tokens")
                    or tu.get("cache_read_tokens")
                    or 0
                )
        sr = chunk.get("stop_reason")
        if sr is not None:
            stop_reason = sr

    return accumulated_text, reasoning_text, tool_calls, usage, stop_reason


def _build_completion_response(
    *,
    model: str,
    text: str,
    reasoning_text: str,
    tool_calls: list[SimpleNamespace],
    usage: dict[str, int],
    stop_reason: int | None,
    conversation_id: str,
) -> SimpleNamespace:
    finish_reason = map_stop_reason_to_finish(stop_reason, has_tool_calls=bool(tool_calls))
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    usage_obj = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    assistant_message = SimpleNamespace(
        content=text,
        tool_calls=tool_calls or None,
        reasoning=reasoning_text or None,
        reasoning_content=reasoning_text or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=f"augment-rest-{conversation_id[:8]}",
        choices=[choice],
        usage=usage_obj,
        model=model,
        conversation_id=conversation_id,
    )


class _ChatCompletions:
    def __init__(self, client: "AugmentRestClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "AugmentRestClient") -> None:
        self.completions = _ChatCompletions(client)



class AugmentRestClient:
    """Minimal OpenAI-client-compatible facade for Augment REST API.

    Exposes ``client.chat.completions.create(...)`` returning a
    ``SimpleNamespace`` matching the OpenAI ChatCompletion shape so existing
    Hermes core dispatch can consume it without special-casing.  Also exposes
    a flat ``chat_completions_create(...)`` helper as documented in the
    Phase 2a spec.

    Persists ``conversation_id`` across calls so server-side prompt caching
    stays valid for the lifetime of the client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str | None = None,
        base_url: str | None = None,
        model: str = "",
        conversation_id: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> None:
        if not api_key:
            raise ValueError("AugmentRestClient requires a non-empty api_key")
        resolved_url = (api_url or base_url or "").strip()
        if not resolved_url or resolved_url == REST_MARKER_BASE_URL:
            raise ValueError(
                "AugmentRestClient requires api_url (tenant URL from "
                "~/.augment/session.json or AUGMENT_API_URL)"
            )
        self.api_key = api_key
        self.api_url = resolved_url.rstrip("/")
        # Public ``base_url`` mirrors the marker scheme so Hermes status/log
        # surfaces don't expose the tenant URL.  ``api_url`` keeps the real
        # tenant endpoint for outgoing HTTP requests.
        self.base_url = REST_MARKER_BASE_URL
        self.model = model
        self.conversation_id = str(conversation_id or uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self._default_headers = dict(default_headers or {})
        self._timeout = _coerce_timeout(timeout)
        self.chat = _ChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Request-Session-Id": self.session_id,
            "X-Request-Id": request_id,
            "conversation-id": self.conversation_id,
            "X-Mode": "sdk",
            "User-Agent": _USER_AGENT,
        }
        for k, v in self._default_headers.items():
            if k.lower() != "authorization":
                headers[k] = v
        return headers

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
    ) -> dict[str, Any]:
        request = build_chat_request(messages or [])
        payload: dict[str, Any] = {
            "mode": "CLI_AGENT",
            "model": model or self.model,
            "message": request["message"],
            "nodes": request["nodes"],
            "chat_history": request["chat_history"],
            "conversation_id": self.conversation_id,
        }
        tool_defs = tools_to_definitions(tools)
        if tool_defs:
            payload["tool_definitions"] = tool_defs
        return payload

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        if stream:
            raise NotImplementedError(
                "augment-rest does not expose a streaming iterator; the NDJSON "
                "transport is parsed internally and returned as a single response."
            )
        effective_model = model or self.model
        if not effective_model:
            raise ValueError("AugmentRestClient: model is required")
        effective_timeout = _coerce_timeout(timeout) if timeout is not None else self._timeout
        payload = self._build_payload(messages or [], tools, effective_model)
        request_id = str(uuid.uuid4())
        url = f"{self.api_url}/chat-stream"
        try:
            with httpx.Client(timeout=effective_timeout) as http:
                with http.stream(
                    "POST",
                    url,
                    headers=self._headers(request_id),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")[:500]
                        raise RuntimeError(
                            f"augment-rest HTTP {response.status_code}: {body}"
                        )
                    text, reasoning, tool_calls, usage, stop_reason = _parse_stream(response)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"augment-rest request failed: {exc}") from exc
        return _build_completion_response(
            model=effective_model,
            text=text,
            reasoning_text=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            conversation_id=self.conversation_id,
        )

    def chat_completions_create(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return self._create_chat_completion(
            messages=messages, tools=tools, stream=stream, **kwargs
        )

