"""Mem0 LLM adapter that uses the Augment subscription via auggie-sdk.

Reads OAuth credentials from ``~/.augment/session.json`` (no env vars
required) and routes mem0 fact-extraction calls through an Auggie agent
configured for ``sonnet4.5``.

Mem0 invokes ``generate_response`` in two shapes:
- Fact extraction: ``response_format={"type": "json_object"}`` — returns a
  JSON string mem0 will then ``json.loads``.
- Procedural memory: no ``response_format`` — returns a plain string.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)


_SESSION_PATH = Path("~/.augment/session.json").expanduser()


def _load_augment_session() -> Tuple[str, str]:
    """Return ``(api_key, api_url)`` from ``~/.augment/session.json``.

    Raises ``RuntimeError`` with an actionable message if the file is
    missing or malformed.  Never logs the token value itself.
    """
    if not _SESSION_PATH.exists():
        raise RuntimeError(
            "Augment session not found at ~/.augment/session.json. "
            "Run `auggie login` first."
        )
    try:
        data = json.loads(_SESSION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ~/.augment/session.json: {exc}") from exc
    api_key = data.get("accessToken")
    api_url = data.get("tenantURL")
    if not api_key or not api_url:
        raise RuntimeError(
            "~/.augment/session.json missing accessToken or tenantURL keys."
        )
    return api_key, api_url


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten OpenAI-style messages into a single Auggie instruction.

    Mem0 typically passes ``[{system}, {user}]``; we render them as labelled
    sections so Auggie sees the system instructions first.
    """
    parts = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = msg.get("content") or ""
        if not content:
            continue
        parts.append(f"### {role}\n{content}")
    return "\n\n".join(parts)


class Mem0AugmentLLM(LLMBase):
    """Mem0 ``LLMBase`` implementation backed by Augment via auggie-sdk."""

    def __init__(self, config=None):
        super().__init__(config)
        self._agent = None
        self._agent_lock = threading.Lock()
        self._model_name = (getattr(self.config, "model", None) or "sonnet4.5")

    def _validate_config(self):
        # No model attribute is fine — we default to sonnet4.5 ourselves.
        return

    def _get_agent(self):
        """Lazy-instantiate the Auggie client (heavy: spawns a subprocess)."""
        with self._agent_lock:
            if self._agent is not None:
                return self._agent
            from auggie_sdk import Auggie

            api_key, api_url = _load_augment_session()
            self._agent = Auggie(
                model=self._model_name,
                api_key=api_key,
                api_url=api_url,
            )
            return self._agent

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> str:
        if tools:
            logger.warning(
                "Mem0AugmentLLM ignoring tools= argument (Augment SDK call shape)."
            )

        prompt = _messages_to_prompt(messages)
        wants_json = bool(
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") in ("json_object", "json_schema")
        )

        agent = self._get_agent()
        try:
            if wants_json:
                result = agent.run(prompt, return_type=dict)
                if isinstance(result, str):
                    return result
                try:
                    return json.dumps(result)
                except (TypeError, ValueError):
                    return json.dumps({"memory": []})
            else:
                result = agent.run(prompt)
                if isinstance(result, str):
                    return result
                return str(result) if result is not None else ""
        except Exception as exc:
            logger.error("Mem0AugmentLLM.generate_response failed: %s", exc)
            if wants_json:
                return json.dumps({"memory": []})
            raise
