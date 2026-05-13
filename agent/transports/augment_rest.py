"""Augment REST transport.

Handles api_mode='augment_rest' for the Augment proprietary chat-stream
endpoint.  ``AugmentRestClient.build_chat_request`` performs all of the
provider-specific message conversion internally, so this transport keeps
convert_messages / convert_tools as identity transforms and only forwards
the keys ``_create_chat_completion`` actually consumes.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


class AugmentRestTransport(ProviderTransport):
    """Transport for api_mode='augment_rest'."""

    @property
    def api_mode(self) -> str:
        return "augment_rest"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        return messages

    def convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **_params,
    ) -> Dict[str, Any]:
        """Return only the keys ``AugmentRestClient._create_chat_completion`` accepts.

        chat-completions kwargs (timeout, base_url, max_tokens, reasoning_config,
        request_overrides, extra_body, etc.) are dropped intentionally.
        """
        return {
            "model": model,
            "messages": messages,
            "tools": tools,
        }

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize the SimpleNamespace returned by AugmentRestClient.

        Shape mirrors OpenAI ChatCompletion: ``response.choices[0].message``
        with ``content``, ``tool_calls`` (each ``.id`` + ``.function.name`` +
        ``.function.arguments``), ``reasoning`` and ``reasoning_content``.
        """
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        usage = None
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        reasoning = getattr(msg, "reasoning", None)
        reasoning_content = getattr(msg, "reasoning_content", None)
        provider_data: Dict[str, Any] = {}
        if reasoning_content is not None:
            provider_data["reasoning_content"] = reasoning_content

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        if response is None:
            return False
        if not hasattr(response, "choices") or response.choices is None:
            return False
        if not response.choices:
            return False
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            return None
        cached = getattr(details, "cached_tokens", 0) or 0
        if cached:
            return {"cached_tokens": cached, "creation_tokens": 0}
        return None


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("augment_rest", AugmentRestTransport)
