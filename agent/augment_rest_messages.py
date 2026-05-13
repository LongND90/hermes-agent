"""OpenAI-shape → Augment REST payload converters.

Kept in a separate module so the AugmentRestClient stays focused on transport
while these stateless helpers can be unit-tested without spinning up HTTP.

Wire schema reverse-engineered from ``@augmentcode/auggie-sdk`` v0.1.15
(``dist/auggie/ai-sdk-provider.js``).
"""

from __future__ import annotations

import json
from typing import Any

# Stop reason → OpenAI finish_reason
# 1=END_TURN, 2=MAX_TOKENS, 3=TOOL_USE_REQUESTED, 4=SAFETY, 5=RECITATION,
# 6=MALFORMED_FUNCTION_CALL (treated as ``stop`` per task brief).
_STOP_REASON_MAP = {
    1: "stop",
    2: "length",
    3: "tool_calls",
    4: "content_filter",
    5: "content_filter",
    6: "stop",
}


def map_stop_reason_to_finish(
    stop_reason: int | None, *, has_tool_calls: bool
) -> str:
    if has_tool_calls and stop_reason in (None, 3):
        return "tool_calls"
    if stop_reason is None:
        return "stop"
    return _STOP_REASON_MAP.get(int(stop_reason), "stop")


def tools_to_definitions(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert OpenAI ``tools`` list into Augment ``tool_definitions``.

    OpenAI shape: ``{"type": "function", "function": {"name", "description",
    "parameters"}}`` (or rare flattened ``{"name", "description", "parameters"}``).
    Augment shape: ``{"name", "description", "input_schema_json"}``.
    """
    if not tools:
        return []
    defs: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if "function" in tool else tool
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        defs.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "input_schema_json": json.dumps(schema),
            }
        )
    return defs


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _user_nodes(text: str, start_id: int) -> list[dict[str, Any]]:
    if not text:
        return []
    return [{"id": start_id, "type": 0, "text_node": {"content": text}}]


def _tool_result_nodes(msg: dict[str, Any], start_id: int) -> list[dict[str, Any]]:
    """Build TOOL_RESULT (type=1) nodes from an OpenAI ``role=tool`` message."""
    tool_use_id = str(msg.get("tool_call_id") or msg.get("tool_use_id") or "")
    content = _message_text(msg.get("content"))
    if not tool_use_id and not content:
        return []
    return [
        {
            "id": start_id,
            "type": 1,
            "tool_result_node": {
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": False,
            },
        }
    ]


def _assistant_response_nodes(msg: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Build response nodes for a historical assistant turn (chat_history)."""
    nodes: list[dict[str, Any]] = []
    text = _message_text(msg.get("content"))
    nid = 0
    if text:
        nodes.append({"id": nid, "type": 0, "content": text})
        nid += 1
    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args_raw = fn.get("arguments")
        if isinstance(args_raw, (dict, list)):
            args_raw = json.dumps(args_raw)
        nodes.append(
            {
                "id": nid,
                "type": 5,
                "tool_use": {
                    "tool_use_id": str(call.get("id") or ""),
                    "tool_name": str(fn.get("name") or ""),
                    "input_json": str(args_raw or "{}"),
                },
            }
        )
        nid += 1
    return nodes, text


def build_chat_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OpenAI ``messages`` list into Augment request fields.

    Returns a dict with keys ``message``, ``nodes``, ``chat_history``:
      - the *latest* user turn becomes ``message`` + ``nodes`` (the request),
      - earlier user/tool/assistant turns become ``chat_history`` entries.

    Caller is responsible for adding ``mode``, ``model``, ``conversation_id``,
    and ``tool_definitions`` to the final payload.
    """
    chat_history: list[dict[str, Any]] = []
    pending_nodes: list[dict[str, Any]] = []
    pending_text = ""
    node_id = 0

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            system_text = _message_text(msg.get("content"))
            if not system_text:
                continue
            pending_nodes.append(
                {
                    "id": node_id,
                    "type": 0,
                    "text_node": {"content": f"System: {system_text}"},
                }
            )
            node_id += 1
            pending_text = (
                f"{pending_text}\n\nSystem: {system_text}" if pending_text else f"System: {system_text}"
            )
        elif role == "user":
            user_text = _message_text(msg.get("content"))
            new_nodes = _user_nodes(user_text, node_id)
            pending_nodes.extend(new_nodes)
            node_id += len(new_nodes)
            if user_text:
                pending_text = (
                    f"{pending_text}\n{user_text}" if pending_text else user_text
                )
        elif role == "tool":
            new_nodes = _tool_result_nodes(msg, node_id)
            pending_nodes.extend(new_nodes)
            node_id += len(new_nodes)
        elif role == "assistant":
            response_nodes, response_text = _assistant_response_nodes(msg)
            chat_history.append(
                {
                    "request_message": pending_text,
                    "request_nodes": pending_nodes,
                    "response_text": response_text,
                    "response_nodes": response_nodes,
                }
            )
            pending_nodes = []
            pending_text = ""
            node_id = 0

    # Re-id pending nodes starting from 0 to match the TS SDK behaviour.
    for i, node in enumerate(pending_nodes):
        node["id"] = i

    return {
        "message": pending_text,
        "nodes": pending_nodes,
        "chat_history": chat_history,
    }
