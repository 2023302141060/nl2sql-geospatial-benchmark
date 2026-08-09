# -*- coding: utf-8 -*-
"""从模型响应解析 tool_calls：仅原生字段与 additional_kwargs（网关兼容），不用正文 JSON 兜底。"""
import json
from typing import Any


def extract_tool_calls_from_additional_kwargs(response: Any) -> list[dict[str, Any]]:
    """兼容部分模型/网关把工具调用塞进 additional_kwargs 的情况。"""
    additional = getattr(response, "additional_kwargs", {}) or {}

    raw_tool_calls = additional.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    normalized_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue

        function_payload = item.get("function") or {}
        tool_name = item.get("name") or function_payload.get("name") or ""
        raw_args = item.get("args")
        if raw_args is None:
            raw_args = function_payload.get("arguments", {})

        parsed_args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            parsed_args = raw_args
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
                if isinstance(loaded, dict):
                    parsed_args = loaded
            except Exception:
                parsed_args = {}

        if tool_name:
            normalized_calls.append({
                "id": item.get("id") or item.get("tool_call_id") or function_payload.get("id") or tool_name,
                "name": tool_name,
                "args": parsed_args,
                "type": item.get("type", "tool_call"),
            })

    return normalized_calls


def merge_native_tool_calls(response: Any) -> list[dict[str, Any]]:
    """合并 AIMessage.tool_calls 与 additional_kwargs 中的 tool_calls（无正文解析）。"""
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    if not tool_calls:
        tool_calls = extract_tool_calls_from_additional_kwargs(response)
    return tool_calls
