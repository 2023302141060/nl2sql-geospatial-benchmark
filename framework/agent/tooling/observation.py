# -*- coding: utf-8 -*-
"""统一构造 ToolMessage（Observation）；本模块为仓库内唯一允许直接实例化 ToolMessage 的位置。"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import ToolMessage

# 并发 tool_calls 中非主执行条目的占位失败码（与 _is_skip_parallel_tool_message 对齐）
FAILURE_CODE_SERIAL_EXECUTION_IGNORED = "serial_execution_ignored"

_IGNORED_SERIAL_CONTENT = (
    "[System] 该工具调用已被系统忽略，因为当前强制要求单步串行执行。"
)


def _normalize_tool_call_id(tool_call_id: Any) -> str:
    raw = "" if tool_call_id is None else str(tool_call_id).strip()
    return raw if raw else uuid.uuid4().hex


def _normalize_failure_code(failure_code: str | None) -> str:
    return "" if failure_code is None else str(failure_code).strip()


def tool_observation(
    *,
    content: str,
    tool_call_id: Any,
    tool_name: str,
    success: bool,
    payload: dict[str, Any] | None = None,
    failure_code: str | None = None,
    followup_hint: str | None = None,
    parallel_skipped: bool = False,
) -> ToolMessage:
    """系统中唯一规范的 ToolMessage 构造器：统一 content 后缀与 additional_kwargs 四键结构。"""
    tid = _normalize_tool_call_id(tool_call_id)
    fc = _normalize_failure_code(failure_code)
    body = str(content or "")
    if followup_hint and str(followup_hint).strip():
        body = f"{body}\n\n[System Hint] {str(followup_hint).strip()}"
    extra: dict[str, Any] = {
        "tool_name": tool_name,
        "success": success,
        "payload": payload,
        "failure_code": fc,
    }
    if parallel_skipped:
        extra["parallel_skipped"] = True
    return ToolMessage(content=body, tool_call_id=tid, additional_kwargs=extra)


def build_ignored_tool_messages(ignored_tool_calls: list[Any]) -> list[ToolMessage]:
    """为未执行的 tool_calls 生成占位 ToolMessage（单步串行策略）。"""
    out: list[ToolMessage] = []
    for call in ignored_tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "unknown_tool")
        out.append(
            tool_observation(
                content=_IGNORED_SERIAL_CONTENT,
                tool_call_id=call.get("id"),
                tool_name=name,
                success=False,
                payload={"status": "ignored"},
                failure_code=FAILURE_CODE_SERIAL_EXECUTION_IGNORED,
                parallel_skipped=True,
            )
        )
    return out


def build_tool_response_sequence(
    tool_calls: list[Any],
    primary_index: int,
    primary: ToolMessage,
) -> list[ToolMessage]:
    """按 AIMessage.tool_calls 顺序返回等长 ToolMessage 列表：主下标为真实结果，其余为忽略占位。"""
    if not isinstance(tool_calls, list) or not tool_calls:
        return [primary]
    n = len(tool_calls)
    if primary_index < 0 or primary_index >= n:
        return [primary]
    out: list[ToolMessage] = []
    for i, call in enumerate(tool_calls):
        if i == primary_index:
            out.append(primary)
        else:
            if not isinstance(call, dict):
                call = {}
            name = str(call.get("name") or "unknown_tool")
            out.append(
                tool_observation(
                    content=_IGNORED_SERIAL_CONTENT,
                    tool_call_id=call.get("id"),
                    tool_name=name,
                    success=False,
                    payload={"status": "ignored"},
                    failure_code=FAILURE_CODE_SERIAL_EXECUTION_IGNORED,
                    parallel_skipped=True,
                )
            )
    return out
