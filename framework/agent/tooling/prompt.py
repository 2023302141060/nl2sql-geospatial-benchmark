# -*- coding: utf-8 -*-
"""从注册表生成 System Prompt 中的 ## Tooling 区块（与 API tools[] 同源）。"""
from __future__ import annotations

from typing import Any

from agent.tooling.registry import ExecutionRoute, TOOL_SPECS, ToolSpec


def _route_hint(spec: ToolSpec) -> str:
    if spec.execution_route == ExecutionRoute.INLINE:
        return "执行方式：由框架在通用工具节点内直接调用。"
    if spec.execution_route == ExecutionRoute.GRAPH_TEXT2SQL:
        return "执行方式：由专用 text2sql 节点执行（勿在对话中手写 SQL）。"
    if spec.execution_route == ExecutionRoute.GRAPH_PYTHON:
        return "执行方式：由专用 python_analysis 节点执行（勿在正文中手写可执行 Python）。"
    return ""


def _fields_summary(spec: ToolSpec) -> str:
    model = spec.args_model
    if model is None:
        return "（无参数模型）"
    lines: list[str] = []
    try:
        schema = model.model_json_schema()
    except Exception:
        return "（参数详见 API 工具 schema）"
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for name, meta in props.items():
        if not isinstance(meta, dict):
            continue
        typ = meta.get("type", "any")
        if isinstance(typ, list):
            typ = "/".join(str(x) for x in typ)
        desc = str(meta.get("description") or "").strip()
        req = "必填" if name in required else "可选"
        extra = f" — {desc}" if desc else ""
        lines.append(f"  - `{name}`（{req}，{typ}）{extra}")
    if not lines:
        return "（无字段说明）"
    return "\n".join(lines)


def build_tooling_markdown(specs: list[ToolSpec] | None = None) -> str:
    """生成 OpenClaw 风格的 ## Tooling 段落。"""
    use = specs if specs is not None else TOOL_SPECS
    parts: list[str] = ["## Tooling", "", "以下为当前 Agent 可用工具集（与接口 `tools[]` 一致）。参数须为顶层平铺键值对，禁止 `parameters` / `args` 套壳。", ""]
    for spec in use:
        parts.append(f"### `{spec.name}`")
        parts.append(spec.description or "（无描述）")
        rh = _route_hint(spec)
        if rh:
            parts.append(rh)
        parts.append("**参数：**")
        parts.append(_fields_summary(spec))
        parts.append("")
    parts.append("**调用约定：** 使用原生 Function / Tool Calling；每轮至多一次 tool call（除非系统另有说明）。")
    return "\n".join(parts).rstrip()
