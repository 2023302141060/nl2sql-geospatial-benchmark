# -*- coding: utf-8 -*-
"""工具单一注册表：名称、描述、参数模型、LangChain 工具对象与执行路由（inline / 图节点）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Type

from pydantic import BaseModel

from agent.tools import (
    MapRenderingArgs,
    PythonAnalysisArgs,
    SchemaSearchArgs,
    Text2SQLArgs,
    map_rendering_tool,
    python_analysis_tool_schema,
    schema_search_tool,
    text2sql_tool_schema,
)


class ExecutionRoute(str, Enum):
    """工具实际执行位置：通用 ToolNode 内联调用，或 LangGraph 专用节点。"""

    INLINE = "inline"
    GRAPH_TEXT2SQL = "graph_text2sql"
    GRAPH_PYTHON = "graph_python"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    lc_tool: Any
    execution_route: ExecutionRoute
    args_model: Type[BaseModel] | None


def _desc(tool: Any) -> str:
    d = getattr(tool, "description", None) or ""
    return str(d).strip()


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name=schema_search_tool.name,
        description=_desc(schema_search_tool),
        lc_tool=schema_search_tool,
        execution_route=ExecutionRoute.INLINE,
        args_model=SchemaSearchArgs,
    ),
    ToolSpec(
        name=text2sql_tool_schema.name,
        description=_desc(text2sql_tool_schema),
        lc_tool=text2sql_tool_schema,
        execution_route=ExecutionRoute.GRAPH_TEXT2SQL,
        args_model=Text2SQLArgs,
    ),
    ToolSpec(
        name=python_analysis_tool_schema.name,
        description=_desc(python_analysis_tool_schema),
        lc_tool=python_analysis_tool_schema,
        execution_route=ExecutionRoute.GRAPH_PYTHON,
        args_model=PythonAnalysisArgs,
    ),
    ToolSpec(
        name=map_rendering_tool.name,
        description=_desc(map_rendering_tool),
        lc_tool=map_rendering_tool,
        execution_route=ExecutionRoute.INLINE,
        args_model=MapRenderingArgs,
    ),
]

ALL_TOOLS = [s.lc_tool for s in TOOL_SPECS]
TOOLS_MAP: dict[str, Any] = {s.name: s.lc_tool for s in TOOL_SPECS}
_SPECS_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in TOOL_SPECS}


def tool_names_ordered() -> list[str]:
    return [s.name for s in TOOL_SPECS]


def spec_for_tool_name(name: str) -> ToolSpec | None:
    return _SPECS_BY_NAME.get(name)


def graph_route_name(name: str) -> str | None:
    """供 route_after_agent 使用：返回 LangGraph 边目标，或 None 表示走通用 tools 节点。"""
    spec = spec_for_tool_name(name)
    if not spec:
        return None
    if spec.execution_route == ExecutionRoute.GRAPH_TEXT2SQL:
        return "text2sql"
    if spec.execution_route == ExecutionRoute.GRAPH_PYTHON:
        return "python_analysis"
    return "tools"
