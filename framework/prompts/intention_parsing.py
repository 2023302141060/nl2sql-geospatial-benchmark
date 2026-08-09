# -*- coding: utf-8 -*-
"""轻量意图槽位提示与 Pre-RAG 可观测性。"""
from __future__ import annotations

import json
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


INTENTION_PARSING_SYSTEM = """你是 GIS 查询意图解析器。只判断问题是否属于本地时空数据查询，并提取结构化槽位；不要改写问题，不要选择物理表，不要生成 SQL 或 Python。

判定规则：
- 地理范围内的检索、统计、比较、排名、时序、空间拓扑或遥感指标分析，默认 is_queryable=true。
- 仅闲聊、元问题、通用代码生成或完全无地理/时空/指标分析语义时，is_queryable=false。
- 不得因为不知道某个字段是否存在而拒绝；Schema 选择由后续 RAG 负责。

提取原则：
- 忠实保留问题中的区域、时间、指标、计算方式、空间谓词、阈值、分析方法和可视化要求。
- 不补造用户未给出的年份、城市或指标；经纬度保持原值，不猜所在城市。
- 多区域、多时间、多指标分别写入 region_set、time_range、metric_set。
- 无值使用空字符串、空数组或空对象，不使用 null。

输出由结构化 Schema 约束，只返回槽位对象。"""


INTENTION_PARSING_USER = """用户问题：
{question}

请判断可查询性并提取槽位。"""


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def get_intention_parsing_messages(question: str) -> tuple[str, str]:
    """返回不携带全库 Schema 的轻量意图解析消息。"""
    return (
        INTENTION_PARSING_SYSTEM,
        INTENTION_PARSING_USER.format(question=escape_braces(question)),
    )


def format_intention_slots_for_display(slots_dict: dict[str, Any] | None) -> str:
    lines = ["  [意图槽位]"]
    if not slots_dict:
        lines.append("    （空）")
        return "\n".join(lines)
    dumped = json.dumps(slots_dict, ensure_ascii=False, indent=2)
    lines.extend(f"    {row}" for row in dumped.splitlines())
    return "\n".join(lines)


def _column_names_from_schema_data(data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for col in data.get("columns") or []:
        if not isinstance(col, dict):
            continue
        name = col.get("name") or col.get("column_name")
        if str(name or "").strip():
            names.append(str(name).strip())
    return names


def format_retrieved_tables_and_columns(schema_bundle: dict[str, Any] | None) -> str:
    lines = ["  [RAG 检索表与字段]"]
    if not schema_bundle:
        return "\n".join([*lines, "    （无）"])
    table_names = schema_bundle.get("table_names") or []
    schemas = schema_bundle.get("schemas") or []
    for index, yaml_text in enumerate(schemas):
        table_name = str(table_names[index]).strip() if index < len(table_names) else f"表{index + 1}"
        columns: list[str] = []
        try:
            data = yaml.safe_load(yaml_text) or {}
            if isinstance(data, dict):
                table_name = str(data.get("table_name") or table_name).strip()
                columns = _column_names_from_schema_data(data)
        except Exception as exc:
            logger.debug("解析 RAG schema YAML 失败: %s", exc, exc_info=True)
        lines.append(f"    · {table_name}: {', '.join(columns) if columns else '（未解析列名）'}")
    if len(table_names) > len(schemas):
        lines.extend(f"    · {name}: （无对应 YAML）" for name in table_names[len(schemas):])
    return "\n".join(lines)


def print_intention_rag_observability(
    slots_dict: dict[str, Any] | None,
    schema_bundle: dict[str, Any] | None = None,
    *,
    flush: bool = True,
) -> None:
    print(format_intention_slots_for_display(slots_dict), flush=flush)
    if schema_bundle is not None:
        print(format_retrieved_tables_and_columns(schema_bundle), flush=flush)
        bindings = schema_bundle.get("semantic_bindings") or []
        coverage = schema_bundle.get("schema_coverage") or {}
        print("  [Schema 语义绑定]", flush=flush)
        if bindings:
            for item in bindings:
                if not isinstance(item, dict):
                    continue
                columns = ", ".join(str(v) for v in (item.get("columns") or [])) or "（表级）"
                print(
                    f"    · {item.get('concept')}: {item.get('table')}[{columns}]",
                    flush=flush,
                )
        else:
            print("    （未形成显式绑定，保留向量候选）", flush=flush)
        if coverage:
            print(f"  [Schema 覆盖状态] {json.dumps(coverage, ensure_ascii=False)}", flush=flush)
