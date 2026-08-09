# -*- coding: utf-8 -*-
"""执行契约：只从结构化 PlanBlueprint 与意图槽位生成。"""
from __future__ import annotations

from typing import Any

from agent.answer_contract import build_answer_projection_contract


CONTRACT_VERSION = 1


def empty_execution_contract(*, queryable: bool = True) -> dict[str, Any]:
    """不可查询或尚无蓝图时的空契约。"""
    return {
        "contract_version": CONTRACT_VERSION,
        "queryable": bool(queryable),
        "requires_python": False,
        "requires_geometry": False,
        "input_mode": "na",
        "entity_level": "unknown",
        "display_fields": [],
        "operation_type": "unknown",
        "time_comparison_mode": "none",
        "answer_projection": {},
        "required_tables": [],
        "required_geometry_tables": [],
        "schema_bindings": [],
        "schema_coverage": {},
        "condition_clauses": [],
    }


def _normalize_input_mode(raw: str) -> str:
    if raw in ("single_file", "multi_file", "either", "na"):
        return raw
    return "na"


def _infer_time_comparison_mode(slots: dict[str, Any], planning_query: str) -> str:
    tr = slots.get("time_range")
    if isinstance(tr, list) and len(tr) >= 2:
        if str(tr[0]).strip() and str(tr[-1]).strip() and str(tr[0]).strip() != str(tr[-1]).strip():
            return "compare_two_times"
    blob = str(planning_query or "").lower()
    if any(
        k in blob
        for k in (
            "逐年",
            "年际",
            "时间序列",
            "序列差分",
            "变迁",
            "difference over time",
            "year over year",
        )
    ):
        return "difference_over_sequence"
    return "none"


def _infer_entity_level(slots: dict[str, Any], planning_query: str) -> str:
    blob = f"{planning_query} {(slots.get('region') or '')}".lower()
    if any(k in blob for k in ("州", "省级", "province", "u.s. state", "us state")):
        return "state"
    if any(k in blob for k in ("县", "区", "县级", "county")):
        return "county"
    if any(k in blob for k in ("网格", "栅格", "cell", "grid", "像元")):
        return "grid"
    return "unknown"


def _infer_operation_type(slots: dict[str, Any], has_python: bool, requires_geometry: bool) -> str:
    if str(slots.get("spatial_predicate") or "").strip() or str(slots.get("spatial_threshold") or "").strip():
        return "spatial_topology"
    am = str(slots.get("analytical_method") or "").strip().lower()
    if am:
        if am in ("correlation", "regression", "clustering", "zonal_statistics", "minimum_bounding"):
            return am
        return "other"
    if has_python and not requires_geometry:
        return "aggregate"
    if has_python:
        return "other"
    return "unknown"


def _default_display_fields(entity_level: str) -> list[str]:
    if entity_level == "state":
        return ["shapeName", "state_name", "name"]
    return []


def build_execution_contract_from_plan(
    *,
    planning_query: str,
    slots: dict[str, Any] | None,
    plan_meta: list[dict[str, Any]],
    schema_bindings: list[dict[str, Any]] | None = None,
    required_tables: list[str] | None = None,
    required_geometry_tables: list[str] | None = None,
    schema_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据权威结构化 plan_meta 生成执行契约。"""
    slots_d = slots if isinstance(slots, dict) else {}
    has_python = any(
        isinstance(item, dict) and item.get("tool") == "python_analysis"
        for item in plan_meta
    )
    py_meta: dict[str, Any] | None = None
    for m in reversed(plan_meta or []):
        if isinstance(m, dict) and m.get("tool") == "python_analysis":
            py_meta = m
            break
    requires_geometry = bool(py_meta.get("requires_geometry")) if py_meta else False
    input_mode = _normalize_input_mode(str(py_meta.get("expected_input_mode") or "na")) if py_meta else "na"
    time_comparison_mode = _infer_time_comparison_mode(slots_d, planning_query)
    entity_level = _infer_entity_level(slots_d, planning_query)
    operation_type = _infer_operation_type(slots_d, has_python, requires_geometry)
    display_fields = list(_default_display_fields(entity_level))
    condition = slots_d.get("condition") if isinstance(slots_d.get("condition"), dict) else {}
    condition_clauses = [
        str(item).strip()
        for item in (condition.get("clauses") or [])
        if str(item).strip()
    ]

    return {
        "contract_version": CONTRACT_VERSION,
        "queryable": True,
        "requires_python": has_python,
        "requires_geometry": requires_geometry,
        "input_mode": input_mode,
        "entity_level": entity_level,
        "display_fields": display_fields,
        "operation_type": operation_type,
        "time_comparison_mode": time_comparison_mode,
        "answer_projection": build_answer_projection_contract(planning_query, slots_d),
        "required_tables": list(required_tables or []),
        "required_geometry_tables": list(required_geometry_tables or []),
        "schema_bindings": [item for item in (schema_bindings or []) if isinstance(item, dict)],
        "schema_coverage": dict(schema_coverage or {}),
        "condition_clauses": list(dict.fromkeys(condition_clauses)),
    }
