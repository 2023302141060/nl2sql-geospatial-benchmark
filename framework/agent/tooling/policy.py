# -*- coding: utf-8 -*-
"""工具调用策略：参数扁平化、Schema 预校验、护栏参数提示（与注册表一致）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from agent.tooling.registry import spec_for_tool_name


def flatten_nested_tool_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """剥离模型幻觉产生的 parameters/args 套壳，合并为单层字典。"""
    if not isinstance(tool_args, dict):
        return {}
    merged = {k: v for k, v in tool_args.items() if k not in ("parameters", "args")}
    for shell in ("parameters", "args"):
        inner = tool_args.get(shell)
        if isinstance(inner, dict):
            for k, v in inner.items():
                if k not in merged or merged.get(k) in (None, "", [], {}):
                    merged[k] = v
    return merged


def validate_tool_args_dict(tool_name: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """按注册表 Pydantic 模型校验参数。成功返回 (model_dump 风格 dict, None)，失败返回 (None, 错误说明)。"""
    spec = spec_for_tool_name(tool_name)
    if not spec or spec.args_model is None:
        return dict(args), None
    model_cls: type[BaseModel] = spec.args_model
    flat = flatten_nested_tool_args(args)
    try:
        validated = model_cls.model_validate(flat)
        return validated.model_dump(), None
    except ValidationError as e:
        return None, f"参数不符合 `{tool_name}` 的约定：{e!s}"
