# -*- coding: utf-8 -*-
"""Executor 侧轻量文本守卫（避免测试 import agent.nodes 时拉起 geopandas 等重依赖）。"""

# 模型把工具参数 JSON 当普通文本输出时的强特征（非原生 tool_calls）
TOOL_PARAM_LEAK_MARKERS = (
    '"queries": [',
    '"sql": "SELECT',
    '"sql": "select',
    '"output_filename":',
    '"geojson_paths":',
)


def text_contains_tool_param_leak(text: str) -> bool:
    return bool(text) and any(m in text for m in TOOL_PARAM_LEAK_MARKERS)
