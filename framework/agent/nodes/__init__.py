# -*- coding: utf-8 -*-
"""LangGraph 节点实现（子模块聚合；通过 __getattr__ 转发到 core，避免 `from . import core` 触发递归）。"""

import importlib


def __getattr__(name: str):
    _core = importlib.import_module(".core", __package__)
    return getattr(_core, name)


def __dir__():
    _core = importlib.import_module(".core", __package__)
    return sorted(set(vars(_core).keys()) | {"__getattr__", "__dir__"})
