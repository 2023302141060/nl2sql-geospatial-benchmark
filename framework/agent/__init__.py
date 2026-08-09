# -*- coding: utf-8 -*-
"""ReAct Agent：Master Agent 自主调度工具的智能体系统。"""
from agent.state import AgentState, IntentionSlots, create_initial_state

__all__ = ["AgentState", "IntentionSlots", "create_initial_state", "build_graph"]


def __getattr__(name: str):
    if name == "build_graph":
        from agent.graph import build_graph as _build_graph

        return _build_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
