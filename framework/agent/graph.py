# -*- coding: utf-8 -*-
"""论文主架构的 LangGraph 实现（拓扑保持稳定）。

节点与论文模块的对应关系：
- ``pre_rag``：Intent Understanding + Schema Pre-filtering；
- ``planner``：Planner Agent；
- 路由、结构化 ``AgentState`` 与 ``review``：Execution State Manager；
- ``text2sql``：SCGA（SQL Code Generation Agent）；
- ``python_analysis``：STCA（Spatio-Temporal Computation Agent）。

V2 优化只允许发生在模块内部及结构化状态通信上，不通过增删上述主模块
改变论文所描述的控制流。
"""
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import (
    intention_and_rag_node,
    direct_answer_node,
    planner_node,
    agent_node,
    route_after_agent,
    tool_node,
    text2sql_node,
    python_analysis_node,
)
from agent.state import AgentState
from agent.reviewer import final_evidence_review_node, route_after_review


def is_queryable_router(state: dict[str, Any]) -> str:
    """条件路由：根据意图解析结果决定走快速通道还是进入规划-执行流程。

    - is_queryable=False → direct_answer（快速通道，直接生成友好回复后结束）
    - is_queryable=True  → planner（进入规划师节点，生成执行蓝图）
    """
    slots = state.get("slots") or {}
    if slots.get("is_queryable") is False:
        return "direct_answer"
    return "planner"


def build_graph(checkpointer=None):
    """构建并编译基于 Plan-and-Execute 架构（含快速通道）的状态图。

    图结构：
        START → pre_rag
                  ├─(is_queryable=False)→ direct_answer → END
                  └─(is_queryable=True) → planner → agent → tools → agent
                                                        └→ review → END / agent / planner

    候选答案必须先经过 Execution State Manager 中的独立证据审查；审查器
    可以要求改写、局部修复或一次重规划。该 review 节点是状态管理内部的
    结果评估/错误处理环节，不是论文架构之外新增的业务 Agent。
    """
    graph = StateGraph(AgentState)

    graph.add_node("pre_rag", intention_and_rag_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("text2sql", text2sql_node)
    graph.add_node("python_analysis", python_analysis_node)
    graph.add_node("review", final_evidence_review_node)

    graph.set_entry_point("pre_rag")

    # 条件路由：pre_rag 完成后根据 is_queryable 决定走快速通道还是规划流程
    graph.add_conditional_edges(
        "pre_rag",
        is_queryable_router,
        {
            "direct_answer": "direct_answer",
            "planner": "planner",
        },
    )

    # 快速通道：direct_answer 直接结束
    graph.add_edge("direct_answer", END)

    # 规划-执行流程：planner → agent → 按 tool_calls 路由工具节点
    graph.add_edge("planner", "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "text2sql": "text2sql",
            "python_analysis": "python_analysis",
            "review": "review",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("text2sql", "agent")
    graph.add_edge("python_analysis", "agent")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "planner": "planner",
            "agent": "agent",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
