# -*- coding: utf-8 -*-
"""
Function Calling Master Agent 智能体系统 — 系统入口。
支持交互式 REPL 与单次调用两种模式。每步执行后输出简要进度。

python main.py -q "筛选出与得克萨斯州（Texas）在空间上接壤的所有州，并计算这些州在 2020 年上半年的 NDVI 总和。"
"""
import argparse
import json
import sys
import io
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

import config
from agent.graph import build_graph
from agent.state import create_initial_state

# 修复 Windows 控制台 GBK 编码输出异常
# 仅在交互式终端（isatty）中覆盖，避免干扰管道传输、pytest 捕获或 Web 框架
if (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    and sys.stdout.encoding
    and sys.stdout.encoding.lower() != "utf-8"
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _safe_message_parsed(last_msg: Any) -> dict[str, Any]:
    """读取 AIMessage.additional_kwargs['parsed']，保证为 dict，避免 key 存在但值为 None 时 .get 崩溃。"""
    additional = getattr(last_msg, "additional_kwargs", None) or {}
    if not isinstance(additional, dict):
        return {}
    raw = additional.get("parsed")
    return raw if isinstance(raw, dict) else {}


def _path_display_name(p: Any) -> str:
    """控制台流式日志仅用文件名，避免全路径被模型抄进工具参数。"""
    if p is None:
        return ""
    try:
        return Path(str(p)).name
    except Exception:
        s = str(p).replace("\\", "/").rstrip("/")
        return s.rsplit("/", 1)[-1] if s else ""


def _format_code_output_log_preview(code_output: str, max_chars: int = 300) -> str:
    """脚本 stdout 预览：先截断再展平换行，避免多行导致「第一行很短却已满 max_chars」的误解。"""
    s = str(code_output or "").strip()
    if not s:
        return ""
    truncated = len(s) > max_chars
    head = s[:max_chars].replace("\r", "\\r").replace("\n", "\\n")
    return head + ("...(已截断)" if truncated else "")


def _format_step_output(node_name: str, update: dict[str, Any]) -> str:
    """根据刚完成的节点与更新内容生成单步输出文案。"""
    if not update:
        return ""
    lines = []
    messages = update.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    if node_name == "agent" and messages:
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, AIMessage):
            parsed = _safe_message_parsed(last_msg)
            content = getattr(last_msg, "content", "") or ""
            tool_calls = getattr(last_msg, "tool_calls", None) or parsed.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                tool_calls = []

            if tool_calls:
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    lines.append(f"[Agent] Tool Call: {tool_call.get('name', '?')}")
                    lines.append("  执行参数:")
                    tc_args = tool_call.get("args")
                    if tc_args is None or not isinstance(tc_args, dict):
                        tc_args = {}
                    log_args = dict(tc_args)
                    for _k in ("geojson_paths", "paths"):
                        if _k in log_args and isinstance(log_args[_k], list):
                            log_args[_k] = [_path_display_name(x) for x in log_args[_k]]
                    try:
                        formatted_args = json.dumps(log_args, ensure_ascii=False, indent=2)
                        for a_line in formatted_args.split("\n"):
                            lines.append(f"    {a_line}")
                    except Exception:
                        lines.append(f"    {tc_args}")
            else:
                msg_type = parsed.get("type")
                # 流式阶段不重复打印 final，收尾 run_once/_extract_final_answer 会统一输出一次
                if msg_type == "final":
                    return ""
                output = parsed.get("output", content)
                title = "[Agent] 自然语言响应："
                lines.append(title)
                for o_line in output.split("\n"):
                    lines.append(f"  {o_line}")

    elif node_name == "tools" and messages:
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_name", "tool")
                success = (getattr(msg, "additional_kwargs", {}) or {}).get("success", True)
                payload = (getattr(msg, "additional_kwargs", {}) or {}).get("payload")
                lines.append(f"[Tool] {tool_name} {'成功' if success else '失败'}")
                if isinstance(payload, dict):
                    if tool_name == "schema_search_tool":
                        tables = payload.get("table_names", [])
                        if tables:
                            lines.append(f"  检索到表: {', '.join(tables)}")
                    elif tool_name == "text2sql_tool":
                        errors = payload.get("errors", [])
                        paths = payload.get("geojson_paths", [])
                        if errors:
                            lines.append(f"  错误: {str(errors[-1])[:120]}")
                        elif paths:
                            lines.append(f"  导出文件: {[_path_display_name(p) for p in paths]}")
                    elif tool_name == "python_analysis_tool":
                        preview = _format_code_output_log_preview(
                            str(payload.get("code_output") or payload.get("error") or "")
                        )
                        if preview:
                            lines.append(f"  输出: {preview}")
                        l2c = payload.get("llm2code_info")
                        if isinstance(l2c, dict):
                            stage = l2c.get("stage", "")
                            nfiles = len(l2c.get("workspace_files") or [])
                            nsql = l2c.get("sql_context_entries", "")
                            ncode = l2c.get("generated_code_chars", "")
                            lines.append(
                                f"  llm2code: stage={stage} 文件={nfiles} SQL条目={nsql} 代码字符={ncode}"
                            )
                else:
                    preview = (getattr(msg, "content", "") or "")[:150].replace("\n", " ")
                    if preview:
                        lines.append(f"  {preview}")
    elif node_name == "direct_answer":
        direct_response = update.get("direct_response") or update.get("final_answer") or ""
        messages = update.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        if not direct_response and messages:
            last = messages[-1]
            if isinstance(last, AIMessage):
                direct_response = getattr(last, "content", "") or ""
        if direct_response:
            lines.append("[DirectAnswer] 快速回复：")
            for line in direct_response.split("\n"):
                lines.append(f"  {line}")

    elif node_name == "planner":
        plan_steps = update.get("plan") or []
        if plan_steps:
            lines.append("[Planner] 执行蓝图：")
            for i, step in enumerate(plan_steps, start=1):
                lines.append(f"  步骤{i}: {step}")
        else:
            lines.append("[Planner] 未生成执行蓝图（问题可能不可查询）。")

    elif node_name in {"pre_rag", "text2sql", "python_analysis"}:
        if node_name == "pre_rag":
            slots = update.get("slots") or {}
            retrieved = update.get("retrieved_table_names") or []
            is_queryable = slots.get("is_queryable", True)
            reject_reason = slots.get("reject_reason") or ""
            lines.append(f"[PreRAG] is_queryable={is_queryable}，候选表数量={len(retrieved)}")
            if reject_reason and not is_queryable:
                lines.append(f"  reject_reason: {reject_reason}")
        elif node_name == "text2sql":
            errors = update.get("errors") or []
            geojson_paths = update.get("geojson_paths") or []
            lines.append(f"[Tool] text2sql_tool {'成功' if not errors else '失败'}")
            lines.append(f"  导出文件数: {len(geojson_paths)}")
            if geojson_paths and not errors:
                lines.append(f"  导出文件: {[_path_display_name(p) for p in geojson_paths]}")
            if errors:
                lines.append("[Text2SQL] 执行报错:")
                for idx, err in enumerate(errors, start=1):
                    lines.append(f"  错误{idx}: {str(err)}")
        else:
            errors = update.get("errors") or []
            code_output = str(update.get("code_output") or "").strip()
            lines.append(f"[Tool] python_analysis_tool {'成功' if not errors else '失败'}")
            lines.append(f"  脚本输出长度: {len(code_output)}")
            if code_output:
                lines.append(f"  输出预览: {_format_code_output_log_preview(code_output)}")
            l2c = update.get("llm2code_info")
            if isinstance(l2c, dict):
                lines.append(
                    f"  llm2code: stage={l2c.get('stage')} 文件={len(l2c.get('workspace_files') or [])} "
                    f"SQL条目={l2c.get('sql_context_entries')} 代码字符={l2c.get('generated_code_chars')}"
                )
            if errors:
                for idx, err in enumerate(errors, start=1):
                    lines.append(f"  错误{idx}: {str(err)}")

    return "\n".join(lines) if lines else ""


def _snapshot_graph_state(graph, run_config: dict) -> dict[str, Any]:
    """从 LangGraph checkpointer 读取 Reducer 合并后的完整 state（勿用手动 {**state, **update}）。"""
    try:
        snap = graph.get_state(run_config)
        if snap and snap.values:
            return dict(snap.values)
    except Exception:
        pass
    return {}


def _unlink_checkpoint_workspace_files(geojson_paths: list[str] | None) -> None:
    """删除上一回合 checkpoint 中记录的、且位于工作区内的导出文件，避免沙盒读到陈旧 GeoJSON。"""
    ws = Path(config.WORKSPACE_DIR).resolve()
    for raw in geojson_paths or []:
        p = Path(str(raw))
        if not p.is_absolute():
            p = (ws / p).resolve()
        try:
            p.relative_to(ws)
        except ValueError:
            continue
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _run_streaming(
    graph,
    state_update: dict[str, Any],
    run_config: dict,
    verbose: bool = True,
    on_step: Callable[[str, dict[str, Any]], None] | None = None,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], Exception | None]:
    """执行 Agent 图并流式打印每步输出，返回 (最终 state, 若中途异常则为该异常否则 None)。

    最终 state 来自 graph.get_state（Reducer 后的全量），stream 的 update 仅为增量，不可浅拷贝合并。
    """
    caught: Exception | None = None
    state: dict[str, Any] = {}
    try:
        for chunk in graph.stream(state_update, config=run_config):
            if on_chunk is not None:
                try:
                    on_chunk(chunk)
                except Exception:
                    pass
            for node_name, update in chunk.items():
                if on_step is not None and node_name:
                    try:
                        on_step(node_name, update if isinstance(update, dict) else {})
                    except Exception:
                        pass
                if verbose and node_name:
                    msg = _format_step_output(node_name, update)
                    if msg:
                        print(msg + "\n")
        state = _snapshot_graph_state(graph, run_config)
    except Exception as e:
        caught = e
        err_type = type(e).__name__
        err_msg = str(e).strip() or "（无详细信息）"
        if verbose:
            print(f"执行出错 [{err_type}]：{err_msg}\n", file=sys.stderr)
        fa = f"执行出错 [{err_type}]：{err_msg}"
        if err_type == "GraphRecursionError":
            fa += (
                "\n\n提示：多步蓝图未完成时易触发递归上限；请确认各步按蓝图使用 text2sql_tool / python_analysis_tool。"
                "可通过环境变量 RECURSION_LIMIT 临时提高上限（默认见 config.RECURSION_LIMIT，治标不治本）。"
            )
        state = _snapshot_graph_state(graph, run_config)
        state["final_answer"] = fa
    return state, caught


def run_graph_for_question(
    question: str,
    *,
    verbose: bool = True,
    on_step: Callable[[str, dict[str, Any]], None] | None = None,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], Exception | None]:
    """构建图并以独立 thread 跑完一轮，返回 (最终 state, stream 中捕获的异常或 None)。

    使用 MemorySaver 以便 stream 结束后 get_state 可取 Reducer 合并后的全量 state。
    on_chunk 在每个 stream chunk 上调用一次；on_step(node_name, update) 在每步增量上调用。
    """
    graph = build_graph(checkpointer=MemorySaver())
    state_update = create_initial_state(question)
    run_config = {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": config.RECURSION_LIMIT,
    }
    return _run_streaming(
        graph,
        state_update,
        run_config,
        verbose=verbose,
        on_step=on_step,
        on_chunk=on_chunk,
    )


def _extract_final_answer(state: dict[str, Any]) -> str:
    """从 state 中提取最终回答。"""
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            parsed = _safe_message_parsed(msg)
            if parsed.get("type") == "final":
                output = parsed.get("output", "").strip()
                if output:
                    return output
            content = getattr(msg, "content", "") or ""
            if content.strip() and not getattr(msg, "tool_calls", None):
                return content.strip()
    return state.get("final_answer") or "（无回答）"


def run_once(question: str, verbose: bool = True) -> str:
    """单次调用：不启用多轮记忆（每问独立 thread_id）。"""
    if verbose:
        print("问题：" + question + "\n")
    state, _ = run_graph_for_question(question, verbose=verbose)
    answer = _extract_final_answer(state)
    if verbose:
        print("--- 最终回答 ---\n" + answer)
    return answer


def run_repl():
    """交互式 REPL：MemorySaver + 固定 thread_id 保留 messages；每轮重置临时字段并清理上一回合工作区导出文件。"""
    print("Function Calling Master Agent 已启动。输入地理分析问题后回车，输入 quit 或 exit 退出。\n")

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    thread_id = uuid.uuid4().hex
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.RECURSION_LIMIT,
    }

    while True:
        try:
            question = input("请输入你的地理分析问题： ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见。")
            break

        try:
            snap = graph.get_state(run_config)
            if snap.values:
                _unlink_checkpoint_workspace_files(snap.values.get("geojson_paths"))
        except Exception:
            pass

        state_update = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "plan": [],
            "plan_meta": [],
            "current_plan_step": None,
            "current_plan_step_index": None,
            "geojson_paths": [],
            "latest_text2sql_geojson_paths": [],
            "sql_queries": [],
            "sql_results": [],
            "schemas": [],
            "schemas_yaml": "",
            "retrieved_table_names": [],
            "slots": None,
            "code": None,
            "code_output": None,
            "errors": [],
            "retry_count": 0,
            "final_answer": None,
            "direct_response": None,
            "map_path": None,
            "python_analysis_contract": None,
            "step_failure_counts": {},
            "last_failure_type": None,
            "last_failure_step_index": None,
            "text2sql_schema_cache": None,
        }

        state, _ = _run_streaming(graph, state_update, run_config, verbose=True)
        answer = _extract_final_answer(state)
        print("\n--- 回答 ---\n" + answer + "\n")


def main():
    parser = argparse.ArgumentParser(description="Function Calling Master Agent 智能体系统")
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="单次调用时的问题文本；不传则进入交互模式",
    )
    args = parser.parse_args()
    if args.question is not None:
        run_once(args.question)
    else:
        run_repl()


if __name__ == "__main__":
    main()
