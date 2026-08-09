# -*- coding: utf-8 -*-
"""Generic evidence review and bounded replanning for the hybrid agent.

The reviewer never sees benchmark gold answers.  It checks only whether the
candidate answer is supported by tool observations and covers the user's
request.  A rejected answer can be revised, repaired with one extra tool step,
or sent back to the planner once.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

import config
from agent.answer_schema import build_strict_answer, build_strict_answer_from_sql_payload


class FinalEvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "revise", "repair", "replan"]
    reason: str = Field(min_length=1, max_length=800)
    missing_requirements: list[str] = Field(default_factory=list, max_length=5)
    recommended_tool: Literal[
        "none", "text2sql_tool", "python_analysis_tool"
    ] = "none"
    repair_objective: str = Field(default="", max_length=800)


FINAL_EVIDENCE_REVIEW_SYSTEM = """你是独立的 GIS Agent 证据审查器。你只能根据用户问题、执行计划、候选答案和真实工具返回进行判断，不能使用外部常识猜答案，也不能接触或推测基准 Gold。

检查四件事：
1. 覆盖性：问题要求的实体、指标、时间、空间范围、统计方法、单位、排序或 Top-K 是否都已回答；
2. 支持性：候选中的每个关键实体和数值是否能在工具证据中找到来源；
3. 结构性：实体粒度、列表数量与顺序、分组键以及数值单位是否与问题一致；
4. 执行充分性：计划的成功条件是否满足，复杂计算是否真的经过相应工具执行。

最低可追溯性要求同样适用于所有领域和指标：
- 最高/最低/Top-K 等比较结果同时保留实体标识和用于比较的指标值；
- 先筛选实体再对该集合聚合时，同时保留入选实体数量（集合很小时也可保留完整实体列表）；
- 当问题给出了具体指标或统计语义时，字段名应表达该语义，避免只用 `value`、`count`、`result` 等无法独立解释的通用键。

决策规则：
- accept：证据足够且答案完整；措辞不同、字段别名或无关格式差异不能成为拒绝理由。
- revise：证据已经足够，只需重新组织或补齐候选答案，不需要再调用工具。
- repair：只缺少一次局部取数或计算；给出 recommended_tool 和不包含代码的 repair_objective。
- replan：计划的能力路由、数据依赖或任务分解本身错误，局部一步无法修复。

只有能指出具体缺失或矛盾时才能拒绝。不要要求问题没有提出的附加字段、解释、文件或可视化。"""


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Bound evidence size while retaining beginnings and endings of records."""
    if depth >= 4:
        return str(value)[:300]
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) > 14:
            items = items[:10] + items[-2:]
        return {str(k): _compact_value(v, depth=depth + 1) for k, v in items}
    if isinstance(value, list):
        values = value if len(value) <= 9 else value[:7] + value[-2:]
        return [_compact_value(v, depth=depth + 1) for v in values]
    if isinstance(value, str):
        return value if len(value) <= 700 else value[:520] + "…" + value[-120:]
    return value


def _tool_evidence(messages: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for msg in messages[-12:]:
        if not isinstance(msg, ToolMessage):
            continue
        extra = getattr(msg, "additional_kwargs", {}) or {}
        item = {
            "tool": extra.get("tool_name") or getattr(msg, "name", ""),
            "success": extra.get("success"),
            "failure_code": extra.get("failure_code"),
            "payload": _compact_value(extra.get("payload")),
        }
        # ToolMessage.content 通常只是 payload 的序列化副本；仅在没有结构化
        # payload 时保留，避免审查阶段把同一证据发送两遍。
        if extra.get("payload") is None:
            item["content"] = _compact_value(str(getattr(msg, "content", "") or ""))
        evidence.append(item)
    return evidence[-4:]


def _review_prompt(state: dict[str, Any]) -> str:
    execution_contract = state.get("execution_contract")
    contract_summary = {}
    if isinstance(execution_contract, dict):
        contract_summary = {
            "requires_python": execution_contract.get("requires_python"),
            "requires_geometry": execution_contract.get("requires_geometry"),
            "answer_projection": execution_contract.get("answer_projection") or {},
            "schema_bindings": execution_contract.get("schema_bindings") or [],
            "schema_coverage": execution_contract.get("schema_coverage") or {},
            "condition_clauses": execution_contract.get("condition_clauses") or [],
        }
    return json.dumps(
        {
            "question": state.get("question") or "",
            "plan_meta": _compact_value(state.get("plan_meta") or []),
            "execution_contract": _compact_value(contract_summary),
            "candidate_answer": _compact_value(
                state.get("final_answer_payload") or state.get("final_answer")
            ),
            "candidate_schema_valid": bool(state.get("final_answer_schema_valid")),
            "tool_evidence": _tool_evidence(state.get("messages") or []),
        },
        ensure_ascii=False,
    )


def semantic_projection_violations(question: str, payload: Any) -> list[str]:
    """Generic, answer-value-independent checks for lossy evidence projection."""
    violations: list[str] = []
    q = str(question or "").lower()

    records: list[dict[str, Any]] = []

    def collect_records(value: Any) -> None:
        if isinstance(value, dict):
            nested = value.get("data_payload") if "data_payload" in value else None
            if nested is not None:
                collect_records(nested)
                return
            if any(not isinstance(v, (dict, list)) for v in value.values()):
                records.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    collect_records(child)
        elif isinstance(value, list):
            for child in value:
                collect_records(child)

    collect_records(payload)

    generic_keys = {"value", "count", "result", "数值", "数量", "结果"}
    if any(str(k).strip().lower() in generic_keys for row in records for k in row):
        violations.append("字段名过于通用，未表达问题中的具体指标或统计语义")

    asks_extreme = bool(re.search(r"最高|最低|最大|最小|最多|最少|top\s*\d+|前\s*\d+", q))
    if asks_extreme and records:
        for row in records:
            keys = [str(k).lower() for k in row]
            entity_keys = [
                k for k in keys
                if re.search(r"(?:^|_)(?:id|name|entity|cell|state|city|county|province)(?:$|_)", k)
                or any(token in k for token in ("网格", "名称", "州", "城市", "实体"))
            ]
            metric_values = [
                v for k, v in row.items()
                if str(k).lower() not in entity_keys
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            ]
            if entity_keys and not metric_values:
                violations.append("极值或排名结果缺少用于比较的指标值")
                break

    if re.search(r"绝对值|绝对相关|绝对.*系数|absolute", q) and re.search(
        r"相关|correlation|斜率|slope", q
    ):
        statistic_rows = [
            row
            for row in records
            if any(re.search(r"corr|correlation|slope|相关|斜率", str(k).lower()) for k in row)
        ]
        for row in statistic_rows:
            has_raw_signed_key = any(
                re.search(r"corr|correlation|slope|相关|斜率", str(k).lower())
                and not re.search(r"abs|absolute|绝对", str(k).lower())
                for k in row
            )
            if not has_raw_signed_key:
                violations.append("按绝对值排序时丢失了原始带符号统计量")
                break

    if re.search(r"(?:按|根据).{1,24}(?:分组|分为|分成|等频|统计各组)", q):
        if any(str(k).lower() in {"group", "group_id", "category_id"} for row in records for k in row):
            violations.append("分组键未表达实际分组维度")

    return list(dict.fromkeys(violations))


def contract_projection_violations(
    question: str,
    payload: Any,
    answer_projection: dict[str, Any] | None,
) -> list[str]:
    """Check candidate shape against the question-derived answer contract."""
    contract = answer_projection if isinstance(answer_projection, dict) else {}
    shape = str(contract.get("output_shape") or "")
    q = str(question or "").lower()
    root = payload.get("data_payload") if isinstance(payload, dict) and "data_payload" in payload else payload

    nested_lists: list[list[Any]] = []
    nested_keys: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                nested_keys.append(str(key).lower())
                walk(child)
        elif isinstance(value, list):
            nested_lists.append(value)
            for child in value:
                walk(child)

    walk(root)
    issues: list[str] = []
    # A list nested inside one record (for example the qualifying entities that
    # produced a bounding box) is still one record.  Only multiple *top-level*
    # records violate the single-record contract.
    if shape == "single_record":
        has_multiple_root_records = isinstance(root, list) and len(root) > 1
        has_nested_record_series = isinstance(root, dict) and any(
            isinstance(items, list)
            and len(items) > 1
            and all(isinstance(item, dict) for item in items)
            for items in root.values()
        )
        if has_multiple_root_records or has_nested_record_series:
            issues.append("问题只要求单个最优结果，但候选仍携带完整序列或多条记录")

    asks_complete_records = bool(
        re.search(r"(?:输出|列出|返回).{0,12}(?:每个|所有|全部|完整)", q)
        or re.search(r"(?:每个|所有|全部).{0,20}(?:输出|列出|返回)", q)
    )
    if asks_complete_records and any(
        re.search(r"preview|sample|示例|预览|抽样", key) for key in nested_keys
    ):
        issues.append("问题要求完整记录，但候选只返回 preview/sample")

    top_k = contract.get("top_k")
    if shape == "ranked_records" and isinstance(top_k, int) and top_k > 0:
        record_lists = [
            items for items in nested_lists
            if items and all(isinstance(item, dict) for item in items)
        ]
        if record_lists and max(len(items) for items in record_lists) != top_k:
            issues.append(f"排名结果数量与答案契约 Top-{top_k} 不一致")
    canonical_fields = {
        str(field).strip().lower()
        for field in (contract.get("canonical_statistic_fields") or [])
        if str(field).strip()
    }
    if canonical_fields:
        present_fields: set[str] = set()

        def collect_fields(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    present_fields.add(str(key).strip().lower())
                    collect_fields(child)
            elif isinstance(value, list):
                for child in value:
                    collect_fields(child)

        collect_fields(root)
        missing_fields = sorted(canonical_fields - present_fields)
        if missing_fields:
            issues.append(
                "请求的统计量缺少稳定协议字段：" + ", ".join(missing_fields)
            )
    if contract.get("return_each_requested_metric"):
        requested_metrics = [
            str(item).strip().lower()
            for item in (contract.get("requested_metrics") or [])
            if str(item).strip()
        ]
        present_fields: list[str] = []

        def collect_metric_fields(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    present_fields.append(str(key).strip().lower())
                    collect_metric_fields(child)
            elif isinstance(value, list):
                for child in value:
                    collect_metric_fields(child)

        collect_metric_fields(root)
        present_blob = " ".join(present_fields)
        ignored_tokens = {"mean", "sum", "avg", "value", "result", "score", "zscore"}
        missing_metrics: list[str] = []
        for metric in requested_metrics:
            tokens = [
                token
                for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", metric)
                if len(token) >= 3 and token not in ignored_tokens
            ]
            if tokens and not any(token in present_blob for token in tokens):
                missing_metrics.append(metric)
        if missing_metrics:
            issues.append("请求分别返回的指标缺失：" + ", ".join(missing_metrics))
    return issues


def _candidate_equals_typed_sql_evidence(state: dict[str, Any]) -> bool:
    """Whether the final payload is a lossless deterministic SCGA projection."""
    execution_contract = state.get("execution_contract")
    if not isinstance(execution_contract, dict):
        return False
    coverage = execution_contract.get("schema_coverage")
    bindings = execution_contract.get("schema_bindings")
    if not isinstance(coverage, dict) or not coverage.get("complete"):
        return False
    if list(coverage.get("uncovered_metrics") or []) or not list(bindings or []):
        return False

    for message in reversed(state.get("messages") or []):
        if not isinstance(message, ToolMessage):
            continue
        additional = getattr(message, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != "text2sql_tool" or additional.get("success") is not True:
            continue
        payload = additional.get("payload")
        if not isinstance(payload, dict):
            return False
        results = [item for item in (payload.get("sql_results") or []) if isinstance(item, dict)]
        if len(results) != 1 or "data_payload" not in results[0]:
            return False
        expected = build_strict_answer_from_sql_payload(
            results[0].get("data_payload"),
            question=str(state.get("question") or ""),
        )
        actual = state.get("final_answer_payload")
        return isinstance(actual, dict) and actual.get("data_payload") == expected.data_payload
    return False


def _candidate_equals_typed_python_evidence(state: dict[str, Any]) -> bool:
    """Whether STCA's typed payload reached the final protocol losslessly."""
    for message in reversed(state.get("messages") or []):
        if not isinstance(message, ToolMessage):
            continue
        additional = getattr(message, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != "python_analysis_tool":
            continue
        if additional.get("success") is not True:
            continue
        payload = additional.get("payload")
        if not isinstance(payload, dict) or "data_payload" not in payload:
            return False
        expected = build_strict_answer(
            payload.get("data_payload"),
            question=str(state.get("question") or ""),
        )
        actual = state.get("final_answer_payload")
        return bool(
            isinstance(actual, dict)
            and actual.get("answer_type") == expected.answer_type
            and actual.get("data_payload") == expected.data_payload
        )
    return False


def _sanitize_review(review: FinalEvidenceReview) -> FinalEvidenceReview:
    """Bound gateway repetition without changing the review decision."""
    reason = str(review.reason or "").strip()
    for marker in ("}{", "} }", "这个对吗", "返回对", " 1 2 3 4 5"):
        pos = reason.find(marker)
        if pos > 0:
            reason = reason[:pos].rstrip(" }；;")
    reason = reason[:500].strip() or "证据审查已完成。"
    repair = str(review.repair_objective or "").strip()[:500]
    return review.model_copy(update={"reason": reason, "repair_objective": repair})


def _control_feedback(review: FinalEvidenceReview) -> HumanMessage:
    missing = "；".join(review.missing_requirements) or "未明确列出"
    content = (
        "【独立证据审查反馈】\n"
        f"结论：{review.decision}\n"
        f"原因：{review.reason}\n"
        f"缺失项：{missing}\n"
        f"修复目标：{review.repair_objective or '请仅依据已有证据重新组织完整答案。'}"
    )
    return HumanMessage(content=content, additional_kwargs={"control_message": True})


def _repair_step_updates(
    state: dict[str, Any], review: FinalEvidenceReview
) -> dict[str, Any]:
    tool_name = review.recommended_tool
    if tool_name not in {"text2sql_tool", "python_analysis_tool"}:
        return {"review_action": "replan"}
    if tool_name == "python_analysis_tool" and not list(state.get("geojson_paths") or []):
        return {"review_action": "replan"}

    objective = review.repair_objective.strip() or review.reason.strip()
    if tool_name == "python_analysis_tool":
        original_question = str(state.get("question") or "").strip()
        artifacts = [
            str(item).strip()
            for item in (state.get("analysis_artifact_paths") or [])
            if str(item).strip()
        ]
        if artifacts:
            objective += (
                "；优先读取并复用已有 STCA 中间产物，且不要覆盖这些输入文件："
                + ", ".join(artifacts)
            )
        if original_question:
            objective += "；修复时仍须完整满足原问题：" + original_question
    step = f"调用 {tool_name}：{objective}"
    plan = list(state.get("plan") or []) + [step]
    plan_meta = list(state.get("plan_meta") or [])
    plan_meta.append(
        {
            "tool": "text2sql" if tool_name == "text2sql_tool" else "python_analysis",
            "intent": objective,
            "objective": objective,
            "success_criteria": list(review.missing_requirements),
            "needs_multiple_exports": False,
            "final_python_step": tool_name == "python_analysis_tool",
            "requires_python_stats": tool_name == "python_analysis_tool",
            "accept_single_wide_table": True,
            "expected_input_mode": "either",
            "requires_geometry": bool(
                (state.get("execution_contract") or {}).get("requires_geometry")
                if isinstance(state.get("execution_contract"), dict)
                else False
            ),
            "preferred_output_type": "either",
            "review_generated": True,
        }
    )
    return {
        "plan": plan,
        "plan_meta": plan_meta,
        "current_plan_step_index": len(plan),
        "current_plan_step": step,
        "review_action": "repair",
    }


def final_evidence_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Review a candidate and request at most a bounded generic correction."""
    count = int(state.get("review_count") or 0)
    max_reviews = int(getattr(config, "V2_MAX_EVIDENCE_REVIEWS", 2) or 2)
    if not bool(getattr(config, "V2_ENABLE_EVIDENCE_REVIEW", True)):
        return {"review_action": "accept"}
    if count >= max_reviews:
        print("  [review] 已达到证据审查上限，保留当前结构化候选答案。", flush=True)
        return {"review_action": "accept"}

    answer_projection = (
        (state.get("execution_contract") or {}).get("answer_projection")
        if isinstance(state.get("execution_contract"), dict)
        else {}
    )
    projection_violations = semantic_projection_violations(
        str(state.get("question") or ""),
        state.get("final_answer_payload") or state.get("final_answer"),
    )
    projection_violations.extend(
        contract_projection_violations(
            str(state.get("question") or ""),
            state.get("final_answer_payload") or state.get("final_answer"),
            answer_projection,
        )
    )
    projection_violations = list(dict.fromkeys(projection_violations))
    if projection_violations:
        needs_full_records = any("完整记录" in issue for issue in projection_violations)
        review = FinalEvidenceReview(
            decision="repair" if needs_full_records else "revise",
            reason="候选答案丢失了通用的语义可追溯信息。",
            missing_requirements=projection_violations,
            recommended_tool="python_analysis_tool" if needs_full_records else "none",
            repair_objective=(
                "重新读取已有数据并在 data_payload 中返回用户要求的全部记录，不得只返回预览"
                if needs_full_records
                else "从已有工具证据中补齐语义字段或比较指标，不改变真实计算值"
            ),
        )
    elif not bool(state.get("final_answer_schema_valid")):
        review = FinalEvidenceReview(
            decision="revise",
            reason="候选答案尚未通过结构化输出协议。",
            missing_requirements=["生成符合 StrictFinalAnswer 协议的答案"],
            recommended_tool="none",
            repair_objective="仅依据现有工具证据重新生成结构化答案",
        )
    elif _candidate_equals_typed_sql_evidence(state):
        review = FinalEvidenceReview(
            decision="accept",
            reason="结构化答案与 SCGA 的完整 typed payload 一致，且 Schema 覆盖与输出契约检查均通过。",
        )
        print("  [review:v2] typed SCGA 证据一致，跳过 Reviewer LLM。", flush=True)
    elif _candidate_equals_typed_python_evidence(state):
        review = FinalEvidenceReview(
            decision="accept",
            reason="结构化答案与 STCA 的完整 typed payload 一致，且输出契约检查均通过。",
        )
        print("  [review:v2] typed STCA 证据一致，跳过 Reviewer LLM。", flush=True)
    else:
        try:
            review_base = config.get_llm()
            if hasattr(review_base, "model_copy"):
                review_base = review_base.model_copy(
                    update={"max_tokens": int(getattr(config, "V2_REVIEW_MAX_TOKENS", 1000) or 1000)}
                )
            review_llm = review_base.with_structured_output(FinalEvidenceReview)
            review = review_llm.invoke(
                [
                    SystemMessage(content=FINAL_EVIDENCE_REVIEW_SYSTEM),
                    HumanMessage(content=_review_prompt(state)),
                ]
            )
            if not isinstance(review, FinalEvidenceReview):
                review = FinalEvidenceReview.model_validate(review)
            review = _sanitize_review(review)
        except Exception as exc:
            # Review is an accuracy layer, not a new availability dependency.
            print(
                f"  [review] 审查器不可用，保留已通过 Schema 的候选：{type(exc).__name__}: {exc}",
                flush=True,
            )
            return {"review_action": "accept", "review_count": count + 1}

    print(
        f"  [review] decision={review.decision} | reason={review.reason}",
        flush=True,
    )
    base: dict[str, Any] = {
        "review_count": count + 1,
        "review_feedback": review.reason + "；" + "；".join(review.missing_requirements),
    }
    if review.decision == "accept":
        base["review_action"] = "accept"
        return base

    base.update(
        {
            "messages": [_control_feedback(review)],
            "final_answer": None,
            "final_answer_payload": None,
            "final_answer_schema_valid": False,
            "final_answer_schema_error": None,
        }
    )
    if review.decision == "revise":
        base["review_action"] = "revise"
        return base
    if review.decision == "repair":
        base.update(_repair_step_updates(state, review))
        if base.get("review_action") == "replan":
            base["replan_count"] = int(state.get("replan_count") or 0) + 1
        return base

    if int(state.get("replan_count") or 0) >= int(
        getattr(config, "V2_MAX_REPLANS", 1) or 1
    ):
        base["review_action"] = "revise"
        return base
    base["review_action"] = "replan"
    base["replan_count"] = int(state.get("replan_count") or 0) + 1
    return base


def route_after_review(state: dict[str, Any]) -> str:
    action = str(state.get("review_action") or "accept")
    if action == "replan":
        return "planner"
    if action in {"revise", "repair"}:
        return "agent"
    return "end"
