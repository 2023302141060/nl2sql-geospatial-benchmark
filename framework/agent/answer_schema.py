# -*- coding: utf-8 -*-
"""最终答案的受约束结构化协议。

该模块只负责答案边界，不参与 SQL/Python 的推理与计算。所有成功结案都必须序列化为
``StrictFinalAnswer``；测试与评价脚本据此区分合法结构化答案和自然语言兜底文本。
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator


AnswerType = Literal["scalar", "entity_list", "ranked_list", "records"]
JsonPayload = str | int | float | bool | None | list[Any] | dict[str, Any]
STRICT_ANSWER_SCHEMA_VERSION = "strict_answer_v1"


class StrictFinalAnswer(BaseModel):
    """成功结案的唯一对外输出对象。

    ``data_payload`` 允许任意合法 JSON 值，但 ``answer_type`` 会进一步约束顶层形态；
    ``extra='forbid'`` 可阻止模型夹带解释、SQL、代码或推理字段。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["strict_answer_v1"] = Field(
        default=STRICT_ANSWER_SCHEMA_VERSION,
        description="固定协议版本。",
    )
    answer_type: AnswerType = Field(
        ...,
        description="答案形态：单值、实体集合、有序排名或记录。",
    )
    data_payload: JsonPayload = Field(
        ...,
        description="仅包含回答问题所需的数据，不得包含解释、推理、SQL 或 Python 代码。",
    )

    @model_validator(mode="before")
    @classmethod
    def decode_stringified_container_payload(cls, value: Any) -> Any:
        """Normalize gateways that stringify only the nested JSON container."""
        if not isinstance(value, dict):
            return value
        if "answer_type" not in value and "data_payload" not in value:
            wrapper_keys = {
                key
                for key in ("scalar", "entity_list", "ranked_list", "records")
                if key in value
            }
            allowed_keys = wrapper_keys | {"schema_version"}
            if len(wrapper_keys) == 1 and set(value).issubset(allowed_keys):
                wrapper = next(iter(wrapper_keys))
                value = {
                    "schema_version": value.get(
                        "schema_version",
                        STRICT_ANSWER_SCHEMA_VERSION,
                    ),
                    "answer_type": wrapper,
                    "data_payload": value[wrapper],
                }
        answer_type = value.get("answer_type")
        payload = value.get("data_payload")
        if answer_type == "scalar" and isinstance(payload, dict):
            if set(payload) == {"data"} and isinstance(payload.get("data"), dict):
                normalized = dict(value)
                normalized["data_payload"] = payload["data"]
                return cls.decode_stringified_container_payload(normalized)
            payload_keys = set(payload)
            allowed_keys = {"scalar", "value", "answer"}
            matched_keys = payload_keys & allowed_keys
            if len(matched_keys) == 1 and payload_keys.issubset(allowed_keys):
                normalized = dict(value)
                normalized["data_payload"] = payload[next(iter(matched_keys))]
                return normalized
            unit_keys = {"unit", "units", "单位"}
            numeric_items = [
                (key, item)
                for key, item in payload.items()
                if key not in unit_keys and isinstance(item, (int, float))
            ]
            if len(numeric_items) == 1 and set(payload).issubset({numeric_items[0][0], *unit_keys}):
                normalized = dict(value)
                normalized["data_payload"] = numeric_items[0][1]
                return normalized
        if answer_type in {"entity_list", "ranked_list"} and isinstance(payload, dict):
            if set(payload) == {"data"} and isinstance(payload.get("data"), dict):
                normalized = dict(value)
                normalized["data_payload"] = payload["data"]
                return cls.decode_stringified_container_payload(normalized)
            payload_keys = set(payload)
            if payload_keys == {answer_type}:
                normalized = dict(value)
                normalized["data_payload"] = payload[answer_type]
                return normalized
            list_items = [
                (key, item)
                for key, item in payload.items()
                if isinstance(item, list)
            ]
            if len(list_items) == 1:
                normalized = dict(value)
                normalized["data_payload"] = list_items[0][1]
                return normalized
        if not isinstance(payload, str) or answer_type == "scalar":
            return value
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError):
            return value
        valid = False
        if answer_type in {"entity_list", "ranked_list"}:
            valid = isinstance(decoded, list)
        elif answer_type == "records":
            valid = isinstance(decoded, dict) or (
                isinstance(decoded, list)
                and all(isinstance(item, dict) for item in decoded)
            )
        if not valid:
            return value
        normalized = dict(value)
        normalized["data_payload"] = decoded
        return normalized

    @model_validator(mode="after")
    def validate_payload_shape(self) -> "StrictFinalAnswer":
        payload = self.data_payload
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("data_payload 必须是有限数值构成的合法 JSON 值") from exc
        if self.answer_type == "scalar" and isinstance(payload, (dict, list)):
            raise ValueError("answer_type=scalar 时 data_payload 必须是 JSON 标量")
        if self.answer_type in {"entity_list", "ranked_list"} and not isinstance(payload, list):
            raise ValueError(f"answer_type={self.answer_type} 时 data_payload 必须是列表")
        if self.answer_type == "records":
            is_record = isinstance(payload, dict)
            is_record_list = isinstance(payload, list) and all(isinstance(x, dict) for x in payload)
            if not (is_record or is_record_list):
                raise ValueError("answer_type=records 时 data_payload 必须是对象或对象列表")
        return self


_RANKING_PATTERN = re.compile(
    r"(?:\btop\s*\d*\b|排名|排行|排序|最高|最低|最大|最小|最多|最少|前\s*\d+\s*(?:个|名|项)?)",
    re.IGNORECASE,
)


def infer_answer_contract_hint(question: str) -> str:
    """Infer a small semantic output contract from the question.

    This is deliberately limited to output shape.  It never supplies an
    answer value and therefore cannot leak benchmark gold data.
    """
    text = str(question or "").strip()
    lowered = text.lower()
    top_k = re.search(
        r"(?:前\s*|top\s*)(\d+)|(?:最高|最低|最大|最小|最多|最少)的?\s*(\d+)\s*个",
        lowered,
        re.IGNORECASE,
    )
    asks_single_entity = bool(
        re.search(r"(?:哪个|哪一个|哪一(?:州|省|市|县|类|个)|是什么|为多少|是多少)", lowered)
    )
    asks_extreme = bool(re.search(r"(?:最高|最低|最大|最小|最多|最少)", lowered))
    asks_identifier = bool(
        re.search(r"(?:网格|空间单元|州|省|市|县).{0,12}(?:id|编号|名称|是什么|哪个)", lowered)
    )
    asks_count = bool(re.search(r"(?:多少个|数量|总数|计数|count)", lowered))
    asks_multiple = bool(
        re.search(r"(?:哪些|所有|分别|逐个|逐月|逐年|各(?:州|类|组|网格|空间单元))", lowered)
    )
    asks_rank = bool(
        re.search(
            r"(?:排名|排行|排序|最多|最少|top\s*\d+|前\s*\d+\s*(?:个|名|项)?)",
            lowered,
        )
    )
    if top_k or asks_rank:
        return (
            "优先使用 ranked_list；保持题目要求的顺序和 Top-K 数量。"
            "每项同时保留实体标识符和用于排序/比较的指标值。"
        )
    if asks_single_entity and asks_extreme:
        return (
            "只返回满足最高/最低条件的单个最优实体，优先使用单条 records；"
            "同时保留实体标识符和用于比较的指标值。即使证据包含完整排序，也不得返回额外实体。"
        )
    if asks_count and not asks_multiple:
        return "优先返回 scalar；只返回题目要求的计数值。"
    if asks_multiple:
        return (
            "返回 entity_list 或 records；若证据同时含实体及指标值，必须使用 records，"
            "不得只保留实体名称。"
        )
    if asks_identifier:
        return (
            "问题询问单个实体；若证据只含标识符可返回 scalar，"
            "若同时含比较指标则使用单条 records 并保留标识符和指标。"
        )
    return "根据问题所需字段选择 scalar 或 records；禁止携带诊断统计、预览表和未被询问的字段。"


def infer_answer_type(data_payload: JsonPayload, question: str = "") -> AnswerType:
    """按顶层形态与问题语义推断答案类型，供确定性工具结果直接封装。"""
    if isinstance(data_payload, list):
        if _RANKING_PATTERN.search(str(question or "")):
            return "ranked_list"
        if all(not isinstance(item, (dict, list)) for item in data_payload):
            return "entity_list"
        if all(isinstance(item, dict) for item in data_payload):
            return "records"
        # 混合列表不属于实体集合；用 records 会被模型拒绝，因此按有序列表保留原始顺序。
        return "ranked_list"
    if isinstance(data_payload, dict):
        return "records"
    return "scalar"


def build_strict_answer(data_payload: Any, question: str = "") -> StrictFinalAnswer:
    """将已由工具真实计算得到的 JSON 数据确定性封装为最终答案。"""
    answer_type = infer_answer_type(data_payload, question)
    return StrictFinalAnswer(answer_type=answer_type, data_payload=data_payload)


def build_strict_answer_from_sql_payload(
    data_payload: Any,
    question: str = "",
) -> StrictFinalAnswer:
    """Deterministically project a completed SQL result into the public schema.

    SCGA is required to make its final SQL query answer the question directly.
    Consequently, a typed SQL payload is stronger evidence than a second LLM
    paraphrase.  The only structural normalization performed here is collapsing
    a one-column result into a scalar/entity list; multi-column rows remain
    records so entity/value pairings cannot be lost.
    """
    if isinstance(data_payload, list) and data_payload and all(
        isinstance(item, dict) and len(item) == 1 for item in data_payload
    ):
        keys = [str(next(iter(item.keys()))) for item in data_payload]
        values = [next(iter(item.values())) for item in data_payload]
        if len(values) == 1:
            return StrictFinalAnswer(answer_type="scalar", data_payload=values[0])
        one_key = len({key.casefold() for key in keys}) == 1
        key = keys[0].casefold() if one_key else ""
        entity_like = bool(
            re.search(
                r"(?:^|_)(?:id|name|state|province|city|county|cell|entity|category|class|type)(?:$|_)",
                key,
            )
            or key in {"shapename", "shapeid", "shapeiso"}
        )
        if one_key and entity_like:
            return StrictFinalAnswer(answer_type="entity_list", data_payload=values)
    return build_strict_answer(data_payload, question=question)


def coerce_strict_answer(value: Any) -> StrictFinalAnswer:
    """校验 Pydantic 对象、字典或纯 JSON 字符串；不接受 Markdown/正文中的 JSON 片段。"""
    if isinstance(value, StrictFinalAnswer):
        return value
    if isinstance(value, str):
        return StrictFinalAnswer.model_validate_json(value)
    return StrictFinalAnswer.model_validate(value)


def serialize_strict_answer(answer: StrictFinalAnswer) -> str:
    """输出紧凑纯 JSON；不生成 Markdown 代码块或附加说明。"""
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def strict_output_transport(
    model_name: str,
    existing_extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select a structured-output transport compatible with the target model."""
    model_lower = str(model_name or "").strip().lower()
    extra = dict(existing_extra or {})
    if "qwen3.7-max" in model_lower:
        # This model requires thinking mode and rejects forced tool_choice.
        extra.pop("thinking", None)
        extra["enable_thinking"] = True
        return "json_schema", extra
    extra.update(
        {
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        }
    )
    return "function_calling", extra


def build_strict_formatter(base_llm: Any) -> tuple[Any, str]:
    """Clone an LLM with model-compatible strict-output request parameters."""
    model_name = (
        getattr(base_llm, "model_name", None)
        or getattr(base_llm, "model", None)
        or ""
    )
    method, extra_body = strict_output_transport(
        str(model_name),
        getattr(base_llm, "extra_body", None),
    )
    formatter_llm = base_llm
    if hasattr(base_llm, "model_copy"):
        formatter_llm = base_llm.model_copy(
            update={"extra_body": extra_body}
        )
    return formatter_llm, method


STRICT_FINAL_SYSTEM_PROMPT = """你是 GIS 问答系统的最终答案结构化器。
你只能根据给定的工具证据和候选答案整理数据，不得补充、猜测或重新计算事实。
严格遵守 StrictFinalAnswer JSON Schema：
1. 只保留直接回答用户问题所需的数据；禁止解释、推理、SQL、Python 代码、Markdown 和无关字段。
2. 实体及其数值、年份及其指标必须位于同一条记录中，不得拆成互不关联的列表。
3. 多实体无序结果使用 entity_list 或 records；排名结果使用 ranked_list 并保持名次顺序；
   单一 JSON 标量使用 scalar；对象或对象列表使用 records。
4. 若候选答案含单位，将单位与数值保留在同一字段或同一记录中。
5. 候选答案只要同时给出了实体名称及其指标值，就必须使用 records 并同时保留二者；
   不得把“实体 + 数值”缩减成只含实体或只含数值的 scalar。
6. 若提供了工具结构化证据，字段名、实体拼写和数值精度必须优先原样取自证据；
   不得翻译实体、追加括号别名或把高精度数值改成候选答案中的四舍五入值。
7. Top-K/排名答案的每一项必须同时保留实体标识符和排序指标；不得只返回名称列表。
8. 字段名应表达用户问题中的指标、统计量或单位；有具体语义时不要退化为 `value`、`count`、`result`。
9. 极值结果保留实体标识及比较指标；先筛选实体再聚合时，保留入选实体数量或小规模完整列表，以便结果可追溯。
"""


def _top_k_projection_violation(
    question: str,
    answer: StrictFinalAnswer,
    evidence_text: str,
) -> str:
    """Detect the common semantic loss: ranked evidence projected to names only."""
    if not re.search(
        r"(?:前\s*\d+|top\s*\d+|排名|排行|排序|最多(?:的)?\s*\d+\s*个|最少(?:的)?\s*\d+\s*个)",
        str(question or ""),
        re.IGNORECASE,
    ):
        return ""
    try:
        evidence = json.loads(str(evidence_text or ""))
    except (TypeError, ValueError):
        return ""

    rows: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                rows.extend(item for item in value if isinstance(item, dict))
            else:
                for item in value:
                    collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(evidence)
    evidence_has_entity_and_metric = any(
        len(row) >= 2
        and any(isinstance(value, str) for value in row.values())
        and any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row.values())
        for row in rows
    )
    if not evidence_has_entity_and_metric:
        return ""

    payload = answer.data_payload
    if not isinstance(payload, list) or not payload:
        return "排名答案必须是非空列表，并逐项保留实体和排序指标。"
    if any(not isinstance(item, dict) for item in payload):
        return "排名证据含实体和指标，但答案退化成了名称/标识符列表。"
    if any(
        not any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in item.values())
        for item in payload
    ):
        return "排名答案中至少一项缺少用于排序的数值指标。"
    return ""


def constrain_candidate_final_answer(
    *,
    base_llm: Any,
    question: str,
    candidate_text: str,
    evidence_text: str = "",
) -> StrictFinalAnswer:
    """Use evidence-aware structured output and then validate it with Pydantic.

    A JSON candidate that is syntactically valid is only trusted when no tool
    evidence is available.  With evidence present, v2 performs one semantic
    projection pass so that an otherwise valid but wrongly shaped answer
    (for example, a scalar for an entity-plus-metric question) is corrected.
    """
    stripped = str(candidate_text or "").strip()
    parsed_candidate: StrictFinalAnswer | None = None
    if stripped:
        try:
            parsed_candidate = coerce_strict_answer(stripped)
        except (ValueError, TypeError):
            pass
    if parsed_candidate is not None and not str(evidence_text or "").strip():
        return parsed_candidate

    formatter_llm, structured_method = build_strict_formatter(base_llm)
    structured_llm = formatter_llm.with_structured_output(
        StrictFinalAnswer,
        method=structured_method,
    )
    try:
        prompt = (
            f"用户问题：\n{question}\n\n"
            f"输出形态提示（只约束形态，不提供答案）：\n"
            f"{infer_answer_contract_hint(question)}\n\n"
            f"工具结构化证据（若非空，以此处字段和值为准）：\n"
            f"{str(evidence_text or '').strip() or '（空）'}\n\n"
            f"候选答案（只用于理解如何回答，不得覆盖证据精度）：\n"
            f"{stripped or '（空）'}"
        )
        result = coerce_strict_answer(
            structured_llm.invoke(
                [
                    SystemMessage(content=STRICT_FINAL_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
        )
        violation = _top_k_projection_violation(question, result, evidence_text)
        if violation:
            result = coerce_strict_answer(
                structured_llm.invoke(
                    [
                        SystemMessage(content=STRICT_FINAL_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"{prompt}\n\n"
                                f"上一次结构化结果未通过语义检查：{violation}\n"
                                "请重新输出，并从工具证据逐项保留实体标识符与排序指标的完整精度。"
                            )
                        ),
                    ]
                )
            )
            remaining = _top_k_projection_violation(question, result, evidence_text)
            if remaining:
                raise ValueError(f"最终答案仍违反排名输出契约：{remaining}")
        return result
    except Exception:
        # The semantic review is an accuracy enhancement, not a new single
        # point of failure.  Preserve an already schema-valid candidate when
        # the formatter itself is temporarily unavailable.
        if parsed_candidate is not None:
            return parsed_candidate
        raise


def latest_finalization_evidence(messages: list[Any]) -> str:
    """提取最近一次成功工具回包中的答案数据，不把 SQL、代码或错误带入结案器。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        additional = getattr(message, "additional_kwargs", {}) or {}
        if additional.get("success") is not True:
            continue
        payload = additional.get("payload")
        if not isinstance(payload, dict):
            continue

        tool_name = str(additional.get("tool_name") or "")
        evidence: dict[str, Any] = {"tool_name": tool_name}
        if tool_name == "text2sql_tool":
            previews = []
            for result in payload.get("sql_results") or []:
                if not isinstance(result, dict):
                    continue
                if result.get("data_payload") is not None:
                    previews.append(result.get("data_payload"))
                elif result.get("data_peek") is not None:
                    previews.append(result.get("data_peek"))
            if previews:
                evidence["data_payload"] = previews
        elif tool_name == "python_analysis_tool" and "data_payload" in payload:
            evidence["data_payload"] = payload.get("data_payload")

        if len(evidence) > 1:
            return json.dumps(evidence, ensure_ascii=False)[:12000]
    return ""


def strict_answer_state_fields(answer: StrictFinalAnswer) -> dict[str, Any]:
    """返回与 LangGraph 消息无关的统一状态字段。"""
    answer_json = serialize_strict_answer(answer)
    return {
        "final_answer": answer_json,
        "final_answer_payload": answer.model_dump(mode="json"),
        "final_answer_schema_valid": True,
        "final_answer_schema_error": None,
    }


def schema_failure_state_fields(exc: Exception) -> dict[str, Any]:
    """结构化失败必须显式标记，供集成测试与主评价直接判负。"""
    error = f"{type(exc).__name__}: {exc}"
    return {
        "final_answer_payload": None,
        "final_answer_schema_valid": False,
        "final_answer_schema_error": error[:2000],
        "errors": [error],
    }
