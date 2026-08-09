# -*- coding: utf-8 -*-
"""Function Calling Agent 节点与工具执行节点。"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

import config
from agent.answer_schema import (
    StrictFinalAnswer,
    build_strict_answer,
    build_strict_answer_from_sql_payload,
    constrain_candidate_final_answer,
    coerce_strict_answer,
    serialize_strict_answer,
)
from agent.answer_contract import build_answer_projection_contract
from agent.execution_contract import build_execution_contract_from_plan, empty_execution_contract
from agent.state import IntentionSlots, PlanBlueprint, PlanStep
from agent.tooling.registry import (
    ALL_TOOLS,
    TOOLS_MAP,
    ExecutionRoute,
    graph_route_name,
    spec_for_tool_name,
)
from agent.tools import (
    FAILURE_CODE_PYTHON_INPUT_UNREADABLE,
    FAILURE_CODE_PYTHON_REQUIRES_GEOMETRY_EXPORT,
    execute_python_analysis_logic,
    execute_text2sql_logic,
    infer_require_geojson_for_python,
    validate_explicit_geojson_paths_in_workspace,
)
from prompts.intention_parsing import get_intention_parsing_messages, print_intention_rag_observability


def _raise_if_tool_payload_infra_failure(tool_name: str, payload: Any, content: Any = "") -> None:
    """Stop the graph immediately for API/network/quota failures."""
    blob = f"{content}\n{json.dumps(payload, ensure_ascii=False, default=str) if isinstance(payload, dict) else payload}".lower()
    markers = (
        "apiconnectionerror", "connection error", "apitimeouterror", "request timed out",
        "internalservererror", "serviceunavailableerror",
        "error code: 500", "error code: 502", "error code: 503", "error code: 504",
        "allocationquota", "freetieronly", "arrearage", "overdue-payment",
        "insufficient balance", "insufficient_quota", "permissiondeniederror",
        "authenticationerror", "invalid api key", "error code: 401", "error code: 403",
        "error code: 429", "ratelimiterror", "rate limit",
    )
    marker = next((m for m in markers if m in blob), None)
    if marker:
        raise RuntimeError(f"{tool_name} infrastructure failure: {marker}")


def _raise_if_exception_infra_failure(context: str, exc: Exception) -> None:
    """Propagate LLM/API infrastructure failures instead of converting them to semantic fallbacks."""
    _raise_if_tool_payload_infra_failure(context, {"error": f"{type(exc).__name__}: {exc}"})
from tools.schema_retriever import (
    build_planner_schemas_yaml_from_rag_list,
    build_rag_query,
    build_text2sql_schemas_yaml_from_bundle,
    extract_table_name_from_schema_yaml,
    enrich_intention_slots_from_schema,
    format_schema_yaml_by_exact_table_names,
    retrieve_top_k_schema_bundle,
)

from .constants import (
    DIRECT_ANSWER_SYSTEM,
    FAILURE_CODE_PYTHON_MISSING_EXPLICIT_FILE,
    FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS,
    FAILURE_CODE_TEXT2SQL_BLOCKED_ON_PYTHON_STEP,
    FAILURE_CODE_TEXT2SQL_RAW_SQL_IN_QUESTION,
    MASTER_SYSTEM_PROMPT,
    MAX_EXECUTOR_AGENT_RETRIES,
    MAX_GRAPH_MESSAGES,
    MAX_MESSAGES_HARD_STOP,
    PLANNER_SYSTEM,
)
from agent.tooling.observation import (
    FAILURE_CODE_SERIAL_EXECUTION_IGNORED,
    build_tool_response_sequence,
    tool_observation,
)
from agent.tooling.parse import merge_native_tool_calls
from agent.tooling.policy import flatten_nested_tool_args, validate_tool_args_dict

# 蓝图 SQL→Python 顺序与 text2sql_node 跨步回溯：
# - 历史：Planner 强制「先 Text2SQL 再 Python」，并防止早期模型在失败时不修代码、却在 SQL/Python 间无意义震荡
#   （State oscillation），故曾在 Python 步硬拦改调 text2sql。
# - 现行：若当前为 Python 蓝图步，仅当「紧邻本轮 AIMessage 之前的连续 ToolMessage 批次」中**全部**为成功
#   （或无真实工具回包）时，才拦截改调 text2sql；任一条真实工具 success=False（非并行占位）则放行，
#   以便沙盒无结构化 failure_code 时模型仍可回溯重写 SQL（如拆分 GeoJSON）。


def direct_answer_node(state: dict[str, Any]) -> dict[str, Any]:
    """快速通道节点：对不可查询的闲聊/非 GIS 问题直接生成友好回复，写入 direct_response 和 final_answer。"""
    question = _get_effective_question(state)

    slots = state.get("slots") or {}
    reject_reason = slots.get("reject_reason") or "该问题不属于 GIS 空间数据分析范畴"

    user_content = f"用户问题：{question}\n拒绝原因：{reject_reason}"

    llm = config.get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=user_content),
        ])
        reply = _normalize_ai_content(getattr(response, "content", "")).strip()
    except Exception as e:
        print(f"  [direct_answer] LLM 调用失败，使用兜底回复。原因：{type(e).__name__}: {e}", flush=True)
        reply = (
            f"抱歉，我是一个专注于本地时空地理数据分析的智能体，无法回答这个问题。"
            f"原因：{reject_reason}。"
            "如果你有关于地理空间数据查询、遥感指标统计、空间拓扑分析等方面的问题，欢迎继续提问！"
        )

    print(f"  [direct_answer] 生成快速回复（长度={len(reply)}）", flush=True)

    ai_msg = AIMessage(
        content=reply,
        additional_kwargs={"parsed": {"type": "final", "tool_calls": [], "output": reply}},
    )
    return {
        "messages": [ai_msg],
        "direct_response": reply,
        "final_answer": reply,
    }


def _blob_matches_python_stats_keywords(planning_query: str, slots: dict[str, Any]) -> bool:
    """用户问题或槽位是否体现超出关系代数的分析语义。"""
    zhs = (
        "回归", "相关性", "相关系数", "皮尔逊", "聚类", "双变量", "协方差",
        "标准化", "归一化", "z-score", "主成分", "显著性检验", "曼-惠特尼",
        "置信区间", "bootstrap", "四分位距", "异常值", "基尼系数", "熵",
        "轮廓系数", "k-means", "蒙特卡洛", "凸包", "最小外接矩形",
        "正态性检验", "统计检验", "t 检验", "t检验", "方差分析",
        "相邻月份", "环比", "同比", "时序差分", "变化检测", "发生变化",
        "用 python", "使用 python", "python 做", "python计算", "python 分析",
    )
    ens = (
        "regression", "correlation", "cluster", "pearson", "covariance",
        "multicolinearity", "standardization", "normalization", "z-score", "pca",
        "mann-whitney", "confidence interval", "bootstrap", "iqr", "silhouette",
        "k-means", "gini", "entropy", "monte carlo", "convex hull",
        "shapiro", "t-test", "ttest", "anova", "statistical test", "python",
    )
    pq = str(planning_query or "")
    if any(k in pq for k in zhs):
        return True
    pl = re.sub(r"[‐‑‒–—―−]", "-", pq.lower())
    if any(k in pl for k in ens):
        return True
    try:
        sj = json.dumps(slots or {}, ensure_ascii=False).lower()
    except Exception:
        sj = ""
    return any(k in sj for k in ens)


def _requires_python_capability(planning_query: str, slots: dict[str, Any]) -> bool:
    """按工具能力边界决定是否需要 Python，而不是让 Planner 自由增加步骤。"""
    if _blob_matches_python_stats_keywords(planning_query, slots):
        return True
    if str(slots.get("spatial_predicate") or "").strip():
        return True
    if str(slots.get("spatial_threshold") or "").strip():
        return True
    spatial_terms = (
        "接壤", "相邻", "相交", "包含", "缓冲区", "几何质心", "几何中心",
        "大地线距离", "最短距离", "叠置", "空间连接", "投影坐标", "epsg:",
    )
    text = str(planning_query or "").lower()
    if any(term in text for term in spatial_terms):
        return True
    # “完全或部分位于/落入某范围”是自然语言中常见的跨图层空间谓词。
    # 不能仅依赖意图模型恰好把它写入 spatial_predicate，否则会把任务误路由为
    # 单步关系代数查询，并丢失行政区几何与网格几何之间的空间运算。
    return bool(
        re.search(r"(?:完全|部分).{0,6}(?:位于|落入|落在)", text)
        or re.search(
            r"(?:行政区|城市|市|县|省|州).{0,8}(?:范围|区域)内.{0,10}"
            r"(?:网格|渔网|栅格|像元|空间单元)",
            text,
        )
    )


def _derive_required_geometry_tables(
    planning_query: str,
    schema_yamls: list[str] | None,
) -> list[str]:
    """Identify geometry layers required by a cross-layer spatial predicate.

    This contract is intentionally role based: when the question relates a
    grid/raster layer to an administrative layer, SCGA must export both raw
    geometry operands.  Single-layer predicates and literal-coordinate queries
    are left unconstrained beyond the existing ``requires_geometry`` flag.
    """
    text = str(planning_query or "")
    asks_grid_layer = bool(re.search(r"网格|渔网|栅格|像元|\bgrid\b|\bcell\b", text, re.IGNORECASE))
    asks_admin_layer = bool(
        re.search(r"行政区|行政边界|州域|省域|县域|市域|(?:省|市|州|县).{0,8}(?:边界|范围|几何|质心)", text)
    )
    if not (asks_grid_layer and asks_admin_layer):
        return []

    role_tables: dict[str, list[str]] = {"grid": [], "admin": []}
    candidate_yamls = list(schema_yamls or [])

    # 向量预检索可能只召回属性表或其中一个几何图层。跨图层空间任务不能因此
    # 静默降级：在本地 Schema 目录中按数据域补齐缺失的几何角色，仍由下游
    # SCGA/STCA 执行，不引入题号或具体实体规则。
    domain_prefix = ""
    if "浙江" in text:
        domain_prefix = "zhejiang_"
    elif "美国" in text:
        domain_prefix = "usa_"
    candidate_domains: set[str] = set()
    for raw_yaml in candidate_yamls:
        try:
            candidate_domain = str((yaml.safe_load(raw_yaml) or {}).get("domain") or "").strip()
        except Exception:
            candidate_domain = ""
        if candidate_domain:
            candidate_domains.add(candidate_domain)
    target_domain = next(iter(candidate_domains)) if len(candidate_domains) == 1 else ""
    known_tables = {
        extract_table_name_from_schema_yaml(raw_yaml)
        for raw_yaml in candidate_yamls
        if isinstance(raw_yaml, str)
    }
    for schema_path in sorted(config.SCHEMAS_DIR.glob("*.yaml")):
        try:
            raw_yaml = schema_path.read_text(encoding="utf-8")
        except OSError:
            continue
        table_name = extract_table_name_from_schema_yaml(raw_yaml)
        if not table_name or table_name in known_tables:
            continue
        if domain_prefix and not table_name.casefold().startswith(domain_prefix):
            continue
        if target_domain:
            try:
                schema_domain = str((yaml.safe_load(raw_yaml) or {}).get("domain") or "").strip()
            except Exception:
                schema_domain = ""
            if schema_domain and schema_domain != target_domain:
                continue
        candidate_yamls.append(raw_yaml)
        known_tables.add(table_name)

    for raw_yaml in candidate_yamls:
        try:
            data = yaml.safe_load(raw_yaml) or {}
        except Exception:
            continue
        table_name = str(data.get("table_name") or "").strip()
        if not table_name:
            continue
        st = data.get("spatiotemporal_properties") or {}
        columns = data.get("columns") or []
        has_geometry = bool(st.get("has_geometry")) or any(
            str(column.get("name") or column.get("column_name") or "").strip() == "geometry"
            for column in columns
            if isinstance(column, dict)
        )
        if not has_geometry:
            continue
        schema_text = " ".join(
            (
                table_name,
                str(data.get("table_description") or ""),
                str(st.get("spatial_granularity") or ""),
            )
        ).lower()
        if re.search(r"网格|渔网|栅格|像元|fishnet|\bgrid\b|\bcell\b", schema_text):
            role_tables["grid"].append(table_name)
        if re.search(r"行政|城市|市级|省级|州级|县级|boundary|state|province|county", schema_text):
            role_tables["admin"].append(table_name)

    required: list[str] = []
    for role in ("grid", "admin"):
        candidates = role_tables[role]
        if candidates:
            required.append(candidates[0])
    return list(dict.fromkeys(required)) if len(required) == 2 else []


def _build_plan_meta_from_blueprint(
    steps: list[PlanStep],
    *,
    planning_query: str = "",
    slots: dict[str, Any] | None = None,
    schema_bindings: list[dict[str, Any]] | None = None,
    required_tables: list[str] | None = None,
    required_geometry_tables: list[str] | None = None,
) -> list[dict[str, Any]]:
    """将唯一受支持的 PlanBlueprint 编译为运行期步骤元数据。"""
    slots_d = slots if isinstance(slots, dict) else {}
    stats_semantic = _blob_matches_python_stats_keywords(planning_query, slots_d)
    meta: list[dict[str, Any]] = []
    n = len(steps)
    for i, step in enumerate(steps):
        objective = step.objective.strip()
        tool = "python_analysis" if step.tool == "python_analysis_tool" else "text2sql"
        needs_multi = any(
            key in objective
            for key in ("多个", "多文件", "多条 SQL", "多条sql", "一次性导出", "多个文件")
        )
        if tool == "python_analysis":
            requires_stats = stats_semantic
            accept_wide = stats_semantic
            exp_mode = "either" if stats_semantic else "single_file"
            requires_geometry = infer_require_geojson_for_python(
                slots_d, planning_query, objective
            )
        else:
            requires_stats = False
            accept_wide = False
            exp_mode = "na"
            requires_geometry = False
        meta.append(
            {
                "tool": tool,
                "objective": objective,
                "intent": objective,
                "success_criteria": list(step.success_criteria),
                "needs_multiple_exports": needs_multi,
                "final_python_step": tool == "python_analysis" and i == n - 1,
                "requires_python_stats": requires_stats,
                "accept_single_wide_table": accept_wide,
                "expected_input_mode": exp_mode,
                "requires_geometry": bool(requires_geometry),
                "preferred_output_type": "na",
                "schema_bindings": list(schema_bindings or []) if tool == "text2sql" else [],
                "required_tables": list(required_tables or []) if tool == "text2sql" else [],
                "required_geometry_tables": list(required_geometry_tables or []) if tool == "text2sql" else [],
            }
        )

    for i, m in enumerate(meta):
        if m.get("tool") != "text2sql":
            continue
        j = next((k for k in range(i + 1, n) if meta[k].get("tool") == "python_analysis"), None)
        if j is not None:
            rg = bool(meta[j].get("requires_geometry"))
            m["requires_geometry"] = rg
            m["preferred_output_type"] = "geojson" if rg else "json"
        else:
            m["preferred_output_type"] = "either"

    return meta


def _render_plan_steps(plan_meta: list[dict[str, Any]]) -> list[str]:
    """从结构化计划确定性渲染展示文本；文本不参与工具路由。"""
    tool_names = {
        "text2sql": "text2sql_tool",
        "python_analysis": "python_analysis_tool",
    }
    return [
        f"调用 {tool_names[item['tool']]}：{item['objective']}"
        for item in plan_meta
    ]


def _apply_text2sql_state_name_export_hint(task: str, slots: dict[str, Any], eff_question: str) -> str:
    """州级排名/「哪个州」类任务：提示导出必须含州名列（shapeName 等），避免结果只有编码。"""
    t = (task or "").strip()
    if not t:
        return t
    blob = f"{t} {eff_question or ''}"
    zh = (
        "州名",
        "哪个州",
        "哪一州",
        "最大州",
        "最小州",
        "排名第",
        "各州名称",
        "输出州",
        "哪一个州",
    )
    en = ("state name", "which state", "largest state", "smallest state", "ranking states", "what state")
    usa_scope = str(slots.get("region_scope") or slots.get("region") or "").strip() == "美国"
    asks_usa_unit_entities = usa_scope and "空间单元" in blob and any(
        token in blob for token in ("找出", "列出", "哪些", "哪个", "排名", "前 ", "前")
    )
    if (
        not any(k in blob for k in zh)
        and not any(k in blob.lower() for k in en)
        and not asks_usa_unit_entities
    ):
        return t
    hint = (
        "\n\n【导出字段】若需按州展示、比较、排名或回答「哪个/哪一州」，"
        "导出 SQL 的 SELECT 必须包含人类可读的州名称列（如 shapeName / state_name / name 等），"
        "不要仅输出州编码（如仅有 state_fp）作为对外展示依据。"
    )
    return t + hint if hint not in t else t


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Planner 节点：只接受 PlanBlueprint 结构化协议。"""
    question = _get_effective_question(state)
    slots = state.get("slots") or {}
    planning_query = question

    if slots.get("is_queryable") is False:
        return {
            "plan": [],
            "plan_meta": [],
            "execution_contract": empty_execution_contract(queryable=False),
        }

    compact_slots = {
        key: value
        for key, value in slots.items()
        if key in {
            "region", "region_scope", "region_set", "time", "time_range",
            "metric", "metric_set", "calc", "spatial_predicate",
            "spatial_threshold", "target_entity_A", "target_entity_B",
            "analytical_method", "visualization",
        }
        and value not in (None, "", [], {})
    }
    schema_bindings = [
        item for item in (state.get("schema_bindings") or []) if isinstance(item, dict)
    ]
    retrieved_tables = [
        str(item) for item in (state.get("retrieved_table_names") or []) if str(item).strip()
    ]
    user_content = (
        f"用户问题：{planning_query}\n"
        f"已解析槽位：{json.dumps(compact_slots, ensure_ascii=False, separators=(',', ':'))}"
    )
    if schema_bindings:
        user_content += (
            "\nSchema 预过滤绑定："
            + json.dumps(schema_bindings, ensure_ascii=False, separators=(",", ":"))
        )
    review_feedback = str(state.get("review_feedback") or "").strip()
    if review_feedback:
        existing_files = [str(p) for p in (state.get("geojson_paths") or []) if str(p).strip()]
        user_content += (
            "\n\n这是一次基于执行证据的重规划。上一版计划存在以下问题：\n"
            f"{review_feedback[:800]}\n"
            f"当前已有数据文件：{existing_files or '无'}\n"
            "只修复缺失或错误的部分；可复用已有证据时不要重复取数。"
        )

    # 关系代数可完整表达的任务无需再调用一个 LLM 来决定“只用 SQL”。
    # 只有跨越 SQL 能力边界的任务才进入结构化 Planner，减少一次调用并避免过度规划。
    requires_python = _requires_python_capability(planning_query, slots)
    if not review_feedback:
        initial_steps = [
            PlanStep(
                tool="text2sql_tool",
                objective=planning_query,
                success_criteria=["返回直接回答问题所需的实体和指标结果"],
            )
        ]
        if requires_python:
            initial_steps.append(
                PlanStep(
                    tool="python_analysis_tool",
                    objective=(
                        "严格按原问题完成高级统计或空间计算；保留其中的坐标系、阈值、"
                        f"算法参数和运算顺序。原问题：{planning_query}"
                    ),
                    success_criteria=["输出原问题明确要求的全部统计量或实体结果"],
                )
            )
        blueprint = PlanBlueprint(steps=initial_steps)
        route_name = "SQL+Python" if requires_python else "关系代数"
        print(f"  [planner:v2] 能力路由={route_name}，跳过规划 LLM。", flush=True)
    else:
        planner_llm = config.get_llm().with_structured_output(PlanBlueprint)
        try:
            blueprint = planner_llm.invoke([
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=user_content),
            ])
            if not isinstance(blueprint, PlanBlueprint):
                blueprint = PlanBlueprint.model_validate(blueprint)
        except Exception as e:
            print(f"  [planner] 规划失败，使用兜底单步计划。原因：{type(e).__name__}: {e}", flush=True)
            _raise_if_exception_infra_failure("planner_llm", e)
            blueprint = PlanBlueprint(
                steps=[
                    PlanStep(
                        tool="text2sql_tool",
                        objective=f"提取并计算回答该问题所需的数据：{planning_query}",
                        success_criteria=["返回可用于回答问题的非空结果"],
                    )
                ]
            )

    if requires_python and len(blueprint.steps) == 2:
        # SQL 步只负责准备原始关系数据；避免 Planner 把质心、距离、聚类等
        # Python 运算提前塞进 SQL，造成一次必然失败后才纠错。
        blueprint = PlanBlueprint(
            steps=[
                PlanStep(
                    tool="text2sql_tool",
                    objective=(
                        "为后续分析提取问题所需的完整实体标识、时间字段、原始指标与原始几何"
                        "（仅在后续空间分析需要时导出几何）；只准备原始数据，不在 SQL 中执行"
                        f"高级统计或空间计算。原问题：{planning_query}"
                    ),
                    success_criteria=["导出后续分析所需的完整原始字段和记录"],
                ),
                PlanStep(
                    tool="python_analysis_tool",
                    objective=(
                        "严格按原问题完成高级统计或空间计算；保留其中的坐标系、阈值、"
                        f"算法参数和运算顺序。原问题：{planning_query}"
                    ),
                    success_criteria=list(blueprint.steps[1].success_criteria),
                ),
            ]
        )

    required_geometry_tables = _derive_required_geometry_tables(
        planning_query,
        [str(item) for item in (state.get("schemas") or []) if isinstance(item, str)],
    )
    plan_meta = _build_plan_meta_from_blueprint(
        blueprint.steps,
        planning_query=planning_query,
        slots=slots if isinstance(slots, dict) else {},
        schema_bindings=schema_bindings,
        required_tables=retrieved_tables,
        required_geometry_tables=required_geometry_tables,
    )
    steps = _render_plan_steps(plan_meta)

    print(f"  [planner] 生成 {len(steps)} 步执行计划：", flush=True)
    for i, step in enumerate(steps, start=1):
        print(f"    步骤{i}: {step}", flush=True)

    execution_contract = build_execution_contract_from_plan(
        planning_query=planning_query,
        slots=slots if isinstance(slots, dict) else {},
        plan_meta=plan_meta,
        schema_bindings=schema_bindings,
        required_tables=retrieved_tables,
        required_geometry_tables=required_geometry_tables,
        schema_coverage=(state.get("schema_coverage") or {}),
    )
    return {
        "plan": steps,
        "plan_meta": plan_meta,
        "execution_contract": execution_contract,
        "current_plan_step_index": 1 if steps else None,
        "current_plan_step": steps[0] if steps else None,
        "review_action": None,
        "review_feedback": None,
        "final_answer": None,
        "final_answer_payload": None,
        "final_answer_schema_valid": False,
        "final_answer_schema_error": None,
    }


def intention_and_rag_node(state: dict[str, Any]) -> dict[str, Any]:
    """前置节点：解析意图槽位并预检索相关 Schema。

    意图模型只做可查询性与槽位提取；原始问题和紧凑槽位直接用于 RAG，
    不再生成一份重复且可能漂移的 expanded_query。
    """
    question = _get_effective_question(state)
    if not question.strip():
        print_intention_rag_observability({}, None)
        return {
            "slots": {},
            "schemas": [],
            "schemas_yaml": "",
            "retrieved_table_names": [],
            "schema_bindings": [],
            "schema_coverage": {},
        }

    def _empty_queryable_slots() -> dict[str, Any]:
        return {
            "is_queryable": True,
            "reject_reason": "",
            "region": "",
            "region_scope": "",
            "region_set": [],
            "time": "",
            "time_range": [],
            "metric": "",
            "metric_set": [],
            "calc": "",
            "spatial_predicate": "",
            "spatial_threshold": "",
            "target_entity_A": "",
            "target_entity_B": "",
            "analytical_method": "",
            "condition": {},
            "visualization": "",
        }

    operation_terms = (
        "查", "查询", "获取", "返回", "给出", "计算", "统计", "找出", "列出", "筛选", "比较", "提取", "分析",
        "平均", "总和", "最高", "最低", "聚类", "回归", "相关", "检验", "排名",
    )
    data_terms = (
        "空间", "网格", "渔网", "美国", "浙江", "杭州", "州", "省", "市", "县",
        "ndvi", "pm2", "xco2", "温度", "气温", "降水", "人口", "灯光", "风速",
        "坡度", "海拔", "土地覆盖", "建设用地", "geometry", "geojson", "epsg",
    )
    q_low = question.lower()
    explicit_data_query = (
        any(term in q_low for term in operation_terms)
        and any(term in q_low for term in data_terms)
    )
    if explicit_data_query:
        slots_dict = _empty_queryable_slots()
        print(
            "  [pre_rag:v2] 显式 GIS/数据查询：保留 Intent Understanding 节点，"
            "使用 Schema 驱动的确定性槽位补全，跳过意图 LLM。",
            flush=True,
        )
    else:
        system_msg, user_msg = get_intention_parsing_messages(question)
        llm = config.get_llm().with_structured_output(IntentionSlots)
        try:
            slots = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
            slots_dict = slots.model_dump() if slots else {}
        except ValidationError:
            slots_dict = _empty_queryable_slots()
        except Exception as exc:
            _raise_if_exception_infra_failure("pre_rag_llm", exc)
            slots_dict = _empty_queryable_slots()

    # 记录意图模型原始给出的检索词。Schema 补全得到的物理字段名只进入执行
    # 契约，不再反向拼接进 RAG 查询，避免错误绑定形成自我强化闭环。
    intent_rag_slots = dict(slots_dict)

    # 低成本确定性语义补全属于 Intent Understanding 模块的一部分：保留
    # LLM 槽位结果，但为快速通道补齐域、时间、指标和分析方法。
    slots_dict = enrich_intention_slots_from_schema(question, slots_dict)

    # ── 阶段 ①：可查询性判定 ──────────────────────────────────────────────
    is_queryable = slots_dict.get("is_queryable", True)
    reject_reason = slots_dict.get("reject_reason") or ""
    print(
        f"  [pre_rag] is_queryable={is_queryable}"
        + (f"  reject_reason={reject_reason!r}" if not is_queryable else ""),
        flush=True,
    )

    if not is_queryable:
        # 不可查询：跳过 Schema 检索，直接返回拒绝状态供 agent_node 生成友好回复
        print(f"  [pre_rag] 提前拒绝，不进行 Schema 检索。", flush=True)
        print_intention_rag_observability(slots_dict, None)
        return {
            "slots": slots_dict,
            "schemas": [],
            "schemas_yaml": "",
            "retrieved_table_names": [],
            "schema_bindings": [],
            "schema_coverage": {},
        }

    # 原问题是唯一语义锚点；槽位只追加检索关键词，不覆盖或改写用户表达。
    rag_terms: list[str] = [question]
    for key in (
        "region", "region_scope", "time", "metric", "calc",
        "spatial_predicate", "analytical_method", "target_entity_A", "target_entity_B",
    ):
        source_slots = intent_rag_slots if key == "metric" else slots_dict
        value = str(source_slots.get(key) or "").strip()
        if value and value not in question:
            rag_terms.append(value)
    for key in ("region_set", "time_range", "metric_set"):
        source_slots = intent_rag_slots if key == "metric_set" else slots_dict
        for value in source_slots.get(key) or []:
            text = str(value).strip()
            if text and text not in question and text not in rag_terms:
                rag_terms.append(text)
    rag_query = " | ".join(rag_terms)
    print(
        "  [pre_rag] RAG query='original question + compact slots': "
        f"{rag_query[:80]}{'...' if len(rag_query) > 80 else ''}",
        flush=True,
    )

    schema_bundle = retrieve_top_k_schema_bundle(
        slots_dict,
        natural_language_query=rag_query,
        semantic_anchor_query=question,
    )
    relevant_yamls = schema_bundle.get("schemas", [])
    print_intention_rag_observability(slots_dict, schema_bundle)
    return {
        "slots": slots_dict,
        "schemas": relevant_yamls,
        "schemas_yaml": build_planner_schemas_yaml_from_rag_list(relevant_yamls),
        "retrieved_table_names": schema_bundle.get("table_names", []),
        "schema_bindings": schema_bundle.get("semantic_bindings", []),
        "schema_coverage": schema_bundle.get("schema_coverage", {}),
    }


def _get_user_question(messages: list) -> str:
    """从消息历史中提取用户原始问题（取最近一次 HumanMessage，支持多轮追问）。"""
    for msg in reversed(messages):
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if (
            isinstance(msg, HumanMessage)
            and not additional.get("control_message")
            and not str(getattr(msg, "content", "")).startswith("Observation:")
        ):
            return getattr(msg, "content", "")
    return ""


def _get_effective_question(state: dict[str, Any]) -> str:
    """优先使用消息链中最新用户问题；若无则回退 state['question']（兼容仅设初始 state 的单轮入口）。"""
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    latest = _get_user_question(messages)
    if latest:
        return str(latest)
    return str(state.get("question") or "")


def _normalize_ai_content(content: Any) -> str:
    """将 AIMessage.content 统一转成字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "")


def _get_parsed_payload(message: Any) -> dict[str, Any]:
    """安全读取消息 additional_kwargs.parsed，避免 NoneType.get 崩溃。"""
    additional = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional, dict):
        return {}
    parsed = additional.get("parsed", {})
    return parsed if isinstance(parsed, dict) else {}


def _coerce_tool_call_args(call: Any) -> dict[str, Any]:
    """tool_call['args'] 可能为 None 或非 dict，统一为 dict，避免后续 .get 链崩溃。"""
    if not isinstance(call, dict):
        return {}
    raw = call.get("args")
    if raw is None:
        return {}
    return raw if isinstance(raw, dict) else {}


def _sanitize_text_for_windows_console(text: str) -> str:
    """避免控制台因 emoji 或非常规符号在 Windows GBK 环境下崩溃。"""
    if not text:
        return text
    replacements = {
        "❌": "[ERROR]",
        "✅": "[OK]",
        "⚠️": "[WARN]",
        "⚠": "[WARN]",
        "📌": "[INFO]",
        "🔥": "[HOT]",
        "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    encoding = str(getattr(sys.stdout, "encoding", "") or "utf-8")
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except LookupError:
        return text


def _has_successful_tool_call(messages: list, tool_name: str) -> bool:
    """检查历史中是否存在目标工具的成功调用。"""
    for msg in messages:
        if isinstance(msg, ToolMessage):
            additional = getattr(msg, "additional_kwargs", {}) or {}
            if additional.get("tool_name") != tool_name:
                continue
            if additional.get("success") is True:
                return True
    return False


def _count_successful_tool_calls(messages: list, tool_name: str) -> int:
    """统计历史中目标工具的成功调用次数。"""
    n = 0
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != tool_name:
            continue
        if additional.get("success") is True:
            n += 1
    return n


def _plan_text2sql_step_count(plan_meta: list[dict[str, Any]] | None) -> int | None:
    """仅依据结构化蓝图统计 text2sql 步数。"""
    if not isinstance(plan_meta, list):
        return None
    n = sum(1 for item in plan_meta if isinstance(item, dict) and item.get("tool") == "text2sql")
    return n or None


def _last_text2sql_step_index_before_python(
    plan_meta: list[dict[str, Any]] | None,
) -> int | None:
    """在首个 python_analysis 步之前，最后一个 text2sql 步的下标（0-based）；无则 None。"""
    if not isinstance(plan_meta, list):
        return None
    py_idx = _first_plan_step_index_by_tool(plan_meta, tool="python_analysis")
    end = py_idx if py_idx is not None else len(plan_meta)
    last_sql: int | None = None
    for i in range(end):
        item = plan_meta[i]
        if isinstance(item, dict) and item.get("tool") == "text2sql":
            last_sql = i
    return last_sql


def _plan_rewind_updates_for_sql_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """将蓝图游标回退到「Python 之前最近一个 text2sql 步」，便于下一轮对齐 text2sql_tool。"""
    plan_list = state.get("plan") if isinstance(state.get("plan"), list) else []
    if not plan_list:
        return {}
    plan_meta = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else None
    idx = _last_text2sql_step_index_before_python(plan_meta)
    if idx is None:
        return {}
    return {
        "current_plan_step_index": idx + 1,
        "current_plan_step": str(plan_list[idx]),
    }


def _first_plan_step_index_by_tool(
    plan_meta: list[dict[str, Any]] | None,
    *,
    tool: str,
) -> int | None:
    """仅依据结构化蓝图返回指定工具的首个步骤下标。"""
    if not isinstance(plan_meta, list):
        return None
    for i, item in enumerate(plan_meta):
        if isinstance(item, dict) and item.get("tool") == tool:
            return i
    return None


def _plan_has_remaining_native_tool_steps(state: dict[str, Any]) -> bool:
    """结构化蓝图的当前游标及之后是否仍有原生工具步骤。"""
    plan_meta = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else []
    if not plan_meta:
        return False
    raw = state.get("current_plan_step_index")
    if isinstance(raw, int) and raw > len(plan_meta):
        return False
    idx = _get_current_step_index(state)
    return any(
        isinstance(item, dict) and item.get("tool") in {"text2sql", "python_analysis"}
        for item in plan_meta[idx:]
    )


def _question_requests_map(state: dict[str, Any]) -> bool:
    """Return True only when the user or parsed slots explicitly request a map."""
    slots = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    visualization = str(slots.get("visualization") or "").strip().lower()
    blob = " ".join(
        [
            str(state.get("question") or ""),
            visualization,
        ]
    ).lower()
    return bool(
        visualization
        or any(
            token in blob
            for token in (
                "绘制地图",
                "生成地图",
                "地图可视化",
                "可视化地图",
                "render map",
                "map visualization",
            )
        )
    )


def _planned_tool_name_for_current_step(state: dict[str, Any]) -> str | None:
    """Resolve the tool expected by the current blueprint cursor."""
    idx = _get_current_step_index(state)
    plan_meta = state.get("plan_meta")
    if not isinstance(plan_meta, list) or idx >= len(plan_meta) or not isinstance(plan_meta[idx], dict):
        return None
    meta_tool = str(plan_meta[idx].get("tool") or "")
    return {
        "text2sql": "text2sql_tool",
        "python_analysis": "python_analysis_tool",
    }.get(meta_tool)


def _executor_tools_for_state(state: dict[str, Any]) -> list[Any]:
    """Expose only tools compatible with the current v2 blueprint step.

    SQL/Python recovery remains possible while a step is active.  Once the
    blueprint cursor is past the last step, no tools are bound and the model
    must synthesize the final answer from existing observations.
    """
    if not bool(getattr(config, "V2_PLAN_AWARE_TOOL_BINDING", False)):
        return list(ALL_TOOLS)

    plan = state.get("plan") if isinstance(state.get("plan"), list) else []
    idx = _get_current_step_index(state)
    if plan and idx >= len(plan):
        return []

    expected = _planned_tool_name_for_current_step(state)
    if expected == "text2sql_tool":
        allowed = {"schema_search_tool", "text2sql_tool"}
    elif expected == "python_analysis_tool":
        # text2sql remains available when the Python input gate reports that
        # an additional or corrected export is required.
        allowed = {"schema_search_tool", "text2sql_tool", "python_analysis_tool"}
    elif expected == "map_rendering_tool":
        allowed = {"map_rendering_tool"}
    else:
        allowed = {
            "schema_search_tool",
            "text2sql_tool",
            "python_analysis_tool",
        }
        if _question_requests_map(state):
            allowed.add("map_rendering_tool")
    return [tool for tool in ALL_TOOLS if str(getattr(tool, "name", "")) in allowed]


def _export_paths_nonempty(state: dict[str, Any], geojson_paths: list[str]) -> bool:
    if geojson_paths and any(str(p).strip() for p in geojson_paths):
        return True
    snap = state.get("latest_text2sql_geojson_paths") or []
    return bool(snap and any(str(p).strip() for p in snap))


def _python_bivariate_inputs_satisfied(
    *,
    current_plan_step: str,
    question: str,
    slots: dict[str, Any],
    geojson_paths: list[str],
    sql_queries: list[Any],
) -> bool:
    """双变量/回归类任务下，当前文件+SQL 是否已满足 Python 最低输入（含单宽表）。"""
    err = _validate_python_input_completeness(
        current_plan_step=current_plan_step,
        question=question,
        slots=slots,
        geojson_paths=geojson_paths,
        sql_queries=sql_queries,
    )
    return err is None


def _python_inputs_ready_for_current_task(
    state: dict[str, Any],
    messages: list[Any],
    *,
    geojson_paths: list[str],
    current_plan_step: str,
    question: str,
    sql_queries: list[Any],
    slots: dict[str, Any],
) -> bool:
    """
    以数据依赖为主：有导出路径且（非双变量类任务或已通过完备性/宽表判定）则视为可进入 Python。
    """
    if not _export_paths_nonempty(state, geojson_paths):
        return False
    if not _python_bivariate_style_task(current_plan_step, question, slots):
        return True
    return _python_bivariate_inputs_satisfied(
        current_plan_step=current_plan_step,
        question=question,
        slots=slots,
        geojson_paths=geojson_paths,
        sql_queries=sql_queries,
    )


def _has_pending_text2sql_before_python(
    state: dict[str, Any],
    messages: list[Any],
    *,
    geojson_paths: list[str],
    current_plan_step: str,
    question: str,
    sql_queries: list[Any],
    slots: dict[str, Any],
) -> bool:
    """
    数据依赖门禁（兼容柔性执行）：输入已就绪则不拦；仅在「仍缺数据且蓝图侧还有 text2sql 步可补」时拦。
    硬规则：游标在 Python 步之前、且从未成功 text2sql、且无导出路径 → 必须先 SQL。
    """
    plan_list = state.get("plan") if isinstance(state.get("plan"), list) else []
    if not plan_list:
        return False
    plan_meta = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else None
    expected_sql = _plan_text2sql_step_count(plan_meta)
    done_sql = _count_successful_tool_calls(messages, "text2sql_tool")
    py_idx = _first_plan_step_index_by_tool(plan_meta, tool="python_analysis")
    cur = _get_current_step_index(state)

    if py_idx is not None and cur < py_idx and done_sql == 0 and not _export_paths_nonempty(state, geojson_paths):
        return True

    if _python_inputs_ready_for_current_task(
        state,
        messages,
        geojson_paths=geojson_paths,
        current_plan_step=current_plan_step,
        question=question,
        sql_queries=sql_queries,
        slots=slots,
    ):
        return False

    if expected_sql is None:
        return False
    if done_sql >= expected_sql:
        return False
    return True


def _extract_select_clause(sql: str) -> str | None:
    """提取主查询（括号深度为 0）的 SELECT 列表，避开 WITH 和嵌套子查询"""
    s = str(sql or "").lower()
    depth = 0
    select_start = -1

    for i in range(len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if select_start == -1 and s[i:].startswith("select"):
                if (i == 0 or s[i - 1].isspace()) and (i + 6 < len(s) and s[i + 6].isspace()):
                    select_start = i + 6
            elif select_start != -1 and s[i:].startswith("from"):
                if (i == 0 or s[i - 1].isspace()) and (i + 4 < len(s) and s[i + 4].isspace()):
                    return sql[select_start:i].strip()

    m = re.search(r"\bselect\s+(.+?)\s+\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _split_select_list_items(select_clause: str) -> list[str]:
    """按顶层逗号切分 SELECT 列表（不解析完整 SQL，仅作启发式）。"""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in select_clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            frag = "".join(cur).strip()
            if frag:
                parts.append(frag)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _alias_from_select_item(item: str) -> str:
    item = item.strip()
    m = re.search(r"\bas\s+([a-zA-Z_][\w]*)\s*$", item, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"(?:^|[\s,])([a-zA-Z_][\w]*)\s*$", item.strip())
    return (m2.group(1).lower() if m2 else item.lower()).strip()


_TRIVIAL_SELECT_NAMES = frozenset(
    {
        "id",
        "gid",
        "fid",
        "geom",
        "geometry",
        "the_geom",
        "name",
        "label",
        "year",
        "month",
        "day",
        "state",
        "statefp",
        "geoid",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "x",
        "y",
        "shapename",
        "shapeid",
        "shapeiso",
        "shapegroup",
        "level",
    }
)


def _alias_semantic_ndvi_metric(alias: str) -> bool:
    a = (alias or "").lower()
    return "ndvi" in a


def _alias_semantic_proportion_metric(alias: str) -> bool:
    a = (alias or "").lower()
    if "proportion" in a or "croplands" in a:
        return True
    if "cropland" in a:
        return True
    return False


def _wide_table_has_ndvi_and_proportion_columns(aliases: list[str]) -> bool:
    """单文件宽表：同时含 NDVI 类与占比/农地类列名时视为输入充分。"""
    if len(aliases) < 2:
        return False
    has_ndvi = any(_alias_semantic_ndvi_metric(a) for a in aliases if a)
    has_prop = any(_alias_semantic_proportion_metric(a) for a in aliases if a)
    return has_ndvi and has_prop


def _metric_hint_tokens(slots: dict[str, Any], question: str) -> set[str]:
    hints: set[str] = set()
    for key in ("metric", "analytical_method", "calc"):
        v = slots.get(key)
        if isinstance(v, str) and len(v.strip()) > 1:
            hints.add(v.strip().lower())
    ms = slots.get("metric_set")
    if isinstance(ms, list):
        for x in ms:
            if isinstance(x, str) and len(x.strip()) > 1:
                hints.add(x.strip().lower())
    for w in re.findall(r"[a-zA-Z_][\w]{2,}", question or ""):
        hints.add(w.lower())
    return {h for h in hints if len(h) >= 2}


def _python_task_inputs_satisfied_by_single_wide_table(
    *,
    geojson_paths: list[str],
    sql_queries: list[Any],
    current_plan_step: str,
    question: str,
    slots: dict[str, Any],
) -> bool:
    """单文件场景：对应 SQL 的 SELECT 是否已包含至少两个可区分指标列（或与槽位/问题关键词匹配的两列）。"""
    basenames: list[str] = []
    for p in geojson_paths:
        bn = Path(str(p)).name.strip()
        if bn:
            basenames.append(bn)
    uniq = list(dict.fromkeys(basenames))
    if len(uniq) != 1:
        return False
    target_bn = uniq[0]
    hints = _metric_hint_tokens(slots, f"{current_plan_step} {question}")
    for q in sql_queries or []:
        if not isinstance(q, dict):
            continue
        out = str(q.get("output_filename") or "").strip()
        if not out or Path(out).name != target_bn:
            continue
        sql_text = str(q.get("sql") or "")
        sel = _extract_select_clause(sql_text)
        if not sel:
            continue
        if re.match(r"^\s*\*\s*$", sel):
            continue
        items = _split_select_list_items(sel)
        if len(items) < 2:
            continue
        aliases = [_alias_from_select_item(x) for x in items]
        if _wide_table_has_ndvi_and_proportion_columns(aliases):
            return True
        if hints:
            matched = 0
            for a in aliases:
                if not a:
                    continue
                if any(h in a or a in h for h in hints if len(h) > 1):
                    matched += 1
            if matched >= 2:
                return True
        substantive = [
            a
            for a in aliases
            if a and a not in _TRIVIAL_SELECT_NAMES and not a.endswith("_id") and len(a) > 1
        ]
        if len(set(substantive)) >= 2:
            return True
    return False


def _python_bivariate_style_task(current_plan_step: str, question: str, slots: dict[str, Any]) -> bool:
    """回归/相关性/双指标等通常需要多文件或宽表。"""
    step = str(current_plan_step or "")
    q = f"{step} {question or ''}"
    low = q.lower()
    is_cluster = "聚类" in q or "cluster" in low or "k-means" in low or "kmeans" in low
    is_geometry_cluster = is_cluster and (
        any(token in q for token in ("几何中心", "质心", "中心坐标", "空间坐标"))
        or any(token in low for token in ("geometry", "centroid", "coordinate"))
    )
    if is_geometry_cluster:
        # 空间聚类可由一个几何文件派生二维质心坐标，不应误判为缺少第二个数据文件。
        return False
    zh_keys = ("回归", "相关性", "相关系数", "皮尔逊", "双变量", "协方差", "聚类")
    if any(k in q for k in zh_keys):
        return True
    en_keys = ("regression", "correlation", "pearson", "bivariate", "covariance", "cluster")
    return any(k in low for k in en_keys)


def _validate_python_input_completeness(
    *,
    current_plan_step: str,
    question: str,
    slots: dict[str, Any],
    geojson_paths: list[str],
    sql_queries: list[Any],
) -> str | None:
    """不满足时返回错误文案；满足则 None。"""
    if not _python_bivariate_style_task(current_plan_step, question, slots):
        return None
    basenames = []
    for p in geojson_paths:
        bn = Path(str(p)).name.strip()
        if bn:
            basenames.append(bn)
    uniq_files = list(dict.fromkeys(basenames))
    if len(uniq_files) >= 2:
        return None
    out_names: list[str] = []
    for q in sql_queries or []:
        if not isinstance(q, dict):
            continue
        o = str(q.get("output_filename") or "").strip()
        if o:
            out_names.append(Path(o).name)
    if len(set(out_names)) >= 2:
        return None
    if any(k in str(current_plan_step) for k in ("联合", "多个文件", "多文件", "宽表", "一次性导出")):
        return None
    if _python_task_inputs_satisfied_by_single_wide_table(
        geojson_paths=geojson_paths,
        sql_queries=sql_queries,
        current_plan_step=current_plan_step,
        question=question,
        slots=slots,
    ):
        return None
    return (
        "当前 Python 步骤（回归/相关性/双指标等）需要至少 2 个输入数据文件，或一次 text2sql 导出多个文件，"
        "或单文件 SQL 已选出至少两个指标相关列；检测到数据不完整，请先完成剩余 text2sql_tool 步骤后再调用 python_analysis_tool。"
    )


_STEP_PROGRESS_BLUEPRINT_FOOTNOTE = "（蓝图可参考；证据足够时可跳过剩余步直接结案。）"


def _build_step_progress_hint(
    plan_steps: list[str], messages: list, plan_meta: list[dict[str, Any]] | None = None
) -> str:
    """根据蓝图与已成功 text2sql 次数生成简短进度提示，引导 Executor 对齐当前步。"""
    foot = _STEP_PROGRESS_BLUEPRINT_FOOTNOTE
    if not plan_steps:
        return f"（暂无多步蓝图；按问题与 Schema 分步取数即可。）{foot}"
    total = len(plan_steps)
    done_sql = _count_successful_tool_calls(messages, "text2sql_tool")
    done_python = _count_successful_tool_calls(messages, "python_analysis_tool")
    expected_sql = _plan_text2sql_step_count(plan_meta)
    completed_steps = min(done_sql + done_python, total)
    k = min(completed_steps + 1, total)
    if expected_sql is not None:
        return (
            f"蓝图中 text2sql_tool 步骤共 {expected_sql} 次，当前已成功 {done_sql} 次。"
            f"python_analysis_tool 已成功 {done_python} 次。"
            f"请优先对齐蓝图第 {k} 步（全蓝图共 {total} 步）；本次 text2sql 仅传该步涉及的表名。"
            f"{foot}"
        )
    return f"请优先对齐蓝图第 {k} 步（全蓝图共 {total} 步）。{foot}"


def _get_current_step_index(state: dict[str, Any]) -> int:
    """获取当前显式维护的蓝图步骤索引（0-based）。"""
    raw = state.get("current_plan_step_index")
    if isinstance(raw, int) and raw > 0:
        return raw - 1
    return 0


def _get_current_step_text(state: dict[str, Any]) -> str:
    """基于显式步骤索引获取当前蓝图步骤文本。"""
    plan = state.get("plan")
    plan_list = plan if isinstance(plan, list) else []
    idx = _get_current_step_index(state)
    if 0 <= idx < len(plan_list):
        return str(plan_list[idx])
    return ""


def _current_blueprint_step_is_python_tool_step(state: dict[str, Any]) -> bool:
    """当前游标是否落在结构化蓝图的 Python 步。"""
    return _planned_tool_name_for_current_step(state) == "python_analysis_tool"


def _resolve_text2sql_task(
    state: dict[str, Any], args_question: str | None
) -> tuple[str, str | None]:
    """
    解析 Text2SQL 子任务与全局背景。
    优先信任 Agent 动态传入的 question：回溯或主动调度时 args_question 高于静态蓝图当前步；
    无参数时再回退蓝图当前步。
    """
    eff = _get_effective_question(state)
    ctx = eff or None

    args_q_str = str(args_question or "").strip()
    plan = state.get("plan")

    if plan and isinstance(plan, list):
        step_task = str(_get_current_step_text(state) or "").strip()
        if args_q_str and args_q_str != step_task:
            print("  [Text2SQL] 识别到 Agent 动态指令，已覆盖静态蓝图任务。", flush=True)
            return args_q_str, ctx
        return step_task or eff, ctx

    return args_q_str or eff, ctx


def _text2sql_forced_question_arg(state: dict[str, Any]) -> str:
    """护栏/续步强制 text2sql 时写入 tool args 的 question：有 plan 用当前步，否则用有效用户问句。"""
    plan = state.get("plan")
    plan_list = plan if isinstance(plan, list) else []
    if plan_list:
        step = str(_get_current_step_text(state) or "").strip()
        if step:
            return step
    return _get_effective_question(state)


def _infer_python_analysis_contract(state: dict[str, Any], messages: list, current_step: str) -> dict[str, Any]:
    """根据槽位、蓝图步骤与上下文推导 Python 分析语义契约。"""
    slots = state.get("slots") or {}
    question = str(_get_effective_question(state) or "")
    step_text = str(current_step or "")
    plan_list = state.get("plan") if isinstance(state.get("plan"), list) else []
    plan_blob = " ".join(str(p) for p in (plan_list or []))
    # 当前步 + 用户问 + 全蓝图：避免后续步骤文案不含「几何中心」时仍误杀 centroid（与蓝图第 2 步一致）
    lowered = f"{question} {step_text} {plan_blob}".lower()

    # 用户明确要求「几何中心/质心」时，用 centroid 做测地距离是语义正确实现，不是对面要素的近似代理。
    centroid_tokens = (
        "几何中心",
        "几何中心点",
        "中心点",
        "质心",
        "centroid",
        "geometric center",
    )
    explicit_centroid_reference = any(token in lowered for token in centroid_tokens)

    contract: dict[str, Any] = {
        "required_operations": [],
        "requires_subset_filtering": False,
        "requires_key_join": False,
        "expected_join_keys": [],
        "forbid_point_proxy": False,
        "explicit_centroid_reference": explicit_centroid_reference,
        "answer_projection": (
            (state.get("execution_contract") or {}).get("answer_projection")
            if isinstance(state.get("execution_contract"), dict)
            else None
        )
        or build_answer_projection_contract(question, slots),
    }

    spatial_predicate = str(slots.get("spatial_predicate") or "").lower()
    threshold = str(slots.get("spatial_threshold") or "").lower()

    if any(token in lowered for token in ["距离", "distance", "within_distance", "公里", "km"]) or threshold:
        contract["required_operations"].append("distance_filter")
        contract["requires_subset_filtering"] = True
        contract["forbid_point_proxy"] = True

    if any(token in lowered for token in ["intersect", "within", "contains", "buffer", "相交", "包含", "缓冲区"]) or spatial_predicate in {"intersects", "contains", "within", "buffer", "overlay"}:
        contract["required_operations"].append("topology_filter")
        contract["requires_subset_filtering"] = True
        contract["forbid_point_proxy"] = True

    sql_queries = state.get("sql_queries") or []
    candidate_keys: list[str] = []
    for q in sql_queries:
        if not isinstance(q, dict):
            continue
        sql_text = str(q.get("sql") or "")
        for key in ["asdf_id", "shapeName", "cell_id", "grid_id", "id"]:
            if key.lower() in sql_text.lower() and key not in candidate_keys:
                candidate_keys.append(key)
    if candidate_keys:
        contract["requires_key_join"] = True
        contract["expected_join_keys"] = candidate_keys

    contract["required_operations"] = list(dict.fromkeys(contract["required_operations"]))
    # 用户明确要「质心/几何中心」时：无条件允许点代理（不因同时存在拓扑/距离等操作而连坐禁止 centroid）
    if explicit_centroid_reference:
        contract["forbid_point_proxy"] = False
    return contract


def _normalize_python_analysis_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """清洗 Python 分析契约，避免历史残留导致过度拦截。"""
    if not isinstance(contract, dict):
        return {}
    out = dict(contract)
    required_ops = [str(op) for op in (out.get("required_operations") or []) if str(op).strip()]
    out["required_operations"] = list(dict.fromkeys(required_ops))

    # 兼容老契约：已废弃 aggregate_after_filter，统一移除，避免无意义的硬拦截。
    out["required_operations"] = [op for op in out["required_operations"] if op != "aggregate_after_filter"]

    requires_key_join = bool(out.get("requires_key_join"))
    has_spatial_subset_driver = any(op in {"distance_filter", "topology_filter"} for op in out["required_operations"])
    # 仅当存在明确空间筛选驱动或实体关联约束时，subset_filtering 才有意义。
    if not has_spatial_subset_driver and not requires_key_join:
        out["requires_subset_filtering"] = False
    else:
        out["requires_subset_filtering"] = bool(out.get("requires_subset_filtering"))
    return out


def _get_last_successful_tool_payload(messages: list, tool_name: str) -> dict[str, Any] | None:
    """获取最近一次成功工具调用的结构化 payload。"""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != tool_name:
            continue
        if additional.get("success") is not True:
            continue
        payload = additional.get("payload")
        if isinstance(payload, dict):
            return payload
    return None


def _get_last_tool_message(messages: list) -> ToolMessage | None:
    """获取最近一条工具消息。"""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg
    return None


def _extract_recent_error(messages: list, tool_name: str) -> str | None:
    """从最近一次工具消息中提取错误信息。"""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != tool_name:
            continue
        payload = additional.get("payload")
        if isinstance(payload, dict):
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                return str(errors[-1])
            error = payload.get("error")
            if error:
                return str(error)
        content = _normalize_ai_content(getattr(msg, "content", ""))
        if content:
            return content[:2000]
    return None


def _count_failed_tool_calls(messages: list, tool_name: str) -> int:
    """统计目标工具连续失败次数。"""
    count = 0
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != tool_name:
            continue
        if additional.get("success") is True:
            break
        count += 1
    return count


def _merge_unique_table_names(*groups: list[str]) -> list[str]:
    """按出现顺序合并表名列表。"""
    merged: list[str] = []
    for group in groups:
        for table_name in group or []:
            if isinstance(table_name, str) and table_name and table_name not in merged:
                merged.append(table_name)
    return merged


def _merge_schema_yaml(existing_yaml: str, incoming_yaml: str) -> str:
    """合并同一张表的裁剪 Schema，补充新增高分列而不丢失已有关键列。"""
    try:
        existing_data = yaml.safe_load(existing_yaml) or {}
    except Exception:
        existing_data = {}
    try:
        incoming_data = yaml.safe_load(incoming_yaml) or {}
    except Exception:
        incoming_data = {}

    if not isinstance(existing_data, dict):
        return incoming_yaml
    if not isinstance(incoming_data, dict):
        return existing_yaml

    merged_data = dict(existing_data)
    for key, value in incoming_data.items():
        if key == "columns":
            continue
        if value not in (None, "", [], {}):
            merged_data[key] = value

    merged_columns: dict[str, dict[str, Any]] = {}
    ordered_column_names: list[str] = []

    def _upsert_column(column: Any) -> None:
        if not isinstance(column, dict):
            return
        col_name = str(column.get("name") or column.get("column_name") or "").strip()
        if not col_name:
            return
        if col_name not in ordered_column_names:
            ordered_column_names.append(col_name)
            merged_columns[col_name] = dict(column)
            return
        merged = dict(merged_columns[col_name])
        for field, field_value in column.items():
            if field_value not in (None, "", [], {}) or field not in merged:
                merged[field] = field_value
        merged_columns[col_name] = merged

    for column in existing_data.get("columns", []) or []:
        _upsert_column(column)
    for column in incoming_data.get("columns", []) or []:
        _upsert_column(column)

    merged_data["columns"] = [merged_columns[name] for name in ordered_column_names]
    return yaml.dump(merged_data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _merge_schema_lists(existing_schemas: list[str], incoming_schemas: list[str]) -> list[str]:
    """按表名合并 schema 列表，重复表执行列级并集。"""
    merged_by_table: dict[str, str] = {}
    ordered_tables: list[str] = []

    for schema_yaml in [*(existing_schemas or []), *(incoming_schemas or [])]:
        table_name = extract_table_name_from_schema_yaml(schema_yaml)
        if not table_name:
            continue
        if table_name not in merged_by_table:
            ordered_tables.append(table_name)
            merged_by_table[table_name] = schema_yaml
            continue
        merged_by_table[table_name] = _merge_schema_yaml(merged_by_table[table_name], schema_yaml)

    return [merged_by_table[table_name] for table_name in ordered_tables]


def _collect_known_table_names(state: dict[str, Any], messages: list) -> list[str]:
    """从 state.schemas 与历史工具结果中汇总可用表名。"""
    table_names: list[str] = []

    for table_name in state.get("retrieved_table_names", []) or []:
        if isinstance(table_name, str) and table_name not in table_names:
            table_names.append(table_name)

    for schema_yaml in state.get("schemas", []) or []:
        table_name = extract_table_name_from_schema_yaml(schema_yaml)
        if table_name and table_name not in table_names:
            table_names.append(table_name)

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        payload = additional.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_tables = payload.get("table_names")
        if isinstance(payload_tables, list):
            for table_name in payload_tables:
                if isinstance(table_name, str) and table_name not in table_names:
                    table_names.append(table_name)

    return table_names


def _build_retry_context_for_text2sql(messages: list) -> str:
    """汇总最近失败 SQL 与错误，帮助 Text2SQL 在重试时进行对比学习。"""
    failed_contexts: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        additional = getattr(msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") != "text2sql_tool":
            continue
        payload = additional.get("payload")
        if not isinstance(payload, dict):
            continue
        errors = payload.get("errors") or []
        queries = payload.get("queries") or []
        if not errors:
            continue
        snippet_lines = ["最近一次失败样本："]
        for idx, query in enumerate(queries[:2], start=1):
            if isinstance(query, dict) and query.get("sql"):
                snippet_lines.append(f"SQL{idx}: {query['sql']}")
        for idx, err in enumerate(errors[:2], start=1):
            snippet_lines.append(f"错误{idx}: {err}")
        failed_contexts.append("\n".join(snippet_lines))
        if len(failed_contexts) >= 2:
            break
    return "\n\n".join(reversed(failed_contexts))


def _build_agent_ai_message(
    response_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    explicit_final: bool = False,
) -> dict[str, Any]:
    """构造统一格式的 AIMessage 输出（追加新消息，不覆盖历史 assistant 内容）。"""
    normalized_tool_calls = tool_calls or []
    parsed: dict[str, Any] = {
        "type": "tool_calls" if normalized_tool_calls else "final",
        "tool_calls": normalized_tool_calls,
        "output": response_text,
    }
    if explicit_final:
        parsed["explicit_final"] = True
    ai_kwargs: dict[str, Any] = {
        "content": response_text,
        "additional_kwargs": {"parsed": parsed},
        "tool_calls": normalized_tool_calls,
    }
    ai_msg = AIMessage(**ai_kwargs)
    return {"messages": [ai_msg]}


_STRICT_FINAL_SYSTEM_PROMPT = """你是 GIS 问答系统的最终答案结构化器。
你只能根据给定的候选答案整理数据，不得补充、猜测或重新计算事实。
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
"""


def _strict_final_updates(answer: StrictFinalAnswer) -> dict[str, Any]:
    """把通过 Pydantic 校验的答案写入消息与显式状态字段。"""
    answer_json = serialize_strict_answer(answer)
    return {
        **_build_agent_ai_message(answer_json, explicit_final=True),
        "final_answer": answer_json,
        "final_answer_payload": answer.model_dump(mode="json"),
        "final_answer_schema_valid": True,
        "final_answer_schema_error": None,
    }


def _schema_failure_updates(candidate_text: str, exc: Exception) -> dict[str, Any]:
    """结构化输出失败时显式标错；评价脚本会将其计为主指标失败。"""
    error = f"{type(exc).__name__}: {exc}"
    msg = f"执行终止：最终答案未通过结构化输出约束（{type(exc).__name__}）。"
    return {
        **_build_agent_ai_message(msg, explicit_final=True),
        "final_answer": msg,
        "final_answer_payload": None,
        "final_answer_schema_valid": False,
        "final_answer_schema_error": error[:2000],
        "errors": [error],
    }


def _constrain_candidate_final_answer(
    *,
    base_llm: Any,
    question: str,
    candidate_text: str,
    evidence_text: str = "",
) -> StrictFinalAnswer:
    """优先校验原生严格 JSON，否则通过 Pydantic structured output 做一次结案整理。"""
    stripped = str(candidate_text or "").strip()
    if stripped:
        try:
            return coerce_strict_answer(stripped)
        except (ValidationError, ValueError, TypeError):
            pass

    # 当前兼容网关可能不提供 response_format/json_schema，但已支持原生 Tool Calling。
    # 显式使用 function_calling，仍由同一份 Pydantic JSON Schema 约束并在返回后再次校验。
    formatter_llm = base_llm
    if hasattr(base_llm, "model_copy"):
        existing_extra = getattr(base_llm, "extra_body", None)
        formatter_llm = base_llm.model_copy(
            update={
                "extra_body": {
                    **(existing_extra if isinstance(existing_extra, dict) else {}),
                    # DeepSeek V4: tool_choice 不能用于 thinking 模式；官方 OpenAI
                    # 兼容参数为 {"thinking": {"type": "disabled"}}。
                    "thinking": {"type": "disabled"},
                    # 同时兼容使用 enable_thinking 布尔开关的 Qwen 网关。
                    "enable_thinking": False,
                }
            }
        )
    structured_llm = formatter_llm.with_structured_output(
        StrictFinalAnswer,
        method="function_calling",
    )
    user_prompt = (
        f"用户问题：\n{question}\n\n"
        f"工具结构化证据（若非空，以此处字段和值为准）：\n"
        f"{str(evidence_text or '').strip() or '（空）'}\n\n"
        f"候选答案（只用于理解如何回答，不得覆盖证据精度）：\n{stripped or '（空）'}"
    )
    result = structured_llm.invoke(
        [
            SystemMessage(content=_STRICT_FINAL_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return coerce_strict_answer(result)


def _latest_finalization_evidence(messages: list[Any]) -> str:
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
            previews: list[Any] = []
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


def _latest_single_sql_data_payload(messages: list[Any]) -> tuple[bool, Any]:
    """Return one complete typed SCGA result from the latest successful call."""
    last_tool = _get_last_tool_message(messages)
    if not isinstance(last_tool, ToolMessage):
        return False, None
    additional = getattr(last_tool, "additional_kwargs", {}) or {}
    if additional.get("tool_name") != "text2sql_tool" or additional.get("success") is not True:
        return False, None
    payload = additional.get("payload")
    if not isinstance(payload, dict):
        return False, None
    results = [item for item in (payload.get("sql_results") or []) if isinstance(item, dict)]
    if len(results) != 1 or "data_payload" not in results[0]:
        return False, None
    return True, results[0].get("data_payload")


def _should_trip_python_semantic_breaker(state: dict[str, Any]) -> bool:
    """当同一步骤上的 Python 分析连续失败达到阈值时，触发断路器（不仅限语义错误）。"""
    if state.get("last_failure_type") == "python_needs_sql_inputs":
        return False
    step_idx = state.get("last_failure_step_index")
    if not isinstance(step_idx, int):
        return False
    counts = state.get("step_failure_counts") or {}
    return int(counts.get(str(step_idx), 0)) >= 3  # 容忍 2 次重试，第 3 次失败熔断


def _toolmessage_failure_code(msg: ToolMessage) -> str | None:
    add = getattr(msg, "additional_kwargs", {}) or {}
    fc = add.get("failure_code")
    if fc:
        return str(fc).strip()
    pl = add.get("payload")
    if isinstance(pl, dict) and pl.get("failure_code"):
        return str(pl.get("failure_code")).strip()
    return None


def _is_skip_parallel_tool_message(msg: Any) -> bool:
    """并行工具被跳过时的占位 ToolMessage，对推理价值低。"""
    if not isinstance(msg, ToolMessage):
        return False
    extra = getattr(msg, "additional_kwargs", {}) or {}
    if extra.get("parallel_skipped") is True:
        return True
    fc = str(extra.get("failure_code") or "").strip()
    if fc == FAILURE_CODE_SERIAL_EXECUTION_IGNORED:
        return True
    c = str(getattr(msg, "content", "") or "")
    if "[System] 该工具调用已被系统忽略" in c:
        return True
    if "本调用已跳过" in c or "被忽略" in c:
        return True
    return "已选择" in c and "跳过" in c


def _last_failed_python_analysis_failure_code(messages: list[Any]) -> str | None:
    """从末尾向前，最近一条失败的 python_analysis_tool 的 failure_code（跳过占位与成功条）。"""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        if _is_skip_parallel_tool_message(msg):
            continue
        add = getattr(msg, "additional_kwargs", {}) or {}
        if add.get("parallel_skipped") is True:
            continue
        if add.get("tool_name") != "python_analysis_tool":
            continue
        if add.get("success") is not False:
            continue
        return _toolmessage_failure_code(msg)
    return None


def _line_looks_like_raw_sql(text: str) -> bool:
    """检测 question 是否以 SQL 开头，或任意一行在去掉空白后以 SELECT/WITH 开头。"""
    t = (text or "").strip()
    if re.match(r"^(select|with)\b", t, re.IGNORECASE):
        return True
    for line in t.splitlines():
        s = line.strip()
        if s and re.match(r"^(select|with)\b", s, re.IGNORECASE):
            return True
    return False


def _text2sql_backtrack_allowed_after_tool_failures(messages: list[Any]) -> bool:
    """末尾 AIMessage 紧前连续 ToolMessage 块中是否存在真实失败（success=False，跳过并行占位）。"""
    if len(messages) < 2 or not isinstance(messages[-1], AIMessage):
        return False
    i = len(messages) - 2
    while i >= 0:
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            break
        if _is_skip_parallel_tool_message(msg):
            i -= 1
            continue
        add = getattr(msg, "additional_kwargs", {}) or {}
        if add.get("parallel_skipped") is True:
            i -= 1
            continue
        if add.get("success") is False:
            return True
        i -= 1
    return False


def _count_tail_python_needs_sql_failures(messages: list[Any]) -> int:
    """从消息末尾向前，连续统计 failure_code 为需补 SQL 的 python_analysis 失败次数（中间可隔 AI/Human）。"""
    n = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        if _is_skip_parallel_tool_message(msg):
            continue
        add = getattr(msg, "additional_kwargs", {}) or {}
        if add.get("parallel_skipped") is True:
            continue
        if add.get("tool_name") != "python_analysis_tool":
            break
        if add.get("success") is not False:
            break
        if _toolmessage_failure_code(msg) == FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS:
            n += 1
        else:
            break
    return n


def _count_consecutive_text2sql_failures_from_end(messages: list[Any]) -> int:
    """从对话末尾向前统计连续的 text2sql_tool 失败条数（跳过并行占位；遇成功或其它真实工具则停止）。"""
    n = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        if _is_skip_parallel_tool_message(msg):
            continue
        add = getattr(msg, "additional_kwargs", {}) or {}
        if add.get("parallel_skipped") is True:
            continue
        if add.get("tool_name") != "text2sql_tool":
            break
        if add.get("success") is True:
            break
        n += 1
    return n


def _should_trip_text2sql_breaker(state: dict[str, Any]) -> bool:
    lim = int(getattr(config, "TEXT2SQL_CONSECUTIVE_FAILURE_LIMIT", 10) or 10)
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        return False
    current_idx = _get_current_step_index(state)
    counts = state.get("step_failure_counts") or {}
    same_step_failures = int(counts.get(f"text2sql:{current_idx}", 0) or 0)
    return max(
        _count_consecutive_text2sql_failures_from_end(messages),
        same_step_failures,
    ) >= lim


def _build_text2sql_breaker_final_message(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    last_errors: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        add = getattr(msg, "additional_kwargs", {}) or {}
        if add.get("tool_name") != "text2sql_tool" or add.get("success") is True:
            continue
        c = str(getattr(msg, "content", "") or "").strip()
        if c:
            last_errors.append(c[:500])
        if len(last_errors) >= 3:
            break
    lim = int(getattr(config, "TEXT2SQL_CONSECUTIVE_FAILURE_LIMIT", 10) or 10)
    blob = "\n---\n".join(reversed(last_errors)) if last_errors else "（无详情）"
    return (
        f"执行终止：同一蓝图步骤的 text2sql_tool 已失败 {lim} 次，已触发熔断以避免与 Schema 检索/护栏形成死循环。\n"
        f"最近错误摘要：\n{blob}\n"
        "建议：拆成更小的 SQL 子问题、核对指标列是否分属不同表并正确 JOIN，或稍后重试。"
    )


def _parse_tool_payload(result: Any) -> tuple[str, dict[str, Any] | None, bool]:
    """解析工具返回值，提取文本、结构化 payload 与成功标记。"""
    text_result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
    payload = None
    success = True

    try:
        payload = json.loads(text_result) if isinstance(text_result, str) else result
    except Exception:
        payload = result if isinstance(result, dict) else None

    if isinstance(payload, dict):
        if "success" in payload:
            success = bool(payload.get("success"))
        elif payload.get("errors"):
            success = False
        elif payload.get("error"):
            success = False
    else:
        success = False

    if isinstance(result, dict):
        text_result = json.dumps(result, ensure_ascii=False, indent=2)

    return text_result, payload if isinstance(payload, dict) else None, success


def _advance_plan_step_after_success(state: dict[str, Any]) -> dict[str, Any]:
    """工具成功后将蓝图游标推进恰好一步（current_plan_step_index 保持 1-based 约定）。"""
    plan_list = state.get("plan") if isinstance(state.get("plan"), list) else []
    if not plan_list:
        return {}
    current_idx = _get_current_step_index(state)
    next_idx = current_idx + 1
    if next_idx < len(plan_list):
        return {
            "current_plan_step_index": next_idx + 1,
            "current_plan_step": str(plan_list[next_idx]),
        }
    return {
        "current_plan_step_index": len(plan_list) + 1,
        "current_plan_step": "",
    }


def _current_step_state_update(state: dict[str, Any]) -> dict[str, Any]:
    """返回当前步的标准化状态，避免失败分支丢失当前步骤上下文。"""
    current_step = _get_current_step_text(state)
    current_idx = _get_current_step_index(state)
    return {
        "current_plan_step_index": current_idx + 1 if current_step else None,
        "current_plan_step": current_step or None,
    }


def _parse_tool_result_to_state(tool_name: str, payload: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
    """将工具结果写回结构化 state。"""
    updates: dict[str, Any] = {}
    if not payload:
        return updates

    if tool_name == "schema_search_tool":
        retrieved = payload.get("schemas") or []
        merged = _merge_schema_lists(list(retrieved), list(state.get("schemas", []) or []))
        updates["schemas"] = merged
        updates["schemas_yaml"] = build_planner_schemas_yaml_from_rag_list(merged)
        if "table_names" in payload:
            updates["retrieved_table_names"] = _merge_unique_table_names(
                list(state.get("retrieved_table_names", []) or []),
                list(payload.get("table_names") or []),
            )
        if "semantic_bindings" in payload:
            existing = [item for item in (state.get("schema_bindings") or []) if isinstance(item, dict)]
            incoming = [item for item in (payload.get("semantic_bindings") or []) if isinstance(item, dict)]
            keyed: dict[tuple[str, str], dict[str, Any]] = {
                (str(item.get("concept") or ""), str(item.get("table") or "")): item
                for item in [*existing, *incoming]
            }
            updates["schema_bindings"] = list(keyed.values())
        if "schema_coverage" in payload:
            updates["schema_coverage"] = dict(payload.get("schema_coverage") or {})

    elif tool_name == "text2sql_tool":
        is_success = not (payload.get("errors") or payload.get("error")) and payload.get("success", True)
        updates.update(_current_step_state_update(state))
        current_idx = _get_current_step_index(state)
        counts = dict(state.get("step_failure_counts") or {})
        failure_key = f"text2sql:{current_idx}"
        if "geojson_paths" in payload:
            updates["geojson_paths"] = payload["geojson_paths"]
        if "queries" in payload:
            updates["sql_queries"] = payload["queries"]
        if "sql_results" in payload:
            updates["sql_results"] = payload["sql_results"]
        if "table_names" in payload:
            updates["retrieved_table_names"] = _merge_unique_table_names(
                list(state.get("retrieved_table_names", []) or []),
                list(payload.get("table_names") or []),
            )
        if "errors" in payload:
            errs = [e for e in payload["errors"] if e]
            if errs:
                updates["errors"] = errs
        # 快照：成功时记录本批次 geojson_paths，失败时清空，防止 Python 节点误用旧批次文件
        if is_success:
            existing_snapshot = list(state.get("latest_text2sql_geojson_paths") or state.get("geojson_paths") or [])
            latest_batch = list(payload.get("geojson_paths") or [])
            merged_snapshot: list[str] = []
            for p in [*existing_snapshot, *latest_batch]:
                if p and p not in merged_snapshot:
                    merged_snapshot.append(p)
            updates["latest_text2sql_geojson_paths"] = merged_snapshot
            updates.update(_advance_plan_step_after_success(state))
            updates["last_failure_type"] = None
            updates["last_failure_step_index"] = None
            counts[failure_key] = 0
            updates["step_failure_counts"] = counts
        else:
            updates["latest_text2sql_geojson_paths"] = []
            counts[failure_key] = int(counts.get(failure_key, 0)) + 1
            updates["step_failure_counts"] = counts
            updates["last_failure_type"] = "text2sql_runtime_or_semantic"
            updates["last_failure_step_index"] = current_idx

    elif tool_name == "python_analysis_tool":
        is_success = not (payload.get("errors") or payload.get("error")) and payload.get("success", True)
        fc_raw = str(payload.get("failure_code") or "").strip()
        needs_sql_inputs = fc_raw == FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS
        updates.update(_current_step_state_update(state))
        if "llm2code_info" in payload:
            updates["llm2code_info"] = payload["llm2code_info"]
        if "code_output" in payload:
            updates["code_output"] = payload["code_output"]
        if "code" in payload:
            updates["code"] = payload["code"]
        if "error" in payload:
            err_text = str(payload["error"] or "").strip()
            if err_text:
                updates["errors"] = [err_text]
        current_idx = _get_current_step_index(state)
        counts = dict(state.get("step_failure_counts") or {})
        if is_success:
            saved_files = [
                str(item).strip()
                for item in (payload.get("saved_files") or [])
                if str(item).strip()
            ]
            if saved_files:
                updates["analysis_artifact_paths"] = saved_files
            updates.update(_advance_plan_step_after_success(state))
            updates["last_failure_type"] = None
            updates["last_failure_step_index"] = None
            counts[str(current_idx)] = 0
            updates["step_failure_counts"] = counts
            plan_list = state.get("plan") if isinstance(state.get("plan"), list) else []
            pm = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else []
            if (
                current_idx < len(pm)
                and isinstance(pm[current_idx], dict)
                and pm[current_idx].get("final_python_step")
            ):
                updates["current_plan_step_index"] = len(plan_list) + 1
                updates["current_plan_step"] = ""
                updates["final_answer_ready"] = True
            else:
                updates["final_answer_ready"] = False
        else:
            if needs_sql_inputs:
                updates["step_failure_counts"] = counts
                updates["last_failure_type"] = "python_needs_sql_inputs"
                updates["last_failure_step_index"] = current_idx
            else:
                counts[str(current_idx)] = int(counts.get(str(current_idx), 0)) + 1
                updates["step_failure_counts"] = counts
                updates["last_failure_type"] = "python_runtime_or_other"
                updates["last_failure_step_index"] = current_idx

    return updates


def _trim_toolmessage_content(msg: ToolMessage, max_chars: int) -> ToolMessage:
    """压缩过长工具正文：默认 20% 头 + 80% 尾；text2sql 成功时改为保留前部，避免截掉中间的 data_peek 表格。"""
    content = str(getattr(msg, "content", "") or "")
    if len(content) <= max_chars:
        return msg

    extra = getattr(msg, "additional_kwargs", None) or {}
    tool_name = str((extra or {}).get("tool_name") or "")
    success = bool((extra or {}).get("success"))
    if tool_name == "text2sql_tool" and success:
        tail_note = "\n\n...[text2sql 正文过长已截断；请仅依据上文 answer_hint / data_peek 回答]..."
        room = max(0, max_chars - len(tail_note))
        new_content = content[:room] + tail_note
    else:
        sep = "\n\n...[中间堆栈/明细已截断]...\n\n"
        budget = max_chars - len(sep)
        if budget < 1:
            new_content = content[:max_chars]
        else:
            head_len = int(budget * 0.2)
            tail_len = budget - head_len
            new_content = content[:head_len] + sep + content[-tail_len:]

    extra = getattr(msg, "additional_kwargs", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    pl = extra.get("payload")
    payload = pl if isinstance(pl, dict) else None
    fc_raw = str(extra.get("failure_code") or "").strip()
    return tool_observation(
        content=new_content,
        tool_call_id=getattr(msg, "tool_call_id", "") or "",
        tool_name=str(extra.get("tool_name") or "tool"),
        success=bool(extra.get("success")),
        payload=payload,
        failure_code=fc_raw or None,
        parallel_skipped=bool(extra.get("parallel_skipped")),
    )


# 窗口候选优先级：数值越小越优先保留（超 win 条时按此裁剪，避免按时间切尾误删护栏/用户问）
_WIN_PRIO_GUARDRAIL_HUMAN = 0
_WIN_PRIO_LAST_ANCHOR = 1
_WIN_PRIO_USER_HUMAN = 2
_WIN_PRIO_SUCCESS_TOOL = 3
_WIN_PRIO_FAILURE_TOOL = 4
_WIN_PRIO_RECENT_AI = 5


def _tool_call_ids_from_ai_message(msg: AIMessage) -> set[str]:
    out: set[str] = set()
    for tc in getattr(msg, "tool_calls", None) or []:
        if isinstance(tc, dict):
            tid = tc.get("id")
        else:
            tid = getattr(tc, "id", None)
        if tid is not None and str(tid).strip():
            out.add(str(tid))
    return out


def _find_parent_ai_index_for_tool_message(messages: list[Any], tool_index: int) -> int | None:
    """向前查找对该 ToolMessage 的 tool_call_id 发出 tool_calls 的 AIMessage 索引。"""
    if not (0 <= tool_index < len(messages)):
        return None
    tm = messages[tool_index]
    if not isinstance(tm, ToolMessage):
        return None
    want = str(getattr(tm, "tool_call_id", "") or "")
    if not want:
        return None
    for j in range(tool_index - 1, -1, -1):
        mj = messages[j]
        if not isinstance(mj, AIMessage):
            continue
        if want in _tool_call_ids_from_ai_message(mj):
            return j
    return None


def _indices_in_ai_tool_turn(messages: list[Any], ai_index: int) -> set[int]:
    """AIMessage 及其后紧跟、且 tool_call_id 属于该轮 tool_calls 的 ToolMessage 索引（整块）。"""
    if not (0 <= ai_index < len(messages)):
        return set()
    mi = messages[ai_index]
    if not isinstance(mi, AIMessage):
        return set()
    ids = _tool_call_ids_from_ai_message(mi)
    if not ids:
        return {ai_index}
    out = {ai_index}
    j = ai_index + 1
    while j < len(messages):
        m = messages[j]
        if not isinstance(m, ToolMessage):
            break
        tid = str(getattr(m, "tool_call_id", "") or "")
        if tid not in ids:
            break
        out.add(j)
        j += 1
    return out


def _can_append_tool_after_out(out: list[Any], tool_msg: ToolMessage) -> bool:
    """OpenAI/百炼：Tool 前须存在带 tool_calls 的 Assistant；同轮多 Tool 可连续跟在 Assistant 或其它 Tool 后。"""
    want = str(getattr(tool_msg, "tool_call_id", "") or "")
    if not out or not want:
        return False
    k = len(out) - 1
    while k >= 0 and isinstance(out[k], ToolMessage):
        k -= 1
    if k < 0:
        return False
    prev_ai = out[k]
    if not isinstance(prev_ai, AIMessage):
        return False
    return want in _tool_call_ids_from_ai_message(prev_ai)


def _strip_trailing_incomplete_ai_tool_calls(seq: list[Any]) -> list[Any]:
    """去掉末尾「带 tool_calls 且无紧跟 Tool 响应」的 AIMessage，避免网关 400。"""
    seq = list(seq)
    while seq:
        last = seq[-1]
        if not isinstance(last, AIMessage):
            break
        ids = _tool_call_ids_from_ai_message(last)
        if not ids:
            break
        seq.pop()
    return seq


def _window_messages_for_agent_executor(messages: list[Any]) -> list[Any]:
    """按类型选最小必要集；超窗口时按优先级保留，而非按时间切尾。

    保证送入 LLM 的片段满足 OpenAI/百炼：Tool 仅接在带对应 tool_calls 的 Assistant 之后；
    含 tool_calls 的 AIMessage 与同轮 Tool 作为整块保留或剔除，并剔除无法配对父级的 Tool。
    """
    if not isinstance(messages, list) or not messages:
        return []
    win = max(
        int(getattr(config, "AGENT_LLM_MESSAGE_WINDOW_MIN", 4)),
        int(getattr(config, "AGENT_LLM_MESSAGE_WINDOW", 10)),
    )
    max_tool_body = int(getattr(config, "AGENT_LLM_TOOLMESSAGE_MAX_CHARS", 1500))
    n = len(messages)
    prio_by_idx: dict[int, int] = {}

    def _offer(i: int, p: int) -> None:
        if not (0 <= i < n):
            return
        cur = prio_by_idx.get(i)
        if cur is None or p < cur:
            prio_by_idx[i] = p

    _offer(n - 1, _WIN_PRIO_LAST_ANCHOR)

    for i in range(n - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and (getattr(m, "additional_kwargs", {}) or {}).get("guardrail_retry"):
            _offer(i, _WIN_PRIO_GUARDRAIL_HUMAN)
            break

    for i in range(n - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, ToolMessage) or _is_skip_parallel_tool_message(m):
            continue
        if (getattr(m, "additional_kwargs", {}) or {}).get("success") is True:
            _offer(i, _WIN_PRIO_SUCCESS_TOOL)
            break

    for i in range(n - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, ToolMessage) or _is_skip_parallel_tool_message(m):
            continue
        if (getattr(m, "additional_kwargs", {}) or {}).get("success") is False:
            _offer(i, _WIN_PRIO_FAILURE_TOOL)
            break

    for i in range(n - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, HumanMessage):
            continue
        if (getattr(m, "additional_kwargs", {}) or {}).get("guardrail_retry"):
            continue
        _offer(i, _WIN_PRIO_USER_HUMAN)
        break

    for i in range(n - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            _offer(i, _WIN_PRIO_RECENT_AI)
            break

    by_value = sorted(prio_by_idx.keys(), key=lambda idx: (prio_by_idx[idx], -idx))
    keep = by_value[:win] if len(by_value) > win else by_value
    keep_set: set[int] = set(keep)

    for i in list(keep_set):
        if isinstance(messages[i], ToolMessage) and _is_skip_parallel_tool_message(messages[i]):
            keep_set.discard(i)

    for i in list(keep_set):
        if not isinstance(messages[i], ToolMessage) or _is_skip_parallel_tool_message(messages[i]):
            continue
        parent = _find_parent_ai_index_for_tool_message(messages, i)
        if parent is None:
            keep_set.discard(i)
        else:
            keep_set |= _indices_in_ai_tool_turn(messages, parent)

    for i in list(keep_set):
        m = messages[i]
        if not isinstance(m, AIMessage):
            continue
        if not (getattr(m, "tool_calls", None) or []):
            continue
        turn = _indices_in_ai_tool_turn(messages, i)
        tool_part = turn - {i}
        if keep_set & tool_part:
            keep_set |= turn
        else:
            keep_set.discard(i)

    if len(keep_set) > win:
        print(
            f"  [agent] 消息窗口因 tool/assistant 配对由 {win} 条扩展为 {len(keep_set)} 条（满足 API 序约束）",
            flush=True,
        )

    ranked = sorted(keep_set)
    out: list[Any] = []
    for idx in ranked:
        m = messages[idx]
        if isinstance(m, ToolMessage):
            if _is_skip_parallel_tool_message(m):
                continue
            if not _can_append_tool_after_out(out, m):
                continue
            out.append(_trim_toolmessage_content(m, max_tool_body))
            continue
        out.append(m)

    out = _strip_trailing_incomplete_ai_tool_calls(out)
    return out


def _system_suffix_question_anchor_if_truncated(
    state: dict[str, Any],
    *,
    raw_message_count: int,
    window_size: int,
) -> str:
    """历史被截断时把用户问题压进 system，减少窗口内占位。"""
    if raw_message_count <= window_size:
        return ""
    if not getattr(config, "AGENT_LLM_INJECT_QUESTION_ANCHOR", True):
        return ""
    q = str(_get_effective_question(state) or "").strip()
    if not q:
        return ""
    return "\n\n【用户问题锚点】（较早轮次已移出窗口）\n" + q[:500]


def _compute_retry_increment_for_agent_entry(messages: list[Any]) -> int:
    """本次进入 Executor 前，若因上一轮工具失败或显式护栏 HumanMessage 需再打回模型，则计 1。

    排除并行占位被忽略的 ToolMessage（serial_execution_ignored），避免误计。
    """
    if not isinstance(messages, list) or not messages:
        return 0
    last = messages[-1]
    if isinstance(last, HumanMessage):
        return 1 if (getattr(last, "additional_kwargs", {}) or {}).get("guardrail_retry") else 0
    if isinstance(last, ToolMessage):
        ak = getattr(last, "additional_kwargs", {}) or {}
        if ak.get("parallel_skipped") is True:
            return 0
        if ak.get("failure_code") == FAILURE_CODE_SERIAL_EXECUTION_IGNORED:
            return 0
        if ak.get("success") is False:
            return 1
        return 0
    return 0


def agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Executor 节点：以 planner 蓝图为参考，调用工具完成任务；模型可在证据充分时提前输出最终答案。

    本函数会：
    - 注入蓝图、进度提示与 Schema 等上下文；
    - 在上一轮工具失败或显式护栏 HumanMessage 时递增 retry_count，并与 MAX_EXECUTOR_AGENT_RETRIES 比较熔断；
    - 检查消息数量、连续 Python 失败等安全阈值；
    - 基于最近工具调用的结果动态调整状态；
    - 拼接动态 system prompt 注入 LLM。
    """

    # 1. 解析 messages
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    prev_retry = int(state.get("retry_count") or 0)
    delta_retry = _compute_retry_increment_for_agent_entry(messages)
    candidate_retry = prev_retry + delta_retry

    def _attach_retry_if_needed(updates: dict[str, Any]) -> dict[str, Any]:
        if delta_retry > 0:
            updates["retry_count"] = candidate_retry
        return updates

    # 2. 死循环保护：消息条数上限
    if len(messages) > MAX_GRAPH_MESSAGES:
        msg = f"对话消息数已超过上限（{MAX_GRAPH_MESSAGES}），已终止执行以防死循环。"
        return _attach_retry_if_needed({**_build_agent_ai_message(msg, explicit_final=True), "final_answer": msg})

    # 3. 死循环保护：Executor 在「工具失败再打回 / 护栏 HumanMessage」累计超过上限时熔断
    if candidate_retry > MAX_EXECUTOR_AGENT_RETRIES:
        msg = "执行器推理轮次过多，已停止以避免死循环。请简化问题或分步提问。"
        return _attach_retry_if_needed({**_build_agent_ai_message(msg, explicit_final=True), "final_answer": msg})

    # 3b. Text2SQL 连续失败熔断（避免护栏与递归上限死锁）
    if _should_trip_text2sql_breaker(state):
        msg = _build_text2sql_breaker_final_message(state)
        return _attach_retry_if_needed({**_build_agent_ai_message(msg, explicit_final=True), "final_answer": msg})

    # 3c. Python「需补 SQL 输入」类失败连续重复（体验熔断；根因仍靠游标回退）
    stall_lim = int(getattr(config, "PYTHON_INPUT_NEEDS_SQL_STALL_LIMIT", 0) or 0)
    if stall_lim > 0 and isinstance(messages, list):
        if _count_tail_python_needs_sql_failures(messages) >= stall_lim:
            fin = (
                f"执行终止：已连续 {stall_lim} 次因 Python 输入数据不完整被拦截（需先补跑 text2sql_tool 导出）。"
                "请检查蓝图是否要求多文件或宽表 SQL，或简化任务后重试。"
            )
            return _attach_retry_if_needed({**_build_agent_ai_message(fin, explicit_final=True), "final_answer": fin})

    # 4. 死循环保护：同一步 Python 连续失败“断路器”
    if _should_trip_python_semantic_breaker(state):
        last_error = "；".join(state.get("errors") or []) or "Python 分析连续失败。"
        return _attach_retry_if_needed({
            **_build_agent_ai_message(
                f"执行终止：同一蓝图步骤上的 Python 分析已连续失败（含语义或运行时错误），已触发断路器。\n{last_error}",
                explicit_final=True,
            ),
            "final_answer": f"执行终止：同一蓝图步骤上的 Python 分析已连续失败（含语义或运行时错误），已触发断路器。\n{last_error}",
        })

    # 5. 成功的 python_analysis_tool：仅当蓝图游标之后不再包含任何「已注册工具名」步骤时，才用 output 短路结案；
    #    否则继续走 Executor，避免跳过 map_rendering_tool 等后续工具步。
    last_tool_msg = _get_last_tool_message(messages)
    if isinstance(last_tool_msg, ToolMessage):
        additional = getattr(last_tool_msg, "additional_kwargs", {}) or {}
        if additional.get("tool_name") == "python_analysis_tool" and additional.get("success") is True:
            payload = additional.get("payload") if isinstance(additional.get("payload"), dict) else {}
            code_output = str(payload.get("code_output") or state.get("code_output") or "").strip()
            if (
                code_output
                and not _plan_has_remaining_native_tool_steps(state)
                and str(state.get("review_action") or "") != "revise"
            ):
                if "data_payload" not in payload:
                    return _attach_retry_if_needed(
                        _schema_failure_updates(
                            code_output,
                            ValueError("python_analysis_tool 成功回包缺少 data_payload"),
                        )
                    )
                try:
                    original_question = str(state.get("question") or _get_effective_question(state))
                    if bool(getattr(config, "V2_PRESERVE_PYTHON_PAYLOAD", False)):
                        strict_answer = build_strict_answer(
                            payload.get("data_payload"),
                            question=original_question,
                        )
                    elif bool(getattr(config, "V2_EVIDENCE_FIRST_FINALIZATION", False)):
                        strict_answer = constrain_candidate_final_answer(
                            base_llm=config.get_llm(),
                            question=original_question,
                            candidate_text=code_output,
                            evidence_text=json.dumps(
                                {
                                    "tool_name": "python_analysis_tool",
                                    "data_payload": payload.get("data_payload"),
                                },
                                ensure_ascii=False,
                            )[:12000],
                        )
                    else:
                        strict_answer = build_strict_answer(
                            payload.get("data_payload"),
                            question=original_question,
                        )
                except Exception as exc:
                    print(
                        "  [agent:v2] Python 证据结案器不可用，回退为确定性封装："
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    try:
                        strict_answer = build_strict_answer(
                            payload.get("data_payload"),
                            question=str(state.get("question") or _get_effective_question(state)),
                        )
                    except (ValidationError, ValueError, TypeError) as fallback_exc:
                        return _attach_retry_if_needed(
                            _schema_failure_updates(code_output, fallback_exc)
                        )
                print(
                    f"  [agent:v2] Python 证据已整理为 {strict_answer.answer_type} 严格结构化答案",
                    flush=True,
                )
                return _attach_retry_if_needed(_strict_final_updates(strict_answer))

    # A completed SQL-only plan already contains the final database result.
    # When Schema coverage is explicit and complete, pass the typed payload
    # directly to the answer protocol instead of asking Executor + formatter
    # LLMs to copy the same rows (which can drop records or alter precision).
    sql_payload_ready, sql_data_payload = _latest_single_sql_data_payload(messages)
    execution_contract = state.get("execution_contract")
    schema_coverage = (
        execution_contract.get("schema_coverage")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("schema_coverage"), dict)
        else {}
    )
    schema_bindings = (
        execution_contract.get("schema_bindings")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("schema_bindings"), list)
        else []
    )
    if (
        sql_payload_ready
        and not _plan_has_remaining_native_tool_steps(state)
        and str(state.get("review_action") or "") != "revise"
        and bool(schema_bindings)
        and bool(schema_coverage.get("complete"))
        and not list(schema_coverage.get("uncovered_metrics") or [])
    ):
        try:
            strict_answer = build_strict_answer_from_sql_payload(
                sql_data_payload,
                question=str(state.get("question") or _get_effective_question(state)),
            )
            print(
                "  [agent:v2] SCGA typed payload 已确定性封装；跳过自然语言转述与二次格式化 LLM。",
                flush=True,
            )
            return _attach_retry_if_needed(_strict_final_updates(strict_answer))
        except (ValidationError, ValueError, TypeError) as exc:
            print(
                f"  [agent:v2] SCGA typed payload 无法直接封装，保留 Executor 结案路径：{type(exc).__name__}: {exc}",
                flush=True,
            )

    # 结构化蓝图已经给出工具与目标时，首轮调用无需再让 Executor LLM
    # 重复选择工具。只有失败纠错、重规划或最终答案整理才交回模型推理。
    expected_tool = _planned_tool_name_for_current_step(state)
    if (
        bool(getattr(config, "V2_DETERMINISTIC_PLAN_EXECUTION", True))
        and expected_tool in {"text2sql_tool", "python_analysis_tool"}
        and str(state.get("review_action") or "") != "revise"
    ):
        recent_tool = _get_last_tool_message(messages)
        recent_extra = (
            getattr(recent_tool, "additional_kwargs", {}) or {}
            if isinstance(recent_tool, ToolMessage)
            else {}
        )
        if recent_extra.get("success") is not False:
            idx = _get_current_step_index(state)
            plan_meta = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else []
            objective = _get_current_step_text(state)
            if idx < len(plan_meta) and isinstance(plan_meta[idx], dict):
                objective = str(plan_meta[idx].get("objective") or objective).strip()
            args: dict[str, Any] = {"question": objective}
            if expected_tool == "text2sql_tool":
                args["table_names"] = []
            else:
                paths = [
                    Path(str(p)).name
                    for p in [
                        *(state.get("analysis_artifact_paths") or []),
                        *(state.get("geojson_paths") or []),
                    ]
                    if str(p).strip()
                ]
                paths = list(dict.fromkeys(paths))
                if not paths:
                    expected_tool = None
                else:
                    args["geojson_paths"] = paths
            if expected_tool:
                call = {
                    "id": f"v2_plan_{idx + 1}_{expected_tool}",
                    "name": expected_tool,
                    "args": args,
                    "type": "tool_call",
                }
                print(
                    f"  [agent:v2] 结构化蓝图直接调度 {expected_tool}，跳过 Executor LLM。",
                    flush=True,
                )
                return _attach_retry_if_needed(_build_agent_ai_message("", [call]))

    # 6. Executor 只接收执行控制信息。Schema 已由 Text2SQL 节点消费，
    # 不再在每轮 Agent 调度中重复传输。
    geojson_paths = state.get("geojson_paths", [])
    geojson_text = "\n".join([f"- {p}" for p in geojson_paths]) if geojson_paths else "（暂无导出数据文件）"

    plan_steps = state.get("plan") or []
    if plan_steps:
        # 按序号将 plan 拼起来
        plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan_steps, start=1))
    else:
        plan_text = "（暂无执行蓝图，请根据问题自行判断调用顺序。）"

    # 步骤进度提示
    pm = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else None
    step_progress_hint = _build_step_progress_hint(
        list(plan_steps) if plan_steps else [], messages, plan_meta=pm
    )
    # 注意：不要用 .format()，全部用 replace，防止用户 plan/schemas 里有花括号报错
    win = max(
        int(getattr(config, "AGENT_LLM_MESSAGE_WINDOW_MIN", 4)),
        int(getattr(config, "AGENT_LLM_MESSAGE_WINDOW", 10)),
    )
    raw_n = len(messages)
    system_prompt = (
        MASTER_SYSTEM_PROMPT.replace("{plan_text}", plan_text)
        .replace("{geojson_text}", geojson_text)
        .replace("{step_progress_hint}", step_progress_hint)
    )
    system_prompt += _system_suffix_question_anchor_if_truncated(
        state, raw_message_count=raw_n, window_size=win
    )

    # 8. 拼接完整消息流（滑动窗口）；工具列表固定为注册表全量，并发由 API 关闭
    llm_messages = _window_messages_for_agent_executor(messages)
    if raw_n > win:
        print(
            f"  [agent] LLM 历史窗口: 原始消息 {raw_n} 条 → 传入 {len(llm_messages)} 条",
            flush=True,
        )
    executor_tools = _executor_tools_for_state(state)
    base_llm = config.get_llm()
    if not executor_tools:
        llm = base_llm
        print("  [agent:v2] 蓝图工具步骤已完成；关闭工具调用并进入证据结案。", flush=True)
    else:
        print(
            "  [agent:v2] 当前步骤允许工具="
            f"{[str(getattr(tool, 'name', '')) for tool in executor_tools]}",
            flush=True,
        )
        try:
            llm = base_llm.bind_tools(executor_tools, parallel_tool_calls=False)
        except Exception:
            try:
                llm = base_llm.bind_tools(executor_tools)
            except Exception as e2:
                print(f"  [agent] 绑定工具失败。原因：{type(e2).__name__}: {e2}", flush=True)
                llm = base_llm

    full_messages = [SystemMessage(content=system_prompt)] + llm_messages
    try:
        response = llm.invoke(full_messages)
    except Exception as e:
        print(f"  [agent] LLM 调用失败。原因：{type(e).__name__}: {e}", flush=True)
        _raise_if_exception_infra_failure("agent_llm", e)
        msg = (
            f"执行器调用语言模型失败（{type(e).__name__}），请稍后重试或简化问题。"
        )
        return _attach_retry_if_needed({
            **_build_agent_ai_message(msg, explicit_final=True),
            "final_answer": msg,
            "errors": [f"{type(e).__name__}: {e}"],
        })

    # 9. 解析 LLM 返回：仅原生 tool_calls + additional_kwargs（无正文 JSON 兜底）
    response_text = _sanitize_text_for_windows_console(_normalize_ai_content(getattr(response, "content", "")))
    current_step_hint = _get_current_step_text(state)
    tool_calls = merge_native_tool_calls(response)
    if not executor_tools and tool_calls:
        discarded_names = [
            str(call.get("name"))
            for call in tool_calls
            if isinstance(call, dict) and call.get("name")
        ]
        print(
            "  [agent:v2] 当前蓝图不再允许工具，丢弃网关残留 tool_calls="
            f"{discarded_names}",
            flush=True,
        )
        tool_calls = []

    def _is_valid_args(args_dict: dict) -> bool:
        if not isinstance(args_dict, dict) or not args_dict:
            return False
        return any(v not in (None, "", [], {}) for v in args_dict.values())

    tool_calls = [tc for tc in tool_calls if isinstance(tc, dict) and _is_valid_args(tc.get("args"))]

    if len(tool_calls) > 1:
        called_names_all = [str(c.get("name")) for c in tool_calls if isinstance(c, dict) and c.get("name")]
        print(
            f"  [agent] 网关返回 {len(tool_calls)} 个 tool_calls {called_names_all}，仅保留第一条（期望 parallel_tool_calls=false）",
            flush=True,
        )
        tool_calls = [tool_calls[0]]

    if tool_calls:
        called_names = [str(c.get("name")) for c in tool_calls if isinstance(c, dict) and c.get("name")]
        print(f"  [agent] LLM 返回 tool_calls={called_names}", flush=True)
    else:
        preview_limit = int(getattr(config, "AGENT_NO_TOOLCALLS_STDOUT_MAX_CHARS", 2000) or 0)
        if preview_limit <= 0:
            preview = response_text
        else:
            preview = response_text[:preview_limit]
            if len(response_text) > preview_limit:
                preview += f"\n...（截断，原文共 {len(response_text)} 字符）"
        print(
            f"  [agent] LLM 返回无 tool_calls | content_len={len(response_text)}\n{preview}",
            flush=True,
        )

    # 10. 构建更新的 agent 状态
    step_index = _get_current_step_index(state)
    current_step = current_step_hint
    updates = _build_agent_ai_message(response_text, tool_calls)
    updates["current_plan_step"] = current_step or None
    updates["current_plan_step_index"] = step_index + 1 if current_step else None
    if not tool_calls:
        try:
            strict_answer = constrain_candidate_final_answer(
                base_llm=base_llm,
                question=str(state.get("question") or _get_effective_question(state)),
                candidate_text=response_text,
                evidence_text=(
                    _latest_finalization_evidence(messages)
                    if bool(getattr(config, "V2_EVIDENCE_FIRST_FINALIZATION", False))
                    else ""
                ),
            )
            updates = _strict_final_updates(strict_answer)
            updates["current_plan_step"] = current_step or None
            updates["current_plan_step_index"] = step_index + 1 if current_step else None
            print(
                f"  [agent] 候选答案已通过 Pydantic 约束：answer_type={strict_answer.answer_type}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"  [agent] 最终答案结构化失败。原因：{type(exc).__name__}: {exc}",
                flush=True,
            )
            updates = _schema_failure_updates(response_text, exc)
            updates["current_plan_step"] = current_step or None
            updates["current_plan_step_index"] = step_index + 1 if current_step else None

    # 11. 如是 python_analysis_tool，把 analysis_contract 提出来
    if tool_calls:
        tc0 = tool_calls[0]
        if isinstance(tc0, dict) and tc0.get("name") == "python_analysis_tool":
            a0 = _coerce_tool_call_args(tc0)
            updates["python_analysis_contract"] = (a0.get("analysis_contract")
                or _infer_python_analysis_contract(state, messages, current_step)
            )
        for _tc in tool_calls:
            if isinstance(_tc, dict) and _tc.get("name") == "text2sql_tool":
                step_s = str(current_step or "")
                print(
                    f"  [agent] text2sql_tool 游标对齐 current_plan_step_index={updates.get('current_plan_step_index')} "
                    f"| 蓝图当前步 {len(step_s)} 字符（全文，与 text2sql_node 将读取的步一致）:\n{step_s}",
                    flush=True,
                )
                break

    return _attach_retry_if_needed(updates)


_PYTHON_ANALYSIS_TOOL_ARG_KEYS = frozenset({
    "question",
    "geojson_paths",
    "analysis_contract",
    "current_plan_step",
    "error_trace",
    "slots_json",
    "sql_queries",
})


def _finalize_python_analysis_tool_args(
    a: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """python_analysis_tool 参数白名单、列表规整；缺 question / geojson_paths 时按 state 回填。"""
    a = dict(a) if isinstance(a, dict) else {}
    df = a.pop("data_file", None)
    if df is not None:
        existing = a.get("geojson_paths")
        has_paths = isinstance(existing, list) and len(existing) > 0
        if not has_paths:
            if isinstance(df, list):
                a["geojson_paths"] = [str(x) for x in df if str(x).strip()]
            else:
                s = str(df).strip()
                a["geojson_paths"] = [s] if s else []
    for junk in ("action", "analysis_type", "params", "data_files", "path", "only_tools"):
        a.pop(junk, None)
    a = {k: v for k, v in a.items() if k in _PYTHON_ANALYSIS_TOOL_ARG_KEYS}
    gp = a.get("geojson_paths")
    if gp is None:
        a["geojson_paths"] = []
    elif not isinstance(gp, list):
        s = str(gp).strip()
        a["geojson_paths"] = [s] if s else []
    else:
        a["geojson_paths"] = [str(x) for x in gp if str(x).strip()]
    st = state if isinstance(state, dict) else None
    if st is not None:
        if not str(a.get("question") or "").strip():
            step_txt = str(_get_current_step_text(st) or "").strip()
            a["question"] = step_txt or str(_get_effective_question(st) or "").strip()
        if not a["geojson_paths"]:
            snap = st.get("latest_text2sql_geojson_paths")
            if isinstance(snap, list) and snap:
                a["geojson_paths"] = [str(x) for x in snap if str(x).strip()]
            else:
                legacy = st.get("geojson_paths")
                if isinstance(legacy, list) and legacy:
                    a["geojson_paths"] = [str(x) for x in legacy if str(x).strip()]
    return a


def _normalize_tool_args_for_executor(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    a = flatten_nested_tool_args(tool_args)
    if tool_name == "text2sql_tool":
        q = a.get("query")
        if (not str(a.get("question") or "").strip()) and isinstance(q, str) and q.strip():
            a = {**a, "question": q}
            a.pop("query", None)
        return a
    if tool_name == "python_analysis_tool":
        return _finalize_python_analysis_tool_args(a, state)
    return a


def tool_node(state: dict[str, Any]) -> dict[str, Any]:
    """工具执行节点：执行本轮 AIMessage 中的**第一条** tool call（与 parallel_tool_calls=false 对齐）。"""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {"messages": []}

    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if not tool_calls:
        parsed = _get_parsed_payload(last_msg)
        tool_calls = parsed.get("tool_calls", [])
    if not tool_calls:
        return {"messages": []}

    tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
    if not tool_calls:
        return {"messages": []}

    tool_call = tool_calls[0]
    call_id = tool_call.get("id")
    tool_name = str(tool_call.get("name") or "")

    tool_args = _normalize_tool_args_for_executor(
        tool_name, tool_call.get("args", {}) or {}, state=state
    )
    tool_func = TOOLS_MAP.get(tool_name)

    if not tool_func:
        error_text = f"未知工具：{tool_name}"
        primary = tool_observation(
            content=error_text,
            tool_call_id=call_id,
            tool_name=tool_name,
            success=False,
            payload={"error": error_text, "status": "error"},
            failure_code="unknown_tool",
        )
        return {
            "messages": build_tool_response_sequence(tool_calls, 0, primary),
            "errors": [error_text],
        }

    try:
        if tool_name == "schema_search_tool":
            if "question" not in tool_args:
                tool_args["question"] = _get_effective_question(state)
            if "slots_json" not in tool_args and state.get("slots"):
                tool_args["slots_json"] = json.dumps(state["slots"], ensure_ascii=False)
        elif tool_name == "map_rendering_tool":
            if "geojson_path" not in tool_args:
                geojson_files = [p for p in state.get("geojson_paths", []) if str(p).lower().endswith(".geojson")]
                if geojson_files:
                    tool_args["geojson_path"] = geojson_files[0]

        spec = spec_for_tool_name(tool_name)
        if spec and spec.execution_route == ExecutionRoute.INLINE:
            validated, verr = validate_tool_args_dict(tool_name, tool_args)
            if verr:
                primary = tool_observation(
                    content=verr,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    success=False,
                    payload={"error": verr, "status": "error"},
                    failure_code="tool_args_validation_failed",
                )
                return {
                    "messages": build_tool_response_sequence(tool_calls, 0, primary),
                    "errors": [verr],
                }
            if validated is not None:
                tool_args = validated

        print(f"  [工具调用] {tool_name}({list(tool_args.keys())})", flush=True)
        result = tool_func.invoke(tool_args)
        content, payload, success = _parse_tool_payload(result)
    except Exception as e:
        content = f"工具执行出错：{type(e).__name__}: {e}"
        payload = {"error": content, "status": "error"}
        success = False

    payload_out: dict[str, Any] | None = payload if isinstance(payload, dict) else None
    if not success:
        if payload_out is None:
            payload_out = {"error": content, "status": "error"}
        else:
            payload_out = {**payload_out, "status": "error"}

    fail_fc: str | None = None
    if not success:
        if isinstance(payload_out, dict) and payload_out.get("failure_code"):
            fail_fc = str(payload_out.get("failure_code"))
        else:
            fail_fc = "tool_execution_failed"

    tm = tool_observation(
        content=str(content),
        tool_call_id=call_id,
        tool_name=tool_name,
        success=success,
        payload=payload_out,
        failure_code=fail_fc,
    )

    parsed_updates = _parse_tool_result_to_state(tool_name, payload_out, state)
    err_delta = parsed_updates.pop("errors", None)
    batch_error_delta: list[str] = list(err_delta) if isinstance(err_delta, list) else []
    if not success and content and (not batch_error_delta):
        batch_error_delta.append(content[:2000])

    seq = build_tool_response_sequence(tool_calls, 0, tm)
    out: dict[str, Any] = {"messages": seq, **parsed_updates}
    if batch_error_delta:
        out["errors"] = batch_error_delta
    return out


def route_after_agent(state: dict[str, Any]) -> str:
    """有工具调用时执行；普通候选答案先进入独立证据审查。"""
    messages = state.get("messages", [])
    if len(messages) > MAX_MESSAGES_HARD_STOP:
        return "end"
    if not messages:
        return "end"
    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        return "end"
    if not isinstance(last_msg, AIMessage):
        return "end"
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if not tool_calls:
        parsed = _get_parsed_payload(last_msg)
        if parsed.get("type") == "tool_calls":
            tool_calls = parsed.get("tool_calls") or []
    if tool_calls:
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if isinstance(name, str) and graph_route_name(name) == "text2sql":
                return "text2sql"
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if isinstance(name, str) and graph_route_name(name) == "python_analysis":
                return "python_analysis"
        return "tools"
    return "review"


def _log_text2sql_console_block(
    *,
    raw_idx: Any,
    cur_idx: int,
    sql_task: str,
    rag_query: str,
    max_body_chars: int = 16000,
) -> None:
    """打印 Text2SQL 子任务与 RAG 查询全文，避免短 preview 被误认为传入 LLM 的内容被截断。"""
    n_task = len(sql_task)
    n_rag = len(rag_query)
    print(
        f"  [Text2SQL] 游标 plan_step_index_raw={raw_idx} plan_step_index_0based={cur_idx} | "
        f"sql_task={n_task} 字符 rag_query={n_rag} 字符（以下全文均原样传入 RAG / Text2SQL LLM）",
        flush=True,
    )

    def _emit(label: str, body: str) -> None:
        nb = len(body)
        if nb <= max_body_chars:
            print(f"  [Text2SQL] {label}（全文 {nb} 字符）:\n{body}", flush=True)
            return
        head = body[: max_body_chars // 2]
        tail = body[-(max_body_chars // 2) :]
        omitted = nb - len(head) - len(tail)
        print(
            f"  [Text2SQL] {label}（共 {nb} 字符，控制台仅展示首尾各 {len(head)}/{len(tail)} 字符，中间省略 {omitted} 字符）:",
            flush=True,
        )
        print(head, flush=True)
        print(f"  ...（省略 {omitted} 字符）...", flush=True)
        print(tail, flush=True)

    _emit("sql_task", sql_task)
    _emit("rag_query", rag_query)


def _try_reuse_text2sql_rag_cache(
    state: dict[str, Any],
    *,
    plan_step_index: int,
    rag_query: str,
    explicit_tables: list[str],
) -> tuple[list[str], str] | tuple[None, None]:
    """若 state.text2sql_schema_cache 与当前检索键一致，返回 (final_table_names, schemas_yaml)，否则 (None, None)。"""
    raw = state.get("text2sql_schema_cache")
    if not isinstance(raw, dict):
        return None, None
    cached_explicit = raw.get("explicit_tables")
    c_exp = list(cached_explicit) if isinstance(cached_explicit, list) else []
    if sorted(c_exp) != sorted(explicit_tables):
        return None, None
    try:
        cached_step = int(raw.get("plan_step_index", -99999))
    except (TypeError, ValueError):
        return None, None
    if cached_step != int(plan_step_index):
        return None, None
    if str(raw.get("rag_query") or "") != str(rag_query):
        return None, None
    ft = raw.get("final_table_names")
    sy = raw.get("schemas_yaml")
    if not isinstance(ft, list) or not isinstance(sy, str) or not str(sy).strip():
        return None, None
    return list(ft), sy


def text2sql_node(state: dict[str, Any]) -> dict[str, Any]:
    """独立 Text2SQL 节点，实现 LangGraph 透明执行。"""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    if not tool_calls:
        parsed = _get_parsed_payload(last_msg)
        tool_calls = parsed.get("tool_calls", [])
    tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
    primary_index = next((i for i, c in enumerate(tool_calls) if c.get("name") == "text2sql_tool"), None)
    if primary_index is None:
        return {}

    call = tool_calls[primary_index]
    args = _coerce_tool_call_args(call)
    call_id = call.get("id", "")
    args_q = args.get("question")
    if isinstance(args_q, str):
        args_q = args_q.strip() or None
    else:
        args_q = None

    if _current_blueprint_step_is_python_tool_step(state):
        if _text2sql_backtrack_allowed_after_tool_failures(messages):
            print(
                "  [Text2SQL] 护栏放行：上一批工具存在失败，允许跨步回溯 text2sql_tool 纠错。",
                flush=True,
            )
        else:
            err_txt = (
                "程序门禁：当前蓝图步为 python_analysis_tool，且上一批工具执行已成功（或无有效失败回包）。"
                "请勿无故回退 text2sql_tool；请继续完成 Python 分析，或直接输出自然语言结案。"
            )
            err_payload = {
                "queries": [],
                "geojson_paths": [],
                "table_names": [],
                "errors": [err_txt],
                "success": False,
            }
            err_msg = json.dumps(err_payload, ensure_ascii=False)
            _, payload, _ = _parse_tool_payload(err_msg)
            primary = tool_observation(
                content=err_msg,
                tool_call_id=call_id,
                tool_name="text2sql_tool",
                success=False,
                payload=payload,
                failure_code=FAILURE_CODE_TEXT2SQL_BLOCKED_ON_PYTHON_STEP,
            )
            updates = _parse_tool_result_to_state("text2sql_tool", payload, state)
            updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
            print("  [Text2SQL] 已拦截：Python 步且无上一批工具失败，拒绝改调 text2sql_tool。", flush=True)
            return updates

    sql_task, user_ctx = _resolve_text2sql_task(state, args_q)
    slots_early = state.get("slots") if isinstance(state.get("slots"), dict) else {}
    sql_task = _apply_text2sql_state_name_export_hint(
        sql_task.strip(), slots_early, _get_effective_question(state)
    )
    plan_meta = state.get("plan_meta") if isinstance(state.get("plan_meta"), list) else None
    analysis_export_for_python = bool(
        plan_meta
        and any(isinstance(item, dict) and item.get("tool") == "python_analysis" for item in plan_meta)
    )
    execution_contract = state.get("execution_contract")
    answer_projection = (
        execution_contract.get("answer_projection")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("answer_projection"), dict)
        else {}
    )
    semantic_bindings = (
        execution_contract.get("schema_bindings")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("schema_bindings"), list)
        else []
    )
    required_geometry_tables = (
        execution_contract.get("required_geometry_tables")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("required_geometry_tables"), list)
        else []
    )
    if semantic_bindings:
        sql_task += (
            "\n\n【Schema 语义绑定】以下映射由 Intent Understanding 与 Schema Pre-filtering "
            "共同确认。SQL 必须保留这些表字段的数据血缘，禁止用其他数值相近字段替代：\n"
            + json.dumps(semantic_bindings, ensure_ascii=False, separators=(",", ":"))
        )
    if required_geometry_tables:
        sql_task += (
            "\n【空间输入契约】该任务需要跨图层空间计算。Text2SQL 只负责分别导出以下图层的"
            "原始 geometry 与关联键，不得在 SQL 中执行空间拓扑函数："
            + "、".join(str(item) for item in required_geometry_tables)
        )
    condition_clauses = (
        execution_contract.get("condition_clauses")
        if isinstance(execution_contract, dict)
        and isinstance(execution_contract.get("condition_clauses"), list)
        else []
    )
    if condition_clauses:
        sql_task += (
            "\n【约束清单】以下条件来自 Intent Understanding 的原句切分。"
            "SQL 必须逐项落实，不得只实现其中一部分：\n- "
            + "\n- ".join(str(item) for item in condition_clauses if str(item).strip())
        )
        sql_task += (
            "\n【枚举值边界】分类字段使用 Schema 给出的枚举值。用户只提出一个类别时，"
            "默认选择语义最直接对应的单一枚举值；除非用户明确要求复合、混合或多个类别，"
            "不得自行扩展到名称相近的复合类别。"
        )
    if analysis_export_for_python:
        sql_task += (
            "\n【中间数据完整性】本步只为后续 STCA 准备输入。最终答案中的排除、阈值或排名条件，"
            "不得提前删除后续计算所需的参考实体、基准组、对照组或全局统计样本；"
            "应保留完整计算操作数，并在 Python 分析完成后再应用最终输出筛选。"
        )
    if isinstance(execution_contract, dict) and not bool(execution_contract.get("requires_geometry")):
        sql_task += (
            "\n【几何最小化】本任务不需要空间几何计算；除非最终问题明确要求几何，"
            "SELECT 不得携带 geometry，导出 JSON 而非 GeoJSON。"
        )
    if not analysis_export_for_python:
        requested_stats = ", ".join(answer_projection.get("statistics") or []) or "问题要求的筛选或聚合"
        sql_task += (
            "\n\n【单步 SQL 闭环契约】本蓝图没有 Python 后处理。SQL 必须直接完成最终的"
            f"{requested_stats}并导出最终答案记录；不得只导出原始明细留给后续计算。"
        )
    ctx_part = (user_ctx or "").strip()
    task_part = sql_task.strip()
    if _line_looks_like_raw_sql(task_part) or (args_q and _line_looks_like_raw_sql(args_q)):
        err_txt = (
            "协议违例：text2sql_tool 的 question 不得直接粘贴 SELECT/WITH SQL。"
            "请仅使用与当前蓝图步一致的自然语言子任务描述，由系统生成 SQL。"
        )
        err_payload = {
            "queries": [],
            "geojson_paths": [],
            "table_names": [],
            "errors": [err_txt],
            "success": False,
        }
        err_msg = json.dumps(err_payload, ensure_ascii=False)
        _, payload, _ = _parse_tool_payload(err_msg)
        primary = tool_observation(
            content=err_msg,
            tool_call_id=call_id,
            tool_name="text2sql_tool",
            success=False,
            payload=payload,
            failure_code=FAILURE_CODE_TEXT2SQL_RAW_SQL_IN_QUESTION,
        )
        updates = _parse_tool_result_to_state("text2sql_tool", payload, state)
        updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
        print("  [Text2SQL] 已拦截：检测到手写 SQL 作为 question。", flush=True)
        return updates

    raw_explicit = args.get("table_names")
    explicit_tables: list[str] = list(raw_explicit) if isinstance(raw_explicit, list) else []

    slots_for_rag = state.get("slots")
    if not isinstance(slots_for_rag, dict):
        slots_for_rag = {}

    cur_idx_for_cache = _get_current_step_index(state)
    text2sql_rag_cache_hit = False

    if explicit_tables:
        schemas_yaml = format_schema_yaml_by_exact_table_names(explicit_tables)
        final_table_names = list(explicit_tables)
        rag_query = "(deterministic, skipped RAG)"
        print(
            f"  [Text2SQL] 命中工具参数指定表名 {final_table_names}，跳过向量检索（全量 YAML）。",
            flush=True,
        )
    else:
        anchor = ctx_part or str(_get_effective_question(state) or "").strip()
        rag_query = build_rag_query("text2sql", global_question=anchor, sql_task=task_part).strip()
        if not rag_query:
            if ctx_part and task_part and ctx_part == task_part:
                rag_query = task_part
            else:
                rag_query = f"{ctx_part} {task_part}".strip()

        reused_tables, reused_yaml = _try_reuse_text2sql_rag_cache(
            state,
            plan_step_index=cur_idx_for_cache,
            rag_query=rag_query,
            explicit_tables=explicit_tables,
        )
        if reused_tables is not None and reused_yaml is not None:
            final_table_names = reused_tables
            schemas_yaml = reused_yaml
            text2sql_rag_cache_hit = True
            print(
                "  [Text2SQL] 复用 text2sql_schema_cache，跳过向量检索；"
                f"表名: {final_table_names}",
                flush=True,
            )
        else:
            pre_rag_schemas = state.get("schemas") if isinstance(state.get("schemas"), list) else []
            pre_rag_tables = (
                state.get("retrieved_table_names")
                if isinstance(state.get("retrieved_table_names"), list)
                else []
            )
            if pre_rag_schemas and pre_rag_tables:
                schema_bundle = {
                    "schemas": list(pre_rag_schemas),
                    "table_names": list(pre_rag_tables),
                }
                dynamic_tables = list(pre_rag_tables)
                print(
                    "  [Text2SQL] 复用 pre_rag Schema bundle，跳过二次向量检索。",
                    flush=True,
                )
            else:
                schema_bundle = retrieve_top_k_schema_bundle(
                    slots_for_rag,
                    natural_language_query=rag_query,
                    semantic_anchor_query=str(_get_effective_question(state) or "").strip(),
                ) or {}
                dynamic_tables = schema_bundle.get("table_names", []) or []
            known_tables = _collect_known_table_names(state, messages)
            final_table_names = _merge_unique_table_names(explicit_tables, dynamic_tables, known_tables)

            schemas_yaml = build_text2sql_schemas_yaml_from_bundle(schema_bundle, final_table_names)
            print(
                f"  [Text2SQL] 未提供表名；合并表名: {final_table_names}；"
                f"Schema 正文优先使用列裁剪 YAML（bundle 未覆盖的表回退全量 YAML）",
                flush=True,
            )

    # 跨图层空间契约中的几何源是硬依赖。即使 Agent 未在工具参数中重复列出，
    # 也必须补入最终表集合及 Schema 上下文；同时保留预检索到的属性表。
    missing_contract_tables = [
        str(table)
        for table in required_geometry_tables
        if str(table).strip() and str(table) not in final_table_names
    ]
    if missing_contract_tables:
        final_table_names = _merge_unique_table_names(final_table_names, missing_contract_tables)
        extra_schemas = format_schema_yaml_by_exact_table_names(missing_contract_tables).strip()
        if extra_schemas and extra_schemas != "（无 Schema）":
            schemas_yaml = f"{schemas_yaml.rstrip()}\n\n{extra_schemas}".strip()
        print(
            f"  [Text2SQL] 空间输入契约补入几何表: {missing_contract_tables}",
            flush=True,
        )

    error_feedback = args.get("error_feedback")
    if not error_feedback:
        err = _extract_recent_error(messages, "text2sql_tool")
        failed_count = _count_failed_tool_calls(messages, "text2sql_tool")
        if err and not _has_successful_tool_call(messages, "text2sql_tool"):
            if failed_count > 1:
                print(f"  [Text2SQL] 已连续失败 {failed_count} 次，不再静默重试，将错误返回给 Agent 层。", flush=True)
            else:
                retry_ctx = _build_retry_context_for_text2sql(messages)
                error_feedback = f"{err}\n\n{retry_ctx}" if retry_ctx else err
                print(f"  [Text2SQL] 自动注入上次错误反馈（第 {failed_count+1} 次尝试）", flush=True)

    cur_idx = _get_current_step_index(state)
    raw_idx = state.get("current_plan_step_index")
    print(f"  [Node] text2sql_node(table_names={final_table_names})", flush=True)
    _log_text2sql_console_block(
        raw_idx=raw_idx,
        cur_idx=cur_idx,
        sql_task=task_part,
        rag_query=rag_query,
    )
    result_str = execute_text2sql_logic(
        sql_task=sql_task,
        table_names=final_table_names,
        user_context=user_ctx,
        error_feedback=error_feedback,
        schemas_yaml=schemas_yaml,
        analysis_export_for_python=analysis_export_for_python,
        ordered_result_required=answer_projection.get("output_shape") == "ranked_records",
        requested_top_k=answer_projection.get("top_k"),
        semantic_bindings=semantic_bindings,
        condition_clauses=condition_clauses,
        required_geometry_tables=required_geometry_tables,
    )

    content, payload, success = _parse_tool_payload(result_str)
    if not success:
        _raise_if_tool_payload_infra_failure("text2sql_tool", payload, content)
    followup_hint = None
    if success:
        if analysis_export_for_python:
            followup_hint = "数据文件已导出，如蓝图仍包含 Python 分析步骤，请继续选择 python_analysis_tool。"
    fail_fc: str | None = None
    if not success:
        if isinstance(payload, dict) and payload.get("failure_code"):
            fail_fc = str(payload.get("failure_code"))
        else:
            fail_fc = "text2sql_execution_failed"
    primary = tool_observation(
        content=str(content),
        tool_call_id=call_id,
        tool_name="text2sql_tool",
        success=success,
        payload=payload,
        failure_code=fail_fc,
        followup_hint=followup_hint,
    )

    updates = _parse_tool_result_to_state("text2sql_tool", payload, state)
    updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
    if explicit_tables:
        updates["text2sql_schema_cache"] = None
    elif not text2sql_rag_cache_hit:
        updates["text2sql_schema_cache"] = {
            "plan_step_index": cur_idx_for_cache,
            "rag_query": rag_query,
            "explicit_tables": list(explicit_tables),
            "final_table_names": list(final_table_names),
            "schemas_yaml": schemas_yaml,
        }
    return updates


def python_analysis_node(state: dict[str, Any]) -> dict[str, Any]:
    """独立 Python Analysis 节点，实现 LangGraph 透明执行。"""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    if not tool_calls:
        parsed = _get_parsed_payload(last_msg)
        tool_calls = parsed.get("tool_calls", [])
    tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
    primary_index = next((i for i, c in enumerate(tool_calls) if c.get("name") == "python_analysis_tool"), None)
    if primary_index is None:
        return {}

    call = tool_calls[primary_index]
    args = _coerce_tool_call_args(call)
    args = _finalize_python_analysis_tool_args(args, state)
    call_id = call.get("id", "")

    question = args.get("question") or _get_effective_question(state)
    error_trace = args.get("error_trace")
    current_plan_step = args.get("current_plan_step") or state.get("current_plan_step") or ""
    analysis_contract = args.get("analysis_contract") or state.get("python_analysis_contract") or _infer_python_analysis_contract(state, messages, current_plan_step)
    analysis_contract = _normalize_python_analysis_contract(analysis_contract)
    if not error_trace:
        err = _extract_recent_error(messages, "python_analysis_tool")
        if err and not _has_successful_tool_call(messages, "python_analysis_tool"):
            error_trace = err

    slots_json = args.get("slots_json")
    if not slots_json and state.get("slots"):
        slots_json = json.dumps(state["slots"], ensure_ascii=False)

    # ── GeoJSON 路径三级优先级解析 ──────────────────────────────────────────
    # 优先级 1：Executor 显式传参（最可信，来自当前轮次的蓝图步骤）
    raw_geojson_arg = args.get("geojson_paths")
    used_explicit_geojson_paths = isinstance(raw_geojson_arg, list) and len(raw_geojson_arg) > 0
    geojson_paths: list[str] = list(raw_geojson_arg) if used_explicit_geojson_paths else []
    # 神级容错：若模型幻觉了自创参数名，强制从所有参数值中捞取 .geojson/.json 路径
    if not geojson_paths:
        for _, v in args.items():
            if isinstance(v, str) and (v.lower().endswith(".geojson") or v.lower().endswith(".json")):
                geojson_paths.append(v)
            elif isinstance(v, dict):
                for sub_v in v.values():
                    if isinstance(sub_v, str) and (sub_v.lower().endswith(".geojson") or sub_v.lower().endswith(".json")):
                        geojson_paths.append(sub_v)

    if not geojson_paths:
        # 优先级 2：本次会话最近一次成功 Text2SQL 的快照（新鲜、可追溯，避免旧批次污染）
        snapshot_paths = state.get("latest_text2sql_geojson_paths") or []
        if snapshot_paths:
            geojson_paths = list(snapshot_paths)
            print("  [python_analysis] 使用 latest_text2sql_geojson_paths 快照路径", flush=True)

    if not geojson_paths:
        # 优先级 3：state.geojson_paths 兼容回退（可能含多轮累积，仅作最后保底）
        geojson_paths = list(state.get("geojson_paths") or [])
        if geojson_paths:
            print("  [python_analysis] 回退使用 state.geojson_paths（可能含多轮累积）", flush=True)

    # 快照 / 回退得到的路径与显式参数同等校验，避免工作区已删除文件仍进沙盒
    if geojson_paths and not used_explicit_geojson_paths:
        path_err_snap = validate_explicit_geojson_paths_in_workspace(geojson_paths)
        if path_err_snap:
            error_msg = json.dumps(
                {
                    "success": False,
                    "error": path_err_snap,
                    "code_output": "",
                    "failure_code": FAILURE_CODE_PYTHON_MISSING_EXPLICIT_FILE,
                },
                ensure_ascii=False,
            )
            _, payload, _ = _parse_tool_payload(error_msg)
            primary = tool_observation(
                content=error_msg,
                tool_call_id=call_id,
                tool_name="python_analysis_tool",
                success=False,
                payload=payload,
                failure_code=FAILURE_CODE_PYTHON_MISSING_EXPLICIT_FILE,
            )
            updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
            updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
            print(f"  [python_analysis] 快照/回退路径在工作区不存在，提前返回：{path_err_snap[:120]}...", flush=True)
            return updates

    if not geojson_paths:
        # 无任何路径可用：提前返回明确错误，避免以空文件列表静默执行
        error_msg = json.dumps({
            "success": False,
            "error": (
                "python_analysis_tool 无法获取有效的 GeoJSON 文件路径。"
                "请先确保 text2sql_tool 已成功执行并导出数据文件，"
                "或在调用 python_analysis_tool 时通过 geojson_paths 参数显式传入文件路径。"
            ),
            "code_output": "",
        }, ensure_ascii=False)
        _, payload, _ = _parse_tool_payload(error_msg)
        primary = tool_observation(
            content=error_msg,
            tool_call_id=call_id,
            tool_name="python_analysis_tool",
            success=False,
            payload=payload,
            failure_code="python_missing_input_paths",
        )
        updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
        updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
        print("  [python_analysis] 无可用 GeoJSON 路径，提前返回错误。", flush=True)
        return updates

    if used_explicit_geojson_paths:
        path_err = validate_explicit_geojson_paths_in_workspace(geojson_paths)
        if path_err:
            error_msg = json.dumps(
                {
                    "success": False,
                    "error": path_err,
                    "code_output": "",
                    "failure_code": FAILURE_CODE_PYTHON_MISSING_EXPLICIT_FILE,
                },
                ensure_ascii=False,
            )
            _, payload, _ = _parse_tool_payload(error_msg)
            primary = tool_observation(
                content=error_msg,
                tool_call_id=call_id,
                tool_name="python_analysis_tool",
                success=False,
                payload=payload,
                failure_code=FAILURE_CODE_PYTHON_MISSING_EXPLICIT_FILE,
            )
            updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
            updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
            print(f"  [python_analysis] 显式 geojson_paths 在工作区不存在，提前返回：{path_err[:120]}...", flush=True)
            return updates

    sql_queries = args.get("sql_queries")
    if not sql_queries:
        sql_queries = state.get("sql_queries") or []

    slots_for_val = state.get("slots") if isinstance(state.get("slots"), dict) else {}

    if _has_pending_text2sql_before_python(
        state,
        messages,
        geojson_paths=geojson_paths,
        current_plan_step=str(current_plan_step or ""),
        question=str(question or ""),
        sql_queries=list(sql_queries or []),
        slots=slots_for_val,
    ):
        err_txt = (
            "程序门禁：当前 Python 所需数据尚未齐备，且蓝图仍有 text2sql 步可补齐；"
            "请先执行 text2sql_tool 导出后再调用 python_analysis_tool。"
        )
        error_msg = json.dumps(
            {
                "success": False,
                "error": err_txt,
                "code_output": "",
                "failure_code": FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS,
            },
            ensure_ascii=False,
        )
        _, payload, _ = _parse_tool_payload(error_msg)
        primary = tool_observation(
            content=error_msg,
            tool_call_id=call_id,
            tool_name="python_analysis_tool",
            success=False,
            payload=payload,
            failure_code=FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS,
        )
        updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
        updates.update(_plan_rewind_updates_for_sql_inputs(state))
        updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
        print("  [python_analysis] 已拦截：存在未完成的 text2sql 步骤。", flush=True)
        return updates

    incomplet = _validate_python_input_completeness(
        current_plan_step=str(current_plan_step or ""),
        question=str(question or ""),
        slots=slots_for_val,
        geojson_paths=geojson_paths,
        sql_queries=list(sql_queries or []),
    )
    if incomplet:
        error_msg = json.dumps(
            {
                "success": False,
                "error": incomplet,
                "code_output": "",
                "failure_code": FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS,
            },
            ensure_ascii=False,
        )
        _, payload, _ = _parse_tool_payload(error_msg)
        primary = tool_observation(
            content=error_msg,
            tool_call_id=call_id,
            tool_name="python_analysis_tool",
            success=False,
            payload=payload,
            failure_code=FAILURE_CODE_PYTHON_REQUIRES_MORE_SQL_INPUTS,
        )
        updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
        updates.update(_plan_rewind_updates_for_sql_inputs(state))
        updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
        print(f"  [python_analysis] 输入完备性校验未通过：{incomplet[:120]}...", flush=True)
        return updates

    _path_names_log = [Path(str(raw_path)).name for raw_path in geojson_paths]
    print(f"  [Node] python_analysis_node(geojson_paths={_path_names_log})", flush=True)
    ec_raw = state.get("execution_contract")
    execution_contract = ec_raw if isinstance(ec_raw, dict) else None
    result_str = execute_python_analysis_logic(
        question=question,
        error_trace=error_trace,
        slots_json=slots_json,
        current_plan_step=current_plan_step,
        analysis_contract=analysis_contract,
        geojson_paths=geojson_paths,
        sql_queries=sql_queries,
        execution_contract=execution_contract,
    )

    content, payload, success = _parse_tool_payload(result_str)
    if not success:
        _raise_if_tool_payload_infra_failure("python_analysis_tool", payload, content)
    py_fc: str | None = None
    if isinstance(payload, dict) and payload.get("failure_code"):
        py_fc = str(payload.get("failure_code"))
    elif not success:
        py_fc = "python_analysis_execution_failed"
    primary = tool_observation(
        content=str(content),
        tool_call_id=call_id,
        tool_name="python_analysis_tool",
        success=success,
        payload=payload if isinstance(payload, dict) else None,
        failure_code=py_fc,
    )

    updates = _parse_tool_result_to_state("python_analysis_tool", payload, state)
    updates["messages"] = build_tool_response_sequence(tool_calls, primary_index, primary)
    return updates
