# -*- coding: utf-8 -*-
"""LangGraph Agent 状态定义：IntentionSlots (Pydantic) + AgentState (TypedDict) + ReAct 消息历史。"""
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator


def merge_path_lists(left: Optional[list[str]], right: Optional[list[str]]) -> list[str]:
    """合并路径列表：保序去重；显式传入空列表 [] 时清空（供 REPL 每轮重置，避免旧路径串车）。"""
    if right is None:
        return list(left or [])
    if not right:
        return []
    left = list(left or [])
    seen = set(left)
    out = list(left)
    for x in right:
        xs = str(x)
        if xs not in seen:
            seen.add(xs)
            out.append(xs)
    return out


def reset_aware_extend(left: Any, right: Any) -> list:
    """列表字段：图内多步追加；REPL 显式 [] 时清空 checkpoint 累积（operator.add 无法用 [] 清空）。"""
    if right is None:
        return list(left or [])
    if isinstance(right, list) and len(right) == 0:
        return []
    left = list(left or [])
    if isinstance(right, list):
        return left + right
    return left + [right]


def _coerce_to_str(v: Any) -> Optional[str]:
    """LLM 可能返回 int/float，统一转为 str 供校验通过。"""
    if v is None:
        return None
    return str(v)


def _coerce_list_to_str_list(v: Any) -> Optional[list[str]]:
    """列表元素可能为 int/float，统一转为 list[str]。"""
    if v is None:
        return None
    if not isinstance(v, list):
        return None
    return [str(x) for x in v]


class IntentionSlots(BaseModel):
    """意图解析槽位：业务与时空、代码路由相关字段。"""

    # ── 中枢判定字段（必须强制输出，去掉了所有的 default，不可省略！）─────────────────────────
    is_queryable: bool = Field(
        ...,
        description="该问题是否为可被数据库查询的有效 GIS 分析问题。闲聊/问候/无明确空间意图时填 false。"
    )
    reject_reason: str = Field(
        ...,
        description="如果 is_queryable 为 false，请给出拒绝理由。如果是有效查询，必须输出空字符串。"
    )
    # ── 业务槽位字段 ──────────────────────────────────────────────────────
    region: Optional[str] = Field(default=None, description="区域名称，如杭州市、浙江省")
    region_scope: Optional[str] = Field(default=None, description="区域所属范围，如美国、浙江、中国")
    region_set: Optional[list[str]] = Field(default=None, description="区域集合")
    time: Optional[str] = Field(default=None, description="时间点，如 2022")
    time_range: Optional[list[str]] = Field(default=None, description="时间范围，如 [2012, 2022]")
    metric: Optional[str] = Field(default=None, description="指标名称，如人口、NDVI")
    metric_set: Optional[list[str]] = Field(default=None, description="指标集合")
    calc: Optional[str] = Field(
        default=None,
        description="派生计算类型：变化率、排名、平均值等",
    )

    spatial_predicate: Optional[str] = Field(
        default=None,
        description="空间谓词：contains, intersects, within_distance, touches, buffer, overlay",
    )
    spatial_threshold: Optional[str] = Field(
        default=None,
        description="空间阈值，如 10km、500km",
    )
    target_entity_A: Optional[str] = Field(
        default=None,
        description="跨图层叠置时的实体 A，如杭州市边界",
    )
    target_entity_B: Optional[str] = Field(
        default=None,
        description="跨图层叠置时的实体 B，如渔网网格",
    )
    analytical_method: Optional[str] = Field(
        default=None,
        description="高级分析算子：correlation, clustering, regression, zonal_statistics, minimum_bounding",
    )
    condition: dict[str, Any] = Field(
        default_factory=dict,
        description="筛选与非空间约束条件（键值对象）；无额外约束时必须输出空对象 {}。",
    )
    visualization: str = Field(
        default="",
        description="可视化或展示意图；无时必须输出空字符串。",
    )

    @field_validator("condition", mode="before")
    @classmethod
    def coerce_condition_dict(cls, v: Any) -> dict[str, Any]:
        if v is None or v == "":
            return {}
        if isinstance(v, dict):
            return dict(v)
        return {}

    @field_validator("time", "region", "region_scope", "calc", "metric", "spatial_predicate", "spatial_threshold", "target_entity_A", "target_entity_B", "analytical_method", "reject_reason", "visualization", mode="before")
    @classmethod
    def coerce_str_fields(cls, v: Any) -> Optional[str]:
        # 保持运行期鲁棒性：即使上游给了 None，也转成空字符串，避免必填字段触发崩溃。
        if v is None:
            return ""
        return _coerce_to_str(v)

    @field_validator("region_set", "time_range", "metric_set", mode="before")
    @classmethod
    def coerce_list_str_fields(cls, v: Any) -> Optional[list[str]]:
        return _coerce_list_to_str_list(v)

    @model_validator(mode="after")
    def validate_core_fields(self):
        """中枢字段一致性约束。"""
        if not self.is_queryable and not str(self.reject_reason or "").strip():
            raise ValueError("当 is_queryable=false 时，reject_reason 不能为空")
        return self


class PlanStep(BaseModel):
    """唯一受支持的结构化计划步骤协议。"""

    tool: Literal["text2sql_tool", "python_analysis_tool"]
    objective: str = Field(min_length=1, max_length=600)
    success_criteria: list[str] = Field(min_length=1, max_length=3)

    @field_validator("objective", mode="before")
    @classmethod
    def normalize_objective(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("success_criteria", mode="before")
    @classmethod
    def normalize_criteria(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("success_criteria 必须是字符串数组")
        return [str(item).strip() for item in value if str(item).strip()][:3]


class PlanBlueprint(BaseModel):
    """V2 计划协议：一个 SQL 步骤，按需追加一个 Python 步骤。"""

    steps: list[PlanStep] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_execution_order(self):
        tools = [step.tool for step in self.steps]
        if tools[0] != "text2sql_tool":
            raise ValueError("计划第一步必须是 text2sql_tool")
        if len(set(tools)) != len(tools):
            raise ValueError("同一种工具在计划中只能出现一次")
        if "python_analysis_tool" in tools and tools[-1] != "python_analysis_tool":
            raise ValueError("python_analysis_tool 必须是最后一步")
        return self


class AgentState(TypedDict, total=False):
    """LangGraph 全局状态。支持 ReAct 循环的消息历史 + 工具产出的中间产物。"""

    # ReAct 消息历史：Master Agent 与工具之间的对话
    messages: Annotated[list[AnyMessage], add_messages]

    # 用户原始问题
    question: str

    # 工具产出的中间产物（供跨工具共享数据）
    slots: Optional[dict[str, Any]]  # 从用户问题中解析出来的语义槽位 (意图、区域、指标等)
    schemas: list[str]  # 检索到的相关数据库表 DDL 语句列表
    schemas_yaml: str  # DDL 信息的 YAML 序列化字符串，供 LLM 读取
    sql_queries: Annotated[list[dict], reset_aware_extend]  # 各轮 Text2SQL 查询记录；[] 清空
    sql_results: Annotated[list[dict], reset_aware_extend]  # 各轮 SQL 摘要；[] 清空
    geojson_paths: Annotated[list[str], merge_path_lists]  # 本次任务中生成的全部导出文件路径（追加去重）
    # STCA 生成的 CSV/JSON 等可复用中间产物。与 SQL 输入分开记录，供
    # Execution State Manager 在局部修复时传回 STCA，避免丢失或覆盖证据。
    analysis_artifact_paths: Annotated[list[str], merge_path_lists]
    code: Optional[str]  # 生成的 Python 分析代码
    code_output: Optional[str]  # Python 代码执行的控制台输出
    errors: Annotated[list[str], reset_aware_extend]  # 运行过程中累积的错误；[] 清空
    retry_count: int  # Executor 入口累计：上一轮工具失败（或 guardrail_retry HumanMessage）再打回模型的次数，用于熔断
    final_answer: Optional[str]  # 成功时为 StrictFinalAnswer 纯 JSON；失败时可为错误文本
    final_answer_payload: Optional[dict[str, Any]]  # 完整结构化答案对象（不抽样）
    final_answer_schema_valid: bool  # 是否通过 StrictFinalAnswer Pydantic 校验
    final_answer_schema_error: Optional[str]  # 结构化结案失败原因；成功时为 None
    # 蓝图最后一步 Python 已成功执行，允许后续无 tool_calls 的自然语言结案（供 Guardrail 豁免）
    final_answer_ready: bool
    map_path: Optional[str]  # 可视化生成的 HTML 地图文件路径
    retrieved_table_names: list[str]  # 检索出的核心表名列表
    # Intent Understanding / Schema Pre-filtering 共同确认的概念→表字段绑定。
    schema_bindings: list[dict[str, Any]]
    schema_coverage: dict[str, Any]
    plan: list[str]  # 由 PlanBlueprint 确定性渲染的步骤文本，仅用于显示和工具任务描述
    # 唯一权威的结构化步骤状态（tool/objective/success_criteria/flags）
    plan_meta: list[dict[str, Any]]
    # 规划完成后的统一执行契约（requires_python / requires_geometry 等）；与 plan_meta 同步写入
    execution_contract: Optional[dict[str, Any]]
    current_plan_step: Optional[str]  # 当前正在执行的任务步骤描述
    current_plan_step_index: Optional[int]  # 当前步骤在 plan 中的索引
    python_analysis_contract: Optional[dict[str, Any]]  # Python 分析任务的约定协议，包含输入输出参数定义
    step_failure_counts: dict[str, int]  # 记录每个步骤的失败次数，用于容错策略
    last_failure_type: Optional[str]  # 最近一次失败的错误类型
    last_failure_step_index: Optional[int]  # 最近一次失败发生的步骤索引
    direct_response: Optional[str]  # LLM 判定的直接回复内容（如闲聊、无法回答时的补救方案）
    # 最近一次成功 text2sql_tool 调用导出的 GeoJSON 路径快照；
    # 成功时由 text2sql_node 写入，失败时清空，供 python_analysis_node 优先读取以避免旧批次污染。
    latest_text2sql_geojson_paths: list[str]
    # text2sql 节点 Schema 结果缓存：同一步、同 rag_query 重试时跳过检索
    text2sql_schema_cache: Optional[dict[str, Any]]
    # v2 通用证据审查与有界重规划状态；审查器不接触 benchmark Gold。
    review_action: Optional[str]
    review_feedback: Optional[str]
    review_count: int
    replan_count: int
def create_initial_state(question: str) -> dict[str, Any]:
    """创建图入口的初始状态。"""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "slots": None,
        "schemas": [],
        "schemas_yaml": "",
        "sql_queries": [],
        "sql_results": [],
        "geojson_paths": [],
        "analysis_artifact_paths": [],
        "code": None,
        "code_output": None,
        "errors": [],
        "retry_count": 0,
        "final_answer": None,
        "final_answer_payload": None,
        "final_answer_schema_valid": False,
        "final_answer_schema_error": None,
        "final_answer_ready": False,
        "map_path": None,
        "retrieved_table_names": [],
        "schema_bindings": [],
        "schema_coverage": {},
        "plan": [],
        "plan_meta": [],
        "execution_contract": None,
        "current_plan_step": None,
        "current_plan_step_index": None,
        "python_analysis_contract": None,
        "step_failure_counts": {},
        "last_failure_type": None,
        "last_failure_step_index": None,
        "direct_response": None,
        "latest_text2sql_geojson_paths": [],
        "text2sql_schema_cache": None,
        "review_action": None,
        "review_feedback": None,
        "review_count": 0,
        "replan_count": 0,
    }
