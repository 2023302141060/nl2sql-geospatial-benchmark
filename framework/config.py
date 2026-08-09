# -*- coding: utf-8 -*-
"""集中配置：LLM、数据库、路径常量。从 .env 读取环境变量。"""
import os
import ssl
import warnings
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# LLM 配置（OpenAI 兼容客户端：ChatOpenAI 读取下列变量）
# 设置 LLM_PROVIDER=qwen/deepseek 时，优先读取对应的
# QWEN_LLM_* / DEEPSEEK_LLM_* Profile；未设置时兼容旧版 LLM_* 配置。
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()
_LLM_PROVIDERS = {"qwen", "deepseek"}
if LLM_PROVIDER and LLM_PROVIDER not in _LLM_PROVIDERS:
    raise ValueError(
        f"不支持的 LLM_PROVIDER={LLM_PROVIDER!r}，可选值为："
        f"{', '.join(sorted(_LLM_PROVIDERS))}"
    )

if LLM_PROVIDER:
    _llm_prefix = LLM_PROVIDER.upper()
    _profile_names = {
        "base_url": f"{_llm_prefix}_LLM_BASE_URL",
        "api_key": f"{_llm_prefix}_LLM_API_KEY",
        "model": f"{_llm_prefix}_LLM_MODEL",
    }
    _profile_values = {
        name: (os.getenv(env_name) or "").strip()
        for name, env_name in _profile_names.items()
    }
    _missing_profile_names = [
        _profile_names[name] for name, value in _profile_values.items() if not value
    ]
    if _missing_profile_names:
        raise ValueError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} 的 Profile 配置不完整，缺少："
            f"{', '.join(_missing_profile_names)}"
        )
    LLM_BASE_URL = _profile_values["base_url"]
    LLM_API_KEY = _profile_values["api_key"]
    LLM_MODEL = _profile_values["model"]
else:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)


def _default_embedding_model() -> str:
    """向量模型名：OpenAI 默认 ada-002；百炼无该模型，默认 text-embedding-v3。"""
    if "dashscope" in (EMBEDDING_BASE_URL or "").lower():
        return "text-embedding-v3"
    return "text-embedding-ada-002"


# Schema RAG；可用环境变量 EMBEDDING_MODEL 覆盖。切换向量模型后应删除 workspace/chroma_db 以重建索引。
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "").strip() or _default_embedding_model()


def _dashscope_tls12_http_client(timeout_seconds: float):
    """创建固定 TLS 1.2、无自动重试的百炼 HTTP 客户端。"""
    import httpx

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return httpx.Client(
        http2=False,
        verify=context,
        timeout=timeout_seconds,
        trust_env=False,
    )


def get_schema_embeddings():
    """构建 Schema RAG 用的 Embeddings。

    Embedding 渠道可通过 EMBEDDING_* 环境变量单独配置，默认回退到 LLM_*。
    百炼兼容端点使用固定 TLS 1.2 的 OpenAI 客户端，规避当前 Windows/NLP
    环境中 TLS 1.3 握手偶发 EOF；请求语义和重试策略不变。
    """
    if "dashscope" in (EMBEDDING_BASE_URL or "").lower():
        from langchain_openai import OpenAIEmbeddings

        timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            # Some DashScope workspace keys enforce a maximum embedding batch
            # size of 10 even though the generic OpenAI client defaults much
            # higher.  Keep this explicit so Chroma indexing is portable.
            chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            check_embedding_ctx_length=False,
            request_timeout=timeout,
            max_retries=0,
            http_client=_dashscope_tls12_http_client(timeout),
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
    )


# 数据库配置
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "NL2SQL")

_db_user_quoted = quote_plus(DB_USER)
_db_password_quoted = quote_plus(DB_PASSWORD)
_db_host = DB_HOST.strip()
_db_port = DB_PORT.strip()
_db_name_quoted = quote_plus(DB_NAME)

DATABASE_URL = (
    f"postgresql+psycopg2://{_db_user_quoted}:{_db_password_quoted}@{_db_host}:{_db_port}/{_db_name_quoted}"
)

# 目录路径
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
CODE_TEMPLATES_DIR = PROJECT_ROOT / "code_templates"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# Agent 无 tool_calls（自然语言结案）时打印到 stdout 的最大字符数；≤0 表示不截断（长回答完整进入 tee 日志）
AGENT_NO_TOOLCALLS_STDOUT_MAX_CHARS = int(os.getenv("AGENT_NO_TOOLCALLS_STDOUT_MAX_CHARS", "1000"))

# run_agent_autotest：tc_*.yaml 中 results.final_answer 用 main._format_code_output_log_preview 截断（默认 16000 字符）
AUTOTEST_YAML_FINAL_ANSWER_MAX_CHARS = int(os.getenv("AUTOTEST_YAML_FINAL_ANSWER_MAX_CHARS", "16000"))

# 代码沙盒超时（秒）
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "60"))
# 沙盒启动前：工作区内 .geojson/.json/.csv 单文件超过此大小（MB）则拒绝执行，避免 Windows 宿主加载大文件 OOM（默认 1024≈1GiB）
SANDBOX_OOM_MAX_FILE_MB = int(os.getenv("SANDBOX_OOM_MAX_FILE_MB", "1024"))

# ReAct 循环最大递归步数（防止 Agent 死循环）
# 支持通过环境变量覆盖，便于复杂任务临时放宽图步数上限。
# 图递归步数上限；复杂蓝图+护栏打回时可适当调高（或通过环境变量覆盖）
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "48"))
# Text2SQL：消息末尾连续失败 ToolMessage 条数达到此值时 agent 节点熔断结案，避免与护栏无限循环
TEXT2SQL_CONSECUTIVE_FAILURE_LIMIT = int(os.getenv("TEXT2SQL_CONSECUTIVE_FAILURE_LIMIT", "3"))

# Python：连续「需补 SQL 输入」类失败（failure_code）达到此值时 agent 提前终止；0 表示关闭
PYTHON_INPUT_NEEDS_SQL_STALL_LIMIT = int(os.getenv("PYTHON_INPUT_NEEDS_SQL_STALL_LIMIT", "0"))

# Schema RAG：外键 BFS 补全上限（防止高度连通 schema 撑爆上下文）
RAG_FK_MAX_TOTAL_TABLES = int(os.getenv("RAG_FK_MAX_TOTAL_TABLES", "16"))
RAG_FK_MAX_DEPTH = int(os.getenv("RAG_FK_MAX_DEPTH", "3"))
# Schema RAG：向量 Top-K、Planner 摘要表数、schema_search_tool 扩大 K（可通过环境变量覆盖）
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K_DEFAULT", "5"))
RAG_TOP_K_TOOL = int(os.getenv("RAG_TOP_K_TOOL", "8"))
RAG_PLANNER_MAX_TABLES = int(os.getenv("RAG_PLANNER_MAX_TABLES", "4"))

# SQL 执行：单次查询最大返回行数（无 LIMIT 时自动注入，防止 pd.read_sql OOM 与下游 GeoJSON 沙盒爆内存/超时）
# 全省级网格等场景应在 SQL 侧聚合或抽样；需全量明细时可用环境变量 SQL_MAX_RESULT_ROWS 提高（默认 10 万）
SQL_MAX_RESULT_ROWS = int(os.getenv("SQL_MAX_RESULT_ROWS", "100000"))
# PostgreSQL statement_timeout（毫秒），经引擎连接 options 下发；慢查询由库端取消
SQL_STATEMENT_TIMEOUT_MS = int(os.getenv("SQL_STATEMENT_TIMEOUT_MS", "30000"))
# 执行连接上 SET LOCAL idle_in_transaction_session_timeout（毫秒）
SQL_IDLE_IN_TRANSACTION_TIMEOUT_MS = int(os.getenv("SQL_IDLE_IN_TRANSACTION_TIMEOUT_MS", "30000"))

# Agent 主执行器 LLM：历史消息滑动窗口（不含 SystemMessage）；窗口大小不低于 MIN
AGENT_LLM_MESSAGE_WINDOW = int(os.getenv("AGENT_LLM_MESSAGE_WINDOW", "10"))
AGENT_LLM_MESSAGE_WINDOW_MIN = int(os.getenv("AGENT_LLM_MESSAGE_WINDOW_MIN", "4"))
AGENT_LLM_INJECT_QUESTION_ANCHOR = os.getenv("AGENT_LLM_INJECT_QUESTION_ANCHOR", "1").lower() in (
    "1",
    "true",
    "yes",
)
# text2sql 返回含 data_peek 时正文可达数万字符；过小会导致 Agent 看不到表格而臆答
AGENT_LLM_TOOLMESSAGE_MAX_CHARS = int(os.getenv("AGENT_LLM_TOOLMESSAGE_MAX_CHARS", "6000"))
# 护栏后一轮：仅向 LLM 绑定 expected_tools 子集（程序收窄工具空间）
AGENT_LLM_GUARDTOOL_NARROW = os.getenv("AGENT_LLM_GUARDTOOL_NARROW", "1").lower() in ("1", "true", "yes")
# 已收窄工具时不再向 system 注入护栏信号行（避免模型再读一遍文字）
AGENT_LLM_GUARDTOOL_SKIP_SYSTEM_SUFFIX = os.getenv("AGENT_LLM_GUARDTOOL_SKIP_SYSTEM_SUFFIX", "1").lower() in (
    "1",
    "true",
    "yes",
)

# v2: keep the Planner, but only expose tools that are compatible with the
# current blueprint step.  This prevents a completed SQL-only plan from
# drifting into an unnecessary Python or map-rendering call.
V2_PLAN_AWARE_TOOL_BINDING = os.getenv("V2_PLAN_AWARE_TOOL_BINDING", "1").lower() in (
    "1",
    "true",
    "yes",
)
V2_DETERMINISTIC_PLAN_EXECUTION = os.getenv("V2_DETERMINISTIC_PLAN_EXECUTION", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# v2: a schema-valid candidate is not automatically semantically correct.
# When real tool evidence exists, run the evidence-aware answer formatter so
# that answer_type and retained fields are aligned with the user question.
V2_EVIDENCE_FIRST_FINALIZATION = os.getenv("V2_EVIDENCE_FIRST_FINALIZATION", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Python 沙盒已返回协议化 JSON；再次让 LLM 改写容易丢字段或降低数值精度。
# SQL/普通 Agent 候选仍使用上面的证据结案器。
V2_PRESERVE_PYTHON_PAYLOAD = os.getenv("V2_PRESERVE_PYTHON_PAYLOAD", "1").lower() in (
    "1",
    "true",
    "yes",
)

# v2 generic control loop: independently verify the candidate against real
# tool evidence, then allow a bounded answer revision, local repair, or replan.
# The reviewer never receives benchmark reference answers.
V2_ENABLE_EVIDENCE_REVIEW = os.getenv("V2_ENABLE_EVIDENCE_REVIEW", "1").lower() in (
    "1",
    "true",
    "yes",
)
V2_MAX_EVIDENCE_REVIEWS = int(os.getenv("V2_MAX_EVIDENCE_REVIEWS", "2"))
V2_MAX_REPLANS = int(os.getenv("V2_MAX_REPLANS", "1"))
V2_REVIEW_MAX_TOKENS = int(os.getenv("V2_REVIEW_MAX_TOKENS", "1000"))

# 启动环境强校验（缺配置早暴露，避免执行到一半再失败）
if not LLM_API_KEY:
    warnings.warn(
        "未设置 LLM_API_KEY（或 OPENAI_API_KEY），系统将无法请求大模型服务。请在 .env 文件中配置。",
        stacklevel=2,
    )
if not DB_PASSWORD:
    warnings.warn(
        "未设置 DB_PASSWORD，系统可能无法连接到数据库。请在 .env 文件中配置。",
        stacklevel=2,
    )


def get_llm():
    """全局 LLM 工厂：根据 LLM_* 构造 ChatOpenAI（主 Agent、Text2SQL、LLM2Code 等共用）。"""
    from langchain_openai import ChatOpenAI
    streaming = (os.getenv("LLM_STREAMING") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    kwargs = {}
    if "dashscope" in (LLM_BASE_URL or "").lower():
        kwargs["http_client"] = _dashscope_tls12_http_client(timeout)
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0,
        streaming=streaming,
        stream_usage=True if streaming else None,
        timeout=timeout,
        max_retries=0,
        **kwargs,
    )
