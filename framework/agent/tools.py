# -*- coding: utf-8 -*-
"""Agent 工具集：Text2SQL、Python 分析、Schema 检索、地图渲染。"""
import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

import config
from prompts.llm2code import get_llm2code_messages
from prompts.text2sql import get_text2sql_messages
from tools.map_renderer import render_spatial_map
from tools.python_sandbox import SandboxOutputParseError, execute_python_sandbox
from tools.schema_retriever import build_rag_query, retrieve_top_k_schema_bundle
from tools.sql_executor import execute_sql_and_save_geojson
from utils.code_memory import load_code_templates, save_code_to_memory
from utils.code_utils import extract_python_code
from utils.python_analysis_input import validate_python_analysis_input_files
from utils.schema_utils import load_schemas_by_table_names

# text2sql 注入 data_peek 时：允许预览的最大行数（与 _build_sql_result_preview 的 cap 必须一致）
# 全美州级排序常 >50 行（50 州+DC+领地等），50 会导致不生成 data_peek、Agent 只能臆答
DATA_PEEK_MAX_ROWS = 100
# Markdown 预览正文长度上限（宽表多列时多行可超过 8k）
DATA_PEEK_MAX_CHARS = 24000


def _extract_schema_table_flags(schemas_yaml: str) -> dict[str, dict[str, bool]]:
    """从拼接后的 schema YAML 中提取表级 has_geometry 与 geometry 列声明。"""
    table_flags: dict[str, dict[str, bool]] = {}
    current_table: str | None = None
    current_has_geometry = False
    current_declares_geometry = False

    def flush_current() -> None:
        nonlocal current_table, current_has_geometry, current_declares_geometry
        if current_table:
            table_flags[current_table] = {
                "has_geometry": current_has_geometry,
                "declares_geometry": current_declares_geometry,
            }

    for raw_line in str(schemas_yaml).splitlines():
        line = raw_line.strip()
        if line.startswith("table_name:"):
            flush_current()
            current_table = line.split(":", 1)[1].strip()
            current_has_geometry = False
            current_declares_geometry = False
            continue
        if line.startswith("has_geometry:"):
            current_has_geometry = line.split(":", 1)[1].strip().lower() == "true"
            continue
        if line.startswith("name:") and line.split(":", 1)[1].strip() == "geometry":
            current_declares_geometry = True

    flush_current()
    return table_flags


def _build_geometry_guardrail(table_flags: dict[str, dict[str, bool]]) -> str:
    """基于 schema 构造几何字段防错提示。"""
    if not table_flags:
        return ""
    return "【核心规则】若需引用 geometry，必须严格遵循 Schema 中表与列的所属关系。若当前表无该列，必须通过 foreign_key JOIN 关联持有该列的表，严禁凭空捏造别名的 geometry 列。"


def _extract_table_columns_from_schema_yaml(schemas_yaml: str) -> dict[str, list[str]]:
    """从拼接后的 schema YAML 中提取每张表的列名列表。"""
    table_columns: dict[str, list[str]] = {}
    current_table: str | None = None

    for raw_line in str(schemas_yaml).splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("table_name:"):
            current_table = stripped.split(":", 1)[1].strip()
            table_columns.setdefault(current_table, [])
            continue
        if not current_table:
            continue
        if stripped.startswith("- name:") or stripped.startswith("name:"):
            col_name = stripped.split(":", 1)[1].strip()
            if col_name and col_name not in table_columns[current_table]:
                table_columns[current_table].append(col_name)

    return table_columns


def _build_schema_column_guardrail(schemas_yaml: str) -> str:
    """构造列级白名单提示，降低幻觉列与错表取列。"""
    return "【列名白名单警告】SELECT/WHERE/JOIN/GROUP BY 等子句中出现的所有列名，必须真实存在于给定的 Schema 定义中。不可因问题提到某指标就强行拼凑列名。"


def _extract_fk_pairs_from_schema_yaml(schemas_yaml: str) -> list[tuple[str, str, str, str]]:
    """提取外键关系，返回 (源表, 源列, 目标表, 目标列)。"""
    fk_pairs: list[tuple[str, str, str, str]] = []
    current_table: str | None = None
    current_col: str | None = None

    for raw_line in str(schemas_yaml).splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("table_name:"):
            current_table = stripped.split(":", 1)[1].strip()
            current_col = None
            continue
        if stripped.startswith("- name:") or stripped.startswith("name:"):
            current_col = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("foreign_key:") and current_table and current_col:
            fk_value = stripped.split(":", 1)[1].strip().strip("'\"")
            if not fk_value or fk_value.lower() == "null" or "." not in fk_value:
                continue
            ref_table, ref_col = fk_value.split(".", 1)
            ref_table = ref_table.strip()
            ref_col = ref_col.strip()
            if ref_table and ref_col:
                fk_pairs.append((current_table, current_col, ref_table, ref_col))

    return fk_pairs


def _build_fk_join_guardrail(schemas_yaml: str) -> str:
    """构造外键 JOIN 提示，强调可达列来源。"""
    return "【外键 JOIN 提示】若需跨表查询，优先依据 Schema 中的 foreign_key 关系进行 JOIN，严禁臆造关联条件。"


def _build_error_learning_prompt(error_feedback: str, schemas_yaml: str) -> str:
    """把数据库报错结构化为下一轮可学习的显式约束。"""
    feedback = str(error_feedback or "")
    lower_feedback = feedback.lower()
    extra_rules: list[str] = []

    missing_column_match = re.search(r"column\s+([a-zA-Z_][\w\.]*)\s+does not exist", feedback, re.IGNORECASE)
    if missing_column_match:
        missing_ref = missing_column_match.group(1)
        if "." in missing_ref:
            alias, col = missing_ref.split(".", 1)
            extra_rules.append(f"- 上轮报错列是 `{missing_ref}`。本轮必须重新核对别名 `{alias}` 是否真的来自含 `{col}` 的表。")
        else:
            extra_rules.append(f"- 上轮报错列是 `{missing_ref}`。本轮必须确认该列真实存在于 Schema 白名单中，若不存在则改写 SQL。")

    if "does not exist" in lower_feedback:
        extra_rules.append("- 遇到 UndefinedColumn / does not exist 时，先检查列属于哪张表，再检查 JOIN 是否把拥有该列的表接入。")
    # 🌟 新增：提取并放大 PostgreSQL 的官方 HINT
    if "hint:" in lower_feedback:
        hint_match = re.search(r"HINT:\s*([^\n]+)", feedback, re.IGNORECASE)
        if hint_match:
            extra_rules.append(f"- 🚨【数据库原生提示 (极其重要)】：数据库给出了官方修复建议：{hint_match.group(1).strip()}！请务必采纳此建议修改表别名！")
    if "geometry" in lower_feedback:
        extra_rules.append("- 本轮必须逐个检查所有 `别名.geometry`，只允许出现在明确声明 `geometry` 列的表别名上。")
    if "area_ratio" in lower_feedback:
        extra_rules.append("- 本轮若需要 `area_ratio`，只能从真正声明该列的表获取，不能从 static/dynamic 别名臆造。")
    if "timeout" in lower_feedback or "canceling statement" in lower_feedback:
        extra_rules.append("- 本轮必须简化 SQL，避免复杂嵌套与不必要 JOIN，优先拆成多条基础 SELECT。")

    table_columns = _extract_table_columns_from_schema_yaml(schemas_yaml)
    candidate_tables: list[str] = []
    for table_name, columns in table_columns.items():
        if "geometry" in lower_feedback and "geometry" in columns:
            candidate_tables.append(f"{table_name}.geometry")
        if "area_ratio" in lower_feedback and "area_ratio" in columns:
            candidate_tables.append(f"{table_name}.area_ratio")

    if candidate_tables:
        extra_rules.append("- 根据当前 Schema，可用的相关真实列包括：" + ", ".join(candidate_tables))

    if not extra_rules:
        extra_rules.append("- 请逐列、逐表、逐别名核对 SQL，确保每个字段都能在 Schema 中定位到来源。")

    return "【从报错中学习】\n" + "\n".join(extra_rules)


class SQLQueryItem(BaseModel):
    sql: str = Field(description="完整 SQL 字符串")
    output_filename: str = Field(description="导出文件名")
    has_geometry: bool = Field(description="结果是否包含几何列")


class SQLGenerationResult(BaseModel):
    queries: list[SQLQueryItem] = Field(description="SQL 列表，按执行顺序")


def _strip_markdown_fences(text: str) -> str:
    """去除模型常见的 markdown 代码块包裹。"""
    content = str(text or "").strip()
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return content


def _strip_leading_json_noise(text: str) -> str:
    """切除 ok / Here is the SQL 等常见非 JSON 前缀，便于 _extract_first_json_object 定位首个 `{`。"""
    s = str(text or "").strip()
    for _ in range(10):
        before = s
        s = re.sub(r"^(?i)(ok|好的|是的|可以)\b\s*[\.。:：,，\-—、\s]*", "", s).lstrip()
        s = re.sub(
            r"^(?i)(here\s+is\s+(the\s+)?(sql|json|result|output)[^.!?\n]*[\.!?\n]\s*)",
            "",
            s,
        ).lstrip()
        s = re.sub(r"^(?i)(the\s+following\s+is\s+[^\n]+\n\s*)", "", s).lstrip()
        s = re.sub(r"^(?i)(below\s+is\s+[^\n]+\n\s*)", "", s).lstrip()
        if s == before:
            break
    return s


def _extract_first_json_object(text: str) -> str:
    """从混杂文本中提取首个完整 JSON 对象。"""
    content = _strip_leading_json_noise(_strip_markdown_fences(text))
    start = content.find("{")
    if start == -1:
        return content

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return content[start:idx + 1]

    return content


def _coerce_structured_sql_generation_result(raw_result) -> SQLGenerationResult:
    """兼容不同 LLM/网关返回形态，尽量恢复为 SQLGenerationResult。"""
    if isinstance(raw_result, SQLGenerationResult):
        return raw_result

    if isinstance(raw_result, dict):
        return SQLGenerationResult.model_validate(raw_result)

    if isinstance(raw_result, str):
        normalized = _extract_first_json_object(raw_result)
        if not normalized.strip():
            raise ValueError("解析失败：模型返回的文本为空或未找到有效的 JSON 结构。")
        return SQLGenerationResult.model_validate_json(normalized)

    content = getattr(raw_result, "content", None)
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            normalized = _extract_first_json_object("\n".join(text_parts))
            if not normalized.strip():
                raise ValueError("解析失败：模型返回的文本为空或未找到有效的 JSON 结构。")
            return SQLGenerationResult.model_validate_json(normalized)
    elif isinstance(content, str):
        normalized = _extract_first_json_object(content)
        if not normalized.strip():
            raise ValueError("解析失败：模型返回的文本为空或未找到有效的 JSON 结构。")
        return SQLGenerationResult.model_validate_json(normalized)

    if hasattr(raw_result, "model_dump"):
        dumped = raw_result.model_dump()
        if isinstance(dumped, dict):
            return SQLGenerationResult.model_validate(dumped)

    raise ValidationError.from_exception_data(
        title="SQLGenerationResult",
        line_errors=[{
            "type": "value_error",
            "loc": ("queries",),
            "msg": f"无法从模型返回结果中恢复结构化 JSON，原始类型: {type(raw_result).__name__}",
            "input": raw_result,
            "ctx": {"error": "structured_output_unrecoverable"},
        }],
    )


class SchemaSearchArgs(BaseModel):
    question: str = Field(description="当前用户的自然语言问题")
    slots_json: str = Field(default="", description="意图解析结果（JSON 字符串），用于结合槽位检索相关 Schema")
    top_k: int = Field(default=config.RAG_TOP_K_TOOL, description="检索返回的主表数量")


class PythonAnalysisArgs(BaseModel):
    question: str = Field(
        description=(
            "【必填】用自然语言极其简短地描述你要做什么（例如：'执行 K-Means 聚类'）。"
            "底层系统内置了代码生成器会自动替你写代码！"
            "绝对禁止在这里写 Python 代码！绝对禁止自创 analysis_steps 等复杂结构！"
        )
    )
    geojson_paths: list[str] = Field(
        default_factory=list,
        description="【极其重要】可用于 Python 分析的数据文件路径列表。你必须将前序步骤导出的所有 .geojson 和 .json 文件的纯文件名，全部放入这唯一的一个数组中！"
    )


class MapRenderingArgs(BaseModel):
    geojson_path: str = Field(description="需要渲染的 GeoJSON 文件路径")


def _parse_slots_json(slots_json: Optional[str]) -> dict:
    """安全解析 slots_json。"""
    if not slots_json:
        return {}
    try:
        return json.loads(slots_json)
    except Exception:
        return {}


def _normalize_workspace_paths(paths: Optional[list[str]]) -> list[str]:
    """将传入路径标准化为 Python 沙盒工作目录可直接访问的相对文件名列表。"""
    if not paths:
        return []

    normalized: list[str] = []
    workspace_dir = config.WORKSPACE_DIR.resolve()

    for raw_path in paths:
        if not raw_path:
            continue

        try:
            candidate = Path(raw_path)
        except Exception:
            continue

        file_name = candidate.name
        if not file_name:
            continue

        resolved_in_workspace = (workspace_dir / file_name).resolve()
        if resolved_in_workspace.exists():
            normalized.append(file_name)
            continue

        try:
            resolved_candidate = candidate.resolve()
        except Exception:
            resolved_candidate = candidate

        if resolved_candidate.exists():
            try:
                if resolved_candidate.is_relative_to(workspace_dir):
                    normalized.append(str(resolved_candidate.relative_to(workspace_dir)).replace("\\", "/"))
                else:
                    normalized.append(file_name)
            except Exception:
                normalized.append(file_name)
            continue

        normalized.append(file_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for path in normalized:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _infer_file_type_from_path(path: str) -> str:
    """仅基于后缀推断文件类型（用于路径兜底）。"""
    lower_path = str(path).lower()
    if lower_path.endswith(".geojson"):
        return "geojson"
    if lower_path.endswith(".json"):
        return "json"
    return "json"


def _match_path_by_filename(output_filename: str, path_names: list[str]) -> str | None:
    """根据文件名匹配标准化路径，支持同 stem 不同后缀的回退匹配。"""
    if not output_filename:
        return None

    out_name = Path(output_filename).name
    if not out_name:
        return None

    # 1) 严格按文件名匹配
    exact = next((p for p in path_names if Path(p).name == out_name), None)
    if exact:
        return exact

    # 2) 同 stem 不同后缀（例如期望 .geojson，实际导出为 .json）
    out_stem = Path(out_name).stem
    return next((p for p in path_names if Path(p).stem == out_stem), None)


def _infer_sql_result_file_type(sql: str, path: str, has_geometry: bool | None = None) -> str:
    """根据 SQL 与导出结果推断文件类型，区分 geojson / json / json_wkb_hex。"""
    lower_path = str(path).lower()
    if lower_path.endswith(".geojson"):
        return "geojson"

    normalized_sql = re.sub(r"\s+", " ", str(sql or "")).strip().lower()
    if has_geometry is False:
        return "json"

    geometry_alias_patterns = [
        r"\bas\s+geometry\b",
        r'\bas\s+"geometry"\b',
        r"\bst_asgeojson\s*\(",
        r"\bst_asmvtgeom\s*\(",
    ]
    if any(re.search(pattern, normalized_sql) for pattern in geometry_alias_patterns):
        return "geojson" if lower_path.endswith(".geojson") else "json"

    geometry_select_patterns = [
        r"\b[a-z_][a-z0-9_]*\.geometry\b",
        r'\b[a-z_][a-z0-9_]*\."geometry"\b',
        r"\bgeometry\b",
        r'\b"geometry"\b',
    ]
    if any(re.search(pattern, normalized_sql) for pattern in geometry_select_patterns):
        return "json_wkb_hex" if lower_path.endswith(".json") else "geojson"

    return "json"


def _format_preview_as_markdown(data_list: list[dict]) -> str:
    """将数据字典列表转换为紧凑的 Markdown 表格，降低 Token 消耗。"""
    if not data_list or not isinstance(data_list, list) or not isinstance(data_list[0], dict):
        return json.dumps(data_list, ensure_ascii=False)
    keys = list(data_list[0].keys())
    header = "| " + " | ".join(str(k) for k in keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    lines = [header, sep]
    for row in data_list:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def _build_sql_result_preview(path: str, file_type: str, *, max_rows: int = DATA_PEEK_MAX_ROWS):
    """生成供 LLM 参考的预览；行数上限须与 execute_text2sql_logic 中 data_peek 的 row_count 门一致。"""
    cap = max(1, min(int(max_rows), DATA_PEEK_MAX_ROWS))
    with open(path, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    if file_type == "geojson":
        feats = file_data.get("features", [])[:cap]
        return [feat.get("properties", {}) for feat in feats]

    if isinstance(file_data, list):
        return file_data[:cap]

    return file_data


# execute_python_analysis_logic 返回用的结构化失败码（与 agent.nodes 门禁对齐）
FAILURE_CODE_PYTHON_REQUIRES_GEOMETRY_EXPORT = "python_requires_geometry_export"
FAILURE_CODE_PYTHON_INPUT_UNREADABLE = "python_input_file_unreadable"


def _blob_attribute_stats_primary(blob: str, slots: dict) -> bool:
    """问题/步骤是否主要表达属性统计、回归、时序等（无强空间算子时倾向允许纯 JSON）。"""
    s = slots if isinstance(slots, dict) else {}
    if str(s.get("analytical_method") or "").strip().lower() in (
        "regression",
        "correlation",
        "clustering",
        "cluster",
    ):
        return True
    raw = blob or ""
    low = raw.lower()
    zh = (
        "回归",
        "相关性",
        "相关系数",
        "皮尔逊",
        "协方差",
        "聚类",
        "月差",
        "环比",
        "同比",
        "时序",
        "趋势",
        "排序",
        "分组统计",
        "均值",
        "求和",
        "累计",
    )
    en = (
        "regression",
        "correlation",
        "pearson",
        "covariance",
        "cluster",
        "month over month",
        "mom",
        "yoy",
        "time series",
        "trend",
        "ranking",
        "rank ",
        "group by",
        "seasonal",
    )
    return any(t in raw for t in zh) or any(t in low for t in en)


def _blob_requires_geometry_semantics(blob: str, slots: dict) -> bool:
    """显式空间拓扑/缓冲/距离/叠置/地图渲染等（与「仅属性表统计」区分）。"""
    s = slots if isinstance(slots, dict) else {}
    vis = str(s.get("visualization") or "")
    if vis and any(k in vis for k in ("地图", "map", "folium", "可视化", "出图", "渲染")):
        return True
    raw = blob or ""
    low = raw.lower()
    zh_spatial = (
        "相交",
        "交叉",
        "缓冲",
        "叠置",
        "叠加",
        "空间叠置",
        "毗邻",
        "邻接",
        "邻域",
        "距离",
        "拓扑",
        "空间过滤",
        "空间筛选",
        "空间查询",
        "地图渲染",
        "渲染地图",
        "画地图",
        "完全位于",
        "部分位于",
        "完全落入",
        "部分落入",
        "完全落在",
        "部分落在",
        "范围内完全或部分",
        "几何质心",
        "几何中心",
        "几何面积",
        "州域总面积",
        "投影坐标系",
    )
    if any(t in raw for t in zh_spatial):
        return True
    # “每类中包含多少空间单元”等普通中文也同时含有“包含”和“空间”，不能据此
    # 推断为几何 contains。空间关系必须由上面的明确术语、距离阈值或槽位给出。
    en_spatial = (
        "intersect",
        "intersection",
        "buffer",
        "overlay",
        "spatial overlay",
        "spatial join",
        "within distance",
        "dwithin",
        "touches",
        "adjacent",
        "topology filter",
        "st_intersects",
        "st_within",
        "st_buffer",
        "st_contains",
        "st_dwithin",
        "st_touches",
        "st_crosses",
        "st_overlaps",
        "map render",
        "folium",
    )
    if any(t in low for t in en_spatial):
        return True
    if re.search(r"\bepsg\s*:\s*\d+\b|\bepsg\s*\d+\b", low):
        return True
    if re.search(r"\bst_[a-z][a-z0-9_]*\s*\(", low):
        return True
    return False


def infer_require_geojson_for_python(
    slots: dict,
    question: str,
    current_plan_step: str = "",
) -> bool:
    """
    仅当槽位或问题/蓝图步骤体现空间分析、拓扑、缓冲、距离、叠置或地图渲染时，
    才要求批次内存在 .geojson；纯回归/相关性/时序/排序等属性任务允许仅 .json。
    """
    s = slots if isinstance(slots, dict) else {}
    if str(s.get("spatial_operation") or "").strip():
        return True
    if str(s.get("spatial_predicate") or "").strip():
        return True
    if str(s.get("spatial_threshold") or "").strip():
        return True

    blob = f"{current_plan_step} {question or ''}".strip()
    if _blob_requires_geometry_semantics(blob, s):
        return True
    if _blob_attribute_stats_primary(blob, s):
        return False
    return False


def _infer_require_geojson_for_python(
    slots: dict, question: str, current_plan_step: str = ""
) -> bool:
    return infer_require_geojson_for_python(slots, question, current_plan_step)


def _validate_python_analysis_input_files(
    path_names: list[str], *, require_geojson: bool = True
) -> tuple[bool, str, str | None]:
    return validate_python_analysis_input_files(
        config.WORKSPACE_DIR.resolve(), path_names, require_geojson=require_geojson
    )


_FORBIDDEN_SPATIAL_FUNCTIONS = [
    "st_touches",
    "st_intersects",
    "st_contains",
    "st_within",
    "st_dwithin",
    "st_crosses",
    "st_overlaps",
    "st_disjoint",
    "st_covers",
    "st_coveredby",
    "st_equals",
    "st_relate",
    "st_distance",
    "st_distancesphere",
    "st_distancespheroid",
    "st_area",
    "st_length",
    "st_perimeter",
    "st_centroid",
    "st_buffer",
    "st_convexhull",
    "st_union",
    "st_intersection",
    "st_difference",
    "st_symdifference",
    "st_simplify",
    "st_transform",
    "st_makeenvelope",
    "st_envelope",
    "st_closestpoint",
    "st_snap",
    "st_voronoipolygons",
    "st_delaunaytriangles",
    "st_clusterkmeans",
    "st_clusterdbscan",
]

_FORBIDDEN_SPATIAL_PATTERN = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_SPATIAL_FUNCTIONS) + r")\s*\(",
    re.IGNORECASE,
)

_FORBIDDEN_BBOX_OPERATOR_PATTERN = re.compile(
    r'("?geometry"?\s*)(&&|&<|&>|<<|>>)(\s*"?geometry"?)',
    re.IGNORECASE,
)


def _contains_forbidden_spatial_function(sql: str) -> str | None:
    """拦截 Text2SQL 越权做高级空间计算的 SQL。

    匹配规则：
    1. ST_* 函数名（忽略大小写）后紧跟可选空白和左括号。
    2. PostGIS 绑定框运算符（&&、&<、&>、<<、>>）。
    返回首个命中的函数名/运算符，未命中返回 None。
    """
    normalized_sql = re.sub(r"\s+", " ", str(sql or ""))
    match = _FORBIDDEN_SPATIAL_PATTERN.search(normalized_sql)
    if match:
        return match.group(1).lower()
    bbox_match = _FORBIDDEN_BBOX_OPERATOR_PATTERN.search(normalized_sql)
    if bbox_match:
        return bbox_match.group(2)
    return None


def _paren_depth_before_index(sql: str, idx: int) -> int:
    depth = 0
    for k in range(max(0, idx)):
        if sql[k] == "(":
            depth += 1
        elif sql[k] == ")":
            depth = max(0, depth - 1)
    return depth


def _has_outer_sql_clause(sql: str, pattern: str) -> bool:
    """Return whether *pattern* occurs at parenthesis depth zero."""
    s = str(sql or "")
    return any(
        _paren_depth_before_index(s, match.start()) == 0
        for match in re.finditer(pattern, s, re.IGNORECASE)
    )


def validate_ordered_sql_projection(
    sql: str,
    *,
    ordered_result_required: bool,
    requested_top_k: int | None = None,
    analysis_export_for_python: bool = False,
) -> str | None:
    """Validate ordering invariants declared by the structured answer contract.

    This is a relational correctness gate, not a question-pattern heuristic.
    Python-analysis exports are excluded because Python owns final ordering.
    """
    if analysis_export_for_python or not ordered_result_required:
        return None
    if not _has_outer_sql_clause(sql, r"\border\s+by\b"):
        return (
            "Top-K SQL validation failed: the final SELECT has no outer ORDER BY. "
            "Ordering only inside a subquery is not preserved after an outer JOIN. "
            "Rewrite the query so the final SELECT explicitly orders by the ranking metric."
        )
    # A full ranked series needs a stable outer ORDER BY but must not be
    # truncated.  LIMIT/FETCH is required only when the question-derived
    # contract declares an explicit Top-K value.
    if not (isinstance(requested_top_k, int) and requested_top_k > 0):
        return None
    limit_matches = [
        match
        for match in re.finditer(
            r"\blimit\s+(\d+)\b|\bfetch\s+(?:first|next)\s+(\d+)\s+rows\b",
            str(sql or ""),
            re.IGNORECASE,
        )
        if _paren_depth_before_index(str(sql or ""), match.start()) == 0
    ]
    if not limit_matches:
        return (
            "Top-K SQL validation failed: the final SELECT has no outer LIMIT/FETCH. "
            "Rewrite the query so the final projection enforces the requested result count."
        )
    actual_limit = int(next(group for group in limit_matches[-1].groups() if group is not None))
    if actual_limit != requested_top_k:
        return (
            "Top-K SQL validation failed: the outer LIMIT/FETCH does not match "
            f"the structured answer contract (expected {requested_top_k}, got {actual_limit})."
        )
    return None


def validate_sql_semantic_bindings(
    sql_text: str,
    semantic_bindings: list[dict[str, Any]] | None,
) -> str | None:
    """Ensure SCGA output preserves the Schema pre-filtering field lineage.

    The check is deliberately structural: it verifies table/column provenance,
    never expected answer values.  Multiple generated SQL statements may jointly
    satisfy the bindings because a single SCGA call can export several files for
    STCA.
    """
    bindings = [item for item in (semantic_bindings or []) if isinstance(item, dict)]
    if not bindings:
        return None
    compact_sql = re.sub(r'["`\[\]]', "", str(sql_text or "")).casefold()
    missing: list[str] = []
    for binding in bindings:
        concept = str(binding.get("concept") or "unknown_metric")
        table_name = str(binding.get("table") or "").strip()
        columns = [str(item).strip() for item in (binding.get("columns") or []) if str(item).strip()]
        if table_name and table_name.casefold() not in compact_sql:
            missing.append(f"{concept}:table={table_name}")
            continue
        absent_columns = [column for column in columns if column.casefold() not in compact_sql]
        if absent_columns:
            missing.append(f"{concept}:columns={','.join(absent_columns)}")
    if not missing:
        return None
    return (
        "SCGA Schema 语义绑定门禁失败：生成的 SQL 未覆盖 Intent/Schema 预过滤确认的数据血缘："
        + "；".join(missing)
        + "。不得用数值相近但语义不同的字段替代；请依据绑定表字段重新生成 SQL。"
    )


def validate_sql_spatial_source_coverage(
    sql_text: str,
    required_geometry_tables: list[str] | None,
) -> str | None:
    """Ensure cross-layer topology exports every raw geometry operand.

    The gate does not prescribe a spatial algorithm or SQL function.  It only
    checks that each role-bound geometry table is represented in a SELECT that
    exports ``geometry`` for downstream STCA processing.
    """
    required = list(
        dict.fromkeys(
            str(table).strip()
            for table in (required_geometry_tables or [])
            if str(table).strip()
        )
    )
    if len(required) < 2:
        return None
    statements = [part for part in re.split(r";\s*", str(sql_text or "")) if part.strip()]
    missing: list[str] = []
    for table in required:
        table_pattern = re.compile(rf"\b{re.escape(table)}\b", re.IGNORECASE)
        covered = any(
            table_pattern.search(statement)
            and re.search(r"\bgeometry\b", statement, re.IGNORECASE)
            for statement in statements
        )
        if not covered:
            missing.append(table)
    if not missing:
        return None
    return (
        "SCGA 空间输入完整性门禁失败：该跨图层空间任务必须分别导出所有空间操作数的原始 geometry；"
        "当前缺少图层："
        + "、".join(missing)
        + "。请仅执行基础 SELECT，可用多条查询分别导出，不得在 SQL 中执行空间拓扑函数。"
    )


def validate_sql_condition_coverage(
    sql_text: str,
    semantic_bindings: list[dict[str, Any]] | None,
    condition_clauses: list[str] | None,
) -> str | None:
    """Check that bound metrics mentioned in comparison clauses are predicates.

    This gate is value-independent: it does not encode benchmark thresholds or
    answers.  It only prevents SCGA from selecting a requested field while
    silently omitting the user's ``higher/lower/at least`` condition.
    """
    clauses = [str(item).strip() for item in (condition_clauses or []) if str(item).strip()]
    bindings = [item for item in (semantic_bindings or []) if isinstance(item, dict)]
    if not clauses or not bindings:
        return None

    compact_sql = re.sub(r'["`\[\]]', "", str(sql_text or "")).casefold()
    inequality = re.compile(r"(?:<=|>=|<>|!=|<|>)")
    equality = re.compile(r"(?<![<>!])=(?!=)")
    missing: list[str] = []

    comparison_words = (
        r"不超过|不少于|不低于|不高于|高出|低于|高于|超过|大于|小于|等于|至少|至多"
    )
    equality_words = r"等于|为|是"
    concept_bindings: dict[str, list[dict[str, Any]]] = {}
    for item in bindings:
        concept_bindings.setdefault(str(item.get("concept") or "unknown_metric"), []).append(item)

    def derived_cross_table_comparison_is_covered(concept: str) -> bool:
        """Accept a comparison between derived aliases of the same metric.

        Multi-granularity SQL commonly computes ``AVG(pm25_mean) AS monthly_avg``
        in a CTE and compares ``monthly_avg > annual_avg`` outside that CTE.
        The physical columns therefore are not adjacent to the final operator,
        although both bound source tables and columns are present.
        """
        siblings = concept_bindings.get(concept, [])
        tables = {
            str(item.get("table") or "").strip().casefold()
            for item in siblings
            if str(item.get("table") or "").strip()
        }
        if len(tables) < 2:
            return False
        for item in siblings:
            table = str(item.get("table") or "").strip().casefold()
            columns = [
                str(column).strip().casefold()
                for column in (item.get("columns") or [])
                if str(column).strip()
            ]
            if table and not re.search(rf"\b{re.escape(table)}\b", compact_sql):
                return False
            if columns and not any(re.search(rf"\b{re.escape(column)}\b", compact_sql) for column in columns):
                return False
        predicate_regions = re.findall(
            r"\b(?:where|having)\b(.+?)(?=\b(?:group\s+by|order\s+by|limit|union|returning)\b|$)",
            compact_sql,
            re.DOTALL,
        )
        return any(inequality.search(region) for region in predicate_regions)

    def nearby_condition_fragments(clause: str, terms: list[str]) -> list[str]:
        """Return only fragments where a metric is locally attached to a predicate.

        A long Chinese clause can contain both ``era5_temp 高于 18.9`` and
        ``按 cell_id 排序``.  Treating every mentioned field as an inequality
        operand makes the identifier fail the gate forever.  Local windows keep
        the condition binding structural without encoding any benchmark value.
        """
        found: list[str] = []
        raw = str(clause or "")
        for term in terms:
            term_text = str(term or "").strip()
            if not term_text:
                continue
            pattern = re.compile(
                rf"{re.escape(term_text)}.{{0,24}}(?:{comparison_words}|{equality_words})",
                re.IGNORECASE,
            )
            found.extend(match.group(0) for match in pattern.finditer(raw))
        return found

    for binding in bindings:
        concept = str(binding.get("concept") or "unknown_metric")
        terms = [concept, *(binding.get("matched_terms") or [])]
        related_fragments = [
            fragment
            for clause in clauses
            for fragment in nearby_condition_fragments(clause, terms)
        ]
        if not related_fragments:
            continue
        if derived_cross_table_comparison_is_covered(concept):
            continue
        needs_inequality = any(
            re.search(comparison_words, fragment)
            and not re.search(equality_words, fragment)
            for fragment in related_fragments
        )
        columns = [str(item).strip().casefold() for item in (binding.get("columns") or []) if str(item).strip()]
        predicate_found = False
        for column in columns:
            for match in re.finditer(rf"\b{re.escape(column)}\b", compact_sql):
                # Only inspect the expression following this column reference;
                # stop at a new SQL clause so an unrelated predicate elsewhere
                # cannot satisfy the requirement.
                tail = compact_sql[match.end(): min(len(compact_sql), match.end() + 80)]
                expression_tail = re.split(
                    r"\b(?:and|or|from|join|where|group\s+by|order\s+by|having|select)\b",
                    tail,
                    maxsplit=1,
                )[0]
                if (
                    inequality.search(expression_tail)
                    if needs_inequality
                    else (
                        inequality.search(expression_tail)
                        or equality.search(expression_tail)
                        or re.search(r"\b(?:in|like|is)\b", expression_tail)
                    )
                ):
                    predicate_found = True
                    break
            if predicate_found:
                break
        if not predicate_found:
            if concept not in missing:
                missing.append(concept)

    if not missing:
        return None
    return (
        "SCGA 条件覆盖门禁失败：以下已绑定指标出现在用户比较条件中，但 SQL 未体现相应筛选谓词："
        + "、".join(missing)
        + "。请逐项落实 Intent Understanding 传入的约束清单。"
    )


def _strip_outer_order_by_without_limit(sql: str) -> tuple[str, str | None]:
    """
    去掉最外层、括号深度为 0 的尾部 ORDER BY（整段 SQL 无 LIMIT 时）。
    用于「导出供 Python 分析」场景，避免无意义全表排序。
    """
    s = str(sql or "").strip().rstrip(";").strip()
    if not s:
        return sql, None
    if re.search(r"\blimit\b", s, re.IGNORECASE):
        return sql, None
    last_ob_start: int | None = None
    for m in re.finditer(r"\border\s+by\b", s, re.IGNORECASE):
        if _paren_depth_before_index(s, m.start()) == 0:
            last_ob_start = m.start()
    if last_ob_start is None:
        return sql, None
    new_sql = s[:last_ob_start].rstrip()
    if not new_sql:
        return sql, None
    note = "分析导出任务：已自动移除无 LIMIT 的 ORDER BY（降低排序开销、避免误导下游）"
    return new_sql, note


_USA_CAMEL_COL_PATTERN = re.compile(
    r'(?<!["a-zA-Z0-9_])((?:[a-zA-Z_][\w]*\.)?)(shapeName|shapeID|shapeISO|shapeGroup|Level)\b(?!["])'
)


def fix_usa_spatial_camel_column_refs(sql: str) -> tuple[str, bool]:
    """
    将未加引号的驼峰列引用改为 PostgreSQL 合法双引号标识符。
    支持带别名(s.shapeName)或不带别名(shapeName)的情况。
    """
    s = str(sql or "")
    if not s.strip():
        return sql, False

    def _repl(m: re.Match[str]) -> str:
        return f'{m.group(1)}"{m.group(2)}"'

    new_s, n = _USA_CAMEL_COL_PATTERN.subn(_repl, s)
    return new_s, n > 0


def normalize_sql_before_execution(
    sql: str,
    *,
    analysis_export_for_python: bool,
) -> tuple[str, list[str]]:
    """
    SQL 执行前程序级修正。返回 (修正后 SQL, 人类可读说明列表)。
    """
    notes: list[str] = []
    s = str(sql or "")
    s, camel_fixed = fix_usa_spatial_camel_column_refs(s)
    if camel_fixed:
        notes.append("已将驼峰列引用改写为双引号标识符（shapeName/shapeID/shapeISO/shapeGroup/Level）")
    if analysis_export_for_python:
        s2, ob_note = _strip_outer_order_by_without_limit(s)
        if ob_note:
            notes.append(ob_note)
            s = s2
    return s, notes


def validate_explicit_geojson_paths_in_workspace(paths: list[str]) -> str | None:
    """
    检查路径 basename 是否存在于 config.WORKSPACE_DIR。
    供 python_analysis_tool 显式传入 geojson_paths 时硬校验，防止模型幻觉文件名。
    """
    ws = Path(config.WORKSPACE_DIR)
    missing: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        bn = Path(str(raw).replace("\\", "/")).name.strip()
        if not bn:
            continue
        if bn in seen:
            continue
        seen.add(bn)
        if not (ws / bn).is_file():
            missing.append(bn)
    if not missing:
        return None
    return (
        "当前 Python 步通过 geojson_paths 收到的以下文件在工作区中不存在，疑似模型幻觉或未先成功执行 text2sql 导出："
        + ", ".join(missing)
    )


@tool(args_schema=SchemaSearchArgs)
def schema_search_tool(
    question: str, slots_json: Optional[str] = None, top_k: int = config.RAG_TOP_K_TOOL
) -> str:
    """检索与当前问题相关的表结构 Schema；当预检索结果不足时应优先调用本工具补充上下文。"""
    slots = _parse_slots_json(slots_json)
    schema_bundle = retrieve_top_k_schema_bundle(
        slots, k=top_k, natural_language_query=build_rag_query("tool", question=question)
    )
    return json.dumps({
        "schemas": schema_bundle.get("schemas", []),
        "schemas_yaml": schema_bundle.get("schemas_yaml", ""),
        "table_names": schema_bundle.get("table_names", []),
        "semantic_bindings": schema_bundle.get("semantic_bindings", []),
        "schema_coverage": schema_bundle.get("schema_coverage", {}),
    }, ensure_ascii=False, indent=2)


class Text2SQLArgs(BaseModel):
    question: str = Field(
        description=(
            "与当前蓝图步骤一致的自然语言子任务描述（可选）。"
            "Text2SQL 节点在有执行蓝图时以蓝图当前步为主生成 SQL，本参数仅作辅助对齐。"
        )
    )
    table_names: list[str] = Field(
        default_factory=list,
        description="强烈建议留空（传递 []）！系统会根据 question 自动触发 RAG 检索出正确的表。仅当发生报错，你需要强制干预选表时才手动传入表名。",
    )


def execute_text2sql_logic(
    sql_task: str,
    table_names: list[str],
    user_context: Optional[str] = None,
    error_feedback: Optional[str] = None,
    schemas_yaml: Optional[str] = None,
    analysis_export_for_python: bool = False,
    ordered_result_required: bool = False,
    requested_top_k: int | None = None,
    semantic_bindings: list[dict[str, Any]] | None = None,
    condition_clauses: list[str] | None = None,
    required_geometry_tables: list[str] | None = None,
) -> str:
    """根据蓝图子任务（及可选全局背景）与相关表结构生成 SQL，执行后导出 GeoJSON/JSON。"""
    schemas_yaml = schemas_yaml or load_schemas_by_table_names(table_names)
    print("  [Text2SQL] 正在调用 LLM 生成 SQL…", flush=True)
    system_msg, user_msg = get_text2sql_messages(sql_task, schemas_yaml, user_context=user_context)
    messages = [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
    table_flags = _extract_schema_table_flags(schemas_yaml)
    geometry_guardrail = _build_geometry_guardrail(table_flags)
    column_guardrail = _build_schema_column_guardrail(schemas_yaml)
    fk_guardrail = _build_fk_join_guardrail(schemas_yaml)
    if geometry_guardrail:
        messages.append(HumanMessage(content=geometry_guardrail))
    if column_guardrail:
        messages.append(HumanMessage(content=column_guardrail))
    if fk_guardrail:
        messages.append(HumanMessage(content=fk_guardrail))

    if error_feedback:
        ef = str(error_feedback)
        last_error_lower = ef.lower()
        if "timeout" in last_error_lower or "canceling statement" in last_error_lower:
            targeted_warning = "【严重超时警告】：SQL 执行超时！请拆解为更基础的 SELECT 语句。\n"
        elif "does not exist" in last_error_lower:
            targeted_warning = "【列名或别名错误】：请仔细检查列名、别名与关联关系，尤其核对 geometry 是否被错误地写在不含几何列的表别名上。\n"
        else:
            targeted_warning = "请仔细检查 SQL 的语法或表关联逻辑是否正确。\n"

        reflection_guide = (
            "错误反馈分析：数据库报告某个字段不存在（UndefinedColumn / does not exist），"
            "通常是因为你把某个列挂在了错误的表别名上（例如把属于表 A 的列写成了表 B 的别名）。\n"
            "修正指南：\n"
            "1) 不要随意删除 JOIN 语句！多表逻辑可能是对的。\n"
            "2) 重新检查 Schema 列白名单：该字段到底属于哪个 table_name。\n"
            "3) 仅修正错误的表别名前缀/补充必要 JOIN，保持其余正确逻辑不变。\n"
        )
        messages.append(HumanMessage(content=(
            f"前一次执行 SQL 发生以下错误：\n{ef}\n"
            f"{targeted_warning}"
            f"{_build_error_learning_prompt(ef, schemas_yaml)}\n"
            f"{reflection_guide}"
            "请务必严格按照要求的 JSON 格式输出修正后的结果，必须且只能输出包含 `queries` 数组的合法 JSON 字符串！"
        )))

    llm = config.get_llm()
    try:
        raw_msg = llm.invoke(messages)
        # 🌟 让大模型的原始回复（哪怕带了 ok 前缀）直接进入我们的强力清洗器
        gen = _coerce_structured_sql_generation_result(raw_msg)
        if not gen or not getattr(gen, "queries", None):
            raise ValueError("LLM 返回了空对象或未提取到 queries 列表")
    except Exception as e:
        raw_preview = ""
        try:
            rm = locals().get("raw_msg")
            if rm is not None:
                c = getattr(rm, "content", None)
                raw_preview = str(c) if c is not None else str(rm)
        except Exception:
            raw_preview = ""
        if len(raw_preview) > 1200:
            raw_preview = raw_preview[:1200] + "... (已截断)"
        print(
            f"  [Text2SQL] 结构化解析失败：{type(e).__name__}: {e}\n"
            f"  [Text2SQL] 原始模型输出预览（可对照网关/模板）：\n{raw_preview or '（空）'}",
            flush=True,
        )
        return json.dumps({
            "queries": [],
            "geojson_paths": [],
            "table_names": table_names,
            "errors": [f"Text2SQL 结构化解析失败：{str(e)}"],
            "raw_llm_output_preview": raw_preview,
        }, ensure_ascii=False)

    print("  [Text2SQL] 正在执行 SQL 并导出结果…", flush=True)
    out_dir = str(config.WORKSPACE_DIR)
    sql_results = []
    geojson_paths = []
    queries_info = []

    binding_error = validate_sql_semantic_bindings(
        "\n".join(str(item.sql or "") for item in gen.queries),
        semantic_bindings,
    )
    if binding_error:
        print(f"  [Text2SQL] {binding_error}", flush=True)
        return json.dumps({
            "queries": [
                {
                    "sql": str(item.sql or ""),
                    "output_filename": item.output_filename,
                    "has_geometry": item.has_geometry,
                }
                for item in gen.queries
            ],
            "geojson_paths": [],
            "table_names": table_names,
            "sql_results": [],
            "errors": [binding_error],
            "failure_code": "scga_schema_binding_violation",
            "success": False,
        }, ensure_ascii=False)

    spatial_source_error = validate_sql_spatial_source_coverage(
        "\n".join(str(item.sql or "") for item in gen.queries),
        required_geometry_tables,
    )
    if spatial_source_error:
        print(f"  [Text2SQL] {spatial_source_error}", flush=True)
        return json.dumps({
            "queries": [
                {
                    "sql": str(item.sql or ""),
                    "output_filename": item.output_filename,
                    "has_geometry": item.has_geometry,
                }
                for item in gen.queries
            ],
            "geojson_paths": [],
            "table_names": table_names,
            "sql_results": [],
            "errors": [spatial_source_error],
            "failure_code": "scga_spatial_source_coverage_violation",
            "success": False,
        }, ensure_ascii=False)

    condition_error = validate_sql_condition_coverage(
        "\n".join(str(item.sql or "") for item in gen.queries),
        semantic_bindings,
        condition_clauses,
    )
    if condition_error:
        print(f"  [Text2SQL] {condition_error}", flush=True)
        return json.dumps({
            "queries": [
                {
                    "sql": str(item.sql or ""),
                    "output_filename": item.output_filename,
                    "has_geometry": item.has_geometry,
                }
                for item in gen.queries
            ],
            "geojson_paths": [],
            "table_names": table_names,
            "sql_results": [],
            "errors": [condition_error],
            "failure_code": "scga_condition_coverage_violation",
            "success": False,
        }, ensure_ascii=False)

    for qi, q in enumerate(gen.queries, start=1):
        sql_exec, sql_fix_notes = normalize_sql_before_execution(
            q.sql,
            analysis_export_for_python=analysis_export_for_python,
        )
        if sql_fix_notes:
            print(
                f"  [Text2SQL] 第 {qi} 条 SQL 程序级修正：{'；'.join(sql_fix_notes)}",
                flush=True,
            )

        ranking_projection_error = validate_ordered_sql_projection(
            sql_exec,
            ordered_result_required=ordered_result_required,
            requested_top_k=requested_top_k,
            analysis_export_for_python=analysis_export_for_python,
        )
        if ranking_projection_error:
            print(f"  [Text2SQL] {ranking_projection_error}", flush=True)
            sql_results.append({
                "success": False,
                "path": "",
                "row_count": 0,
                "has_geometry": q.has_geometry,
                "error": ranking_projection_error,
            })
            queries_info.append({
                "sql": sql_exec,
                "output_filename": q.output_filename,
                "has_geometry": q.has_geometry,
            })
            continue

        forbidden_fn = _contains_forbidden_spatial_function(sql_exec)
        if forbidden_fn:
            print(
                f"  [Text2SQL] 警告：第 {qi} 条 SQL 因包含禁用空间函数或运算符 {forbidden_fn} 被拦截，未执行。",
                flush=True,
            )
            sql_results.append({
                "success": False,
                "path": "",
                "row_count": 0,
                "has_geometry": q.has_geometry,
                "error": (
                    f"SQL 包含被禁止的空间函数或运算符 {forbidden_fn}。"
                    "Text2SQL 仅允许基础取数（SELECT 属性列和原始 geometry 列）；"
                    "空间拓扑判断（如 ST_Within/ST_Intersects/&&）和缓冲区/距离/质心等计算必须交给 python_analysis_tool。"
                    "请重写 SQL：仅提取几何列和属性列，将拓扑运算留给 Python 沙盒。"
                ),
            })
            queries_info.append({"sql": sql_exec, "output_filename": q.output_filename, "has_geometry": q.has_geometry})
            continue

        print(f"  [Text2SQL] 执行 SQL (导出到 {q.output_filename}):\n{sql_exec}\n", flush=True)
        r = execute_sql_and_save_geojson.invoke({
            "sql": sql_exec,
            "output_filename": q.output_filename,
            "geom_col": "geometry",
            "out_dir": out_dir,
        })
        if r.get("success") and r.get("path"):
            inferred_file_type = _infer_sql_result_file_type(sql_exec, r["path"], r.get("has_geometry"))
            r["file_type"] = inferred_file_type
        if r.get("success") and r.get("path") and isinstance(r.get("row_count"), int) and 0 < r.get("row_count") <= DATA_PEEK_MAX_ROWS:
            try:
                peek = _build_sql_result_preview(
                    r["path"],
                    r.get("file_type", "json"),
                    max_rows=int(r["row_count"]),
                )
                if isinstance(peek, list) and peek and isinstance(peek[0], dict):
                    peek_str = _format_preview_as_markdown(peek)
                else:
                    peek_str = json.dumps(peek, ensure_ascii=False)
                # Keep the database result as typed JSON for deterministic
                # hand-off to the Execution State Manager.  The legacy
                # Markdown preview remains internal and is only serialized as
                # a fallback when a typed payload cannot be produced.
                r["data_payload"] = peek
                r["data_peek"] = (
                    peek_str[:DATA_PEEK_MAX_CHARS] + "... (已截断)"
                    if len(peek_str) > DATA_PEEK_MAX_CHARS
                    else peek_str
                )
            except Exception as e:
                r["data_peek"] = f"预览失败: {str(e)}"
        sql_results.append(r)
        queries_info.append({"sql": sql_exec, "output_filename": q.output_filename, "has_geometry": q.has_geometry})
        if r.get("success") and r.get("path"):
            geojson_paths.append(r["path"])

    failed_errors = [r.get("error", "未知错误") for r in sql_results if not r.get("success")]

    def _one_sql_result_payload(r: dict) -> dict:
        # Typed result goes first so message-window truncation cannot discard
        # the database evidence needed for deterministic finalization.
        out: dict[str, Any] = {}
        typed_payload = r.get("data_payload")
        if typed_payload is not None:
            out["data_payload"] = typed_payload
        else:
            dp = r.get("data_peek")
            if dp is not None:
                out["data_peek"] = dp
        out["success"] = r.get("success")
        out["row_count"] = r.get("row_count")
        out["path"] = r.get("path")
        out["file_type"] = r.get("file_type")
        out["has_geometry"] = r.get("has_geometry")
        if r.get("error"):
            out["error"] = r.get("error")
        return out

    sql_results_payload = [_one_sql_result_payload(r) for r in sql_results]
    # 键顺序：先 answer_hint（若有），再 sql_results，便于 ToolMessage 截断时仍保留预览
    payload_root: dict[str, Any] = {}
    if any(r.get("data_payload") is not None or r.get("data_peek") for r in sql_results):
        payload_root["answer_hint"] = (
            "【必读】sql_results[].data_payload 为与数据库一致的结构化查询结果（排序与 SQL 一致）。"
            "请仅用其中的州名/地区名与数值撰写最终回答，禁止使用常识或训练记忆替代。"
        )
    payload_root["sql_results"] = sql_results_payload
    payload_root["queries"] = queries_info
    payload_root["geojson_paths"] = geojson_paths
    payload_root["table_names"] = table_names
    payload_root["errors"] = failed_errors
    return json.dumps(payload_root, ensure_ascii=False, indent=2)


def _bare_open_call_spans(code: str) -> list[tuple[int, int]]:
    """定位非属性调用的 ``open(`` 调用整体区间 [start, end)，供区分读文件与落盘写入。"""
    s = code or ""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(?<!\.)open\s*\(", s, re.IGNORECASE):
        start = m.start()
        j = m.end()
        depth = 1
        while j < len(s) and depth:
            ch = s[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        spans.append((start, j))
    return spans


def _open_call_looks_like_write_mode(call: str) -> bool:
    """允许 ``open(..., "w"|"wb"|"a"|...)`` 等工作区落盘；禁止无写入模式的 open（默认可读）。"""
    c = (call or "").lower()
    if re.search(r"\bmode\s*=\s*['\"]w", c):
        return True
    for pat in (
        r",\s*['\"]w['\"]",
        r",\s*['\"]w\+['\"]",
        r",\s*['\"]wb['\"]",
        r",\s*['\"]a['\"]",
        r",\s*['\"]a\+['\"]",
        r",\s*['\"]ab['\"]",
        r",\s*['\"]x['\"]",
        r",\s*['\"]xb['\"]",
    ):
        if re.search(pat, c):
            return True
    return False


def generated_python_code_uses_disallowed_open(code: str) -> bool:
    """禁止用 ``open()`` 读数据；允许带写入模式的 ``open(..., 'w'/...)`` 等工作区落盘（与 LLM2Code 契约一致）。"""
    for a, b in _bare_open_call_spans(code):
        if not _open_call_looks_like_write_mode(code[a:b]):
            return True
    return False


_ASSIGN_PATH_LITERAL = re.compile(
    r"^(\w+)\s*=\s*[\"']([^\"']+\.(?:geojson|json))[\"']\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _collect_path_assignments_for_reads(code: str) -> dict[str, str]:
    """变量名小写 -> 数据文件 basename；含一层 var=var 链式展开。"""
    assign: dict[str, str] = {}
    for m in _ASSIGN_PATH_LITERAL.finditer(code or ""):
        k = m.group(1).strip().lower()
        bn = Path(m.group(2).replace("\\", "/")).name
        if k and bn:
            assign[k] = bn
    changed = True
    while changed:
        changed = False
        for m in re.finditer(r"^(\w+)\s*=\s*(\w+)\s*$", code or "", re.MULTILINE):
            d, s = m.group(1).strip().lower(), m.group(2).strip().lower()
            if d in assign or s not in assign:
                continue
            assign[d] = assign[s]
            changed = True
    return assign


_READ_API_ARG_PATTERNS = (
    re.compile(r"(?:geopandas|gpd)\.read_file\s*\(\s*([^,)]+)", re.IGNORECASE),
    re.compile(r"(?:pandas|pd)\.read_json\s*\(\s*([^,)]+)", re.IGNORECASE),
)


_DATA_PATH_LITERAL = re.compile(
    r"(?P<quote>['\"])(?P<path>[^'\"\n]+\.(?:geojson|json))(?P=quote)",
    re.IGNORECASE,
)


def _canonicalize_authorized_input_literals(
    code: str,
    allowed_basenames: set[str],
) -> tuple[str, list[str]]:
    """Rewrite authorized input literals to sandbox-local basenames.

    The sandbox already runs with ``cwd=config.WORKSPACE_DIR``.  LLM-generated
    prefixes such as ``/workspace/`` or ``workspace/`` therefore point at a
    different/nonexistent directory on Windows.  Canonicalizing only paths
    whose basename is already authorized preserves the file whitelist.
    """
    rewrites: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw_path = match.group("path")
        basename = Path(raw_path.replace("\\", "/")).name
        if basename not in allowed_basenames or raw_path == basename:
            return match.group(0)
        rewrites.append(f"{raw_path}->{basename}")
        quote = match.group("quote")
        return f"{quote}{basename}{quote}"

    return _DATA_PATH_LITERAL.sub(replace, str(code or "")), list(dict.fromkeys(rewrites))


def _read_api_unauthorized_basenames(code: str, allowed_basenames: set[str]) -> list[str]:
    """
    检查 gpd.read_file / pd.read_json 的首参是否解析到白名单外的 .geojson/.json。
    复杂表达式（f-string、拼接）无法静态解析时不报。
    open() 读路径由 generated_python_code_uses_disallowed_open 单独约束；写入模式 open 不在此列。
    """
    if not (code or "").strip() or not allowed_basenames:
        return []
    assign = _collect_path_assignments_for_reads(code)
    bad: list[str] = []
    seen: set[str] = set()
    for pat in _READ_API_ARG_PATTERNS:
        for m in pat.finditer(code):
            arg = (m.group(1) or "").strip()
            if not arg or re.search(r"[+]|f[\"']|%\(|\.format\s*\(", arg):
                continue
            lit = re.match(r'^[\"\']([^\"\'\n]+\.(?:geojson|json))[\"\']', arg, re.IGNORECASE)
            if lit:
                bn = Path(lit.group(1).replace("\\", "/")).name
            elif re.match(r"^\w+$", arg):
                bn = assign.get(arg.lower())
                if not bn:
                    continue
            else:
                continue
            if bn and bn not in allowed_basenames and bn not in seen:
                seen.add(bn)
                bad.append(bn)
    return bad


def _noncanonical_read_api_args(code: str, allowed_basenames: set[str]) -> list[str]:
    """Return read_file/read_json arguments that cannot be proven workspace-local."""
    assignments = _collect_path_assignments_for_reads(code)
    bad: list[str] = []
    for pattern in _READ_API_ARG_PATTERNS:
        for match in pattern.finditer(code or ""):
            arg = str(match.group(1) or "").strip()
            literal = re.match(r'^["\']([^"\'\n]+\.(?:geojson|json))["\']$', arg, re.IGNORECASE)
            if literal:
                raw_path = literal.group(1)
                basename = Path(raw_path.replace("\\", "/")).name
                if raw_path == basename and basename in allowed_basenames:
                    continue
            elif re.fullmatch(r"\w+", arg):
                basename = assignments.get(arg.lower())
                if basename in allowed_basenames:
                    continue
            bad.append(arg[:160] or "（空参数）")
    return list(dict.fromkeys(bad))


def _repair_single_input_read_path(
    code: str,
    *,
    allowed_basenames: set[str],
) -> tuple[str, str | None]:
    """单输入任务中，将 LLM 自造的读取文件名安全映射到唯一授权输入。

    多输入任务无法无歧义判断映射关系，仍由白名单门禁拒绝并交给模型修复。
    """
    if len(allowed_basenames) != 1:
        return code, None
    rogue = _read_api_unauthorized_basenames(code, allowed_basenames)
    if len(rogue) != 1:
        return code, None
    allowed = next(iter(allowed_basenames))
    bad = rogue[0]
    repaired = str(code or "").replace(f'"{bad}"', f'"{allowed}"').replace(
        f"'{bad}'", f"'{allowed}'"
    )
    if repaired == code or _read_api_unauthorized_basenames(repaired, allowed_basenames):
        return code, None
    return repaired, f"已将唯一输入文件引用 {bad} 映射为授权文件 {allowed}"


def _build_llm2code_info_summary(
    path_names: list[str],
    normalized_sql_queries: list[dict],
    current_plan_step: str,
    *,
    generated_code_chars: int | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """供工具返回与日志使用的 llm2code 运行摘要（不含完整脚本）。"""
    step = (current_plan_step or "").strip()
    info: dict[str, Any] = {
        "workspace_files": [
            {"path": p, "file_type": _infer_file_type_from_path(p)} for p in path_names
        ],
        "sql_context_entries": len(normalized_sql_queries),
        "current_plan_step_preview": step[:280] + ("…" if len(step) > 280 else ""),
        "current_plan_step_chars": len(step),
    }
    if generated_code_chars is not None:
        info["generated_code_chars"] = generated_code_chars
    if stage:
        info["stage"] = stage
    return info


def _step_explicitly_requests_saved_artifact(step_text: str) -> bool:
    """Whether the blueprint itself asks for a deliverable/detail artifact."""
    text = str(step_text or "")
    explicit = (
        "保存为",
        "保存至",
        "落盘",
        "导出结果",
        "输出明细",
        "完整列表",
        "生成文件",
        "可视化",
        "渲染地图",
        "生成地图",
    )
    if any(token in text for token in explicit):
        return True
    return bool(
        re.search(r"(?:输出|列出|返回).{0,12}(?:每个|所有|全部|完整)", text)
        or re.search(r"(?:每个|所有|全部).{0,20}(?:输出|列出|返回)", text)
    )


def execute_python_analysis_logic(
    question: str,
    error_trace: Optional[str] = None,
    slots_json: Optional[str] = None,
    current_plan_step: Optional[str] = None,
    analysis_contract: Optional[dict[str, Any]] = None,
    geojson_paths: Optional[list[str]] = None,
    sql_queries: Optional[list[dict]] = None,
    execution_contract: Optional[dict[str, Any]] = None,
) -> str:
    """对已导出的空间或属性数据执行 Python 分析；向控制台与返回 JSON 写入 llm2code 诊断摘要。"""
    # 归一化数据文件路径
    path_names = _normalize_workspace_paths(geojson_paths)
    if not path_names:
        print("  [LLM2Code] 中止：没有可用的数据文件路径", flush=True)
        return json.dumps({
            "success": False,
            "error": "没有可用的数据文件路径",
            "llm2code_info": {"stage": "no_paths_after_normalize", "raw_path_count": len(geojson_paths or [])},
        }, ensure_ascii=False)
    # 解析 slots 信息
    slots = _parse_slots_json(slots_json)
    # 规范化当前步骤文本：有 plan 时应始终优先使用明确子任务，避免 llm2code 退化为仅看全局问题。
    step_text = str(current_plan_step or "").strip()
    if not step_text:
        step_text = "请根据工具参数中要求的分析目标（question），读取并处理当前工作区的数据文件，直接计算并输出最终的统计指标与结论。"
    blob_q = f"{step_text} {question or ''}".strip()
    require_geojson = infer_require_geojson_for_python(slots, question, current_plan_step=step_text)
    ec = execution_contract if isinstance(execution_contract, dict) else {}
    if int(ec.get("contract_version") or 0) == 1:
        require_geojson = bool(ec.get("requires_geometry"))
        print(
            f"  [LLM2Code] execution_contract v1：requires_geometry={require_geojson}（覆盖启发式推断）",
            flush=True,
        )
    valid_inputs, validate_error, val_code = _validate_python_analysis_input_files(
        path_names, require_geojson=require_geojson
    )
    if (
        not valid_inputs
        and val_code == "requires_geometry"
        and _blob_attribute_stats_primary(blob_q, slots)
        and not _blob_requires_geometry_semantics(blob_q, slots)
    ):
        require_geojson = False
        valid_inputs, validate_error, val_code = _validate_python_analysis_input_files(
            path_names, require_geojson=require_geojson
        )
    if not valid_inputs:
        print(f"  [LLM2Code] 中止：输入文件校验失败 — {validate_error}", flush=True)
        if val_code == "requires_geometry":
            fc = FAILURE_CODE_PYTHON_REQUIRES_GEOMETRY_EXPORT
        elif val_code == "unreadable_json":
            fc = FAILURE_CODE_PYTHON_INPUT_UNREADABLE
        else:
            fc = "python_input_validate_failed"
        info = _build_llm2code_info_summary(path_names, [], step_text, stage="validate_inputs_failed")
        info["require_geojson"] = require_geojson
        info["validate_error_code"] = val_code
        return json.dumps(
            {
                "success": False,
                "error": validate_error,
                "failure_code": fc,
                "llm2code_info": info,
            },
            ensure_ascii=False,
        )

    # 加载代码模板（如有）
    code_templates = load_code_templates(slots)
    normalized_sql_queries: list[dict] = []
    sql_results_map: dict[str, dict] = {}
    # 规范化 SQL 查询结果（方便文件名索引）
    for item in (sql_queries or []):
        if isinstance(item, dict):
            normalized_item = dict(item)
            normalized_sql_queries.append(normalized_item)
            output_filename = str(normalized_item.get("output_filename") or "").strip()
            if output_filename:
                sql_results_map[output_filename] = normalized_item

    # 为每一个真实存在的路径补全映射关系
    for raw_path in path_names:
        file_name = Path(raw_path).name
        if file_name and file_name not in sql_results_map:
            sql_results_map[file_name] = {
                "output_filename": file_name,
                "normalized_path": raw_path,
                "file_type": _infer_file_type_from_path(raw_path),
            }

    # 把路径补全到 normalized_sql_queries 条目
    for item in normalized_sql_queries:
        output_filename = str(item.get("output_filename") or "").strip()
        if not output_filename:
            continue
        matched_path = _match_path_by_filename(output_filename, path_names)
        if matched_path:
            item["normalized_path"] = matched_path
            item["file_type"] = item.get("file_type") or _infer_file_type_from_path(matched_path)

    # 组装合约等上下文信息
    contract = dict(analysis_contract or {})
    if ec:
        contract["execution_contract"] = {
            "requires_geometry": bool(ec.get("requires_geometry")),
            "operation_type": ec.get("operation_type"),
            "entity_level": ec.get("entity_level"),
            "time_comparison_mode": ec.get("time_comparison_mode"),
            "answer_projection": ec.get("answer_projection") or {},
            "schema_bindings": ec.get("schema_bindings") or [],
            "condition_clauses": ec.get("condition_clauses") or [],
        }
    base_info = _build_llm2code_info_summary(path_names, normalized_sql_queries, step_text)
    base_info["question_preview"] = (question or "")[:200] + ("…" if len(question or "") > 200 else "")
    base_info["has_error_trace"] = bool(error_trace and str(error_trace).strip())
    base_info["code_templates_loaded"] = len(code_templates or [])
    base_info["workspace_file_types"] = [
        {"path": p, "file_type": _infer_file_type_from_path(p)} for p in path_names
    ]
    print(
        f"  [LLM2Code] 输入：工作区文件 {len(path_names)} 个，SQL 上下文条目 {len(normalized_sql_queries)}，"
        f"蓝图当前步 {base_info['current_plan_step_chars']} 字，重试上下文={'有' if base_info['has_error_trace'] else '无'}",
        flush=True,
    )
    for i, p in enumerate(path_names, 1):
        print(f"    [{i}] {p} (file_type={_infer_file_type_from_path(p)})", flush=True)
    step_full = step_text
    if step_full:
        print(f"  [LLM2Code] 当前蓝图步骤（全文 {len(step_full)} 字）:\n{step_full}", flush=True)
    print("  [LLM2Code] 调用 LLM 生成 Python…", flush=True)

    system_msg, user_msg = get_llm2code_messages(
        question,
        json.dumps(slots, ensure_ascii=False),
        step_text,
        json.dumps(contract, ensure_ascii=False, indent=2),
        path_names,
        sql_queries=normalized_sql_queries,
        code_templates=code_templates if code_templates else None,
        error_trace=error_trace,
    )
    llm = config.get_llm()
    msg = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    content = msg.content if hasattr(msg, "content") else str(msg)
    code = extract_python_code(content)

    if generated_python_code_uses_disallowed_open(code):
        print("  [LLM2Code] 中止：生成代码含禁止的 open() 读文件调用", flush=True)
        err_info = _build_llm2code_info_summary(
            path_names,
            normalized_sql_queries,
            step_text,
            generated_code_chars=len(code),
            stage="disallowed_open_call",
        )
        return json.dumps(
            {
                "success": False,
                "error": (
                    "生成的代码不得使用无写入模式的 open() 读取数据文件；请改用 geopandas.read_file() 读取 .geojson，"
                    "或用 pandas.read_json() 读取 .json。落盘结果可使用 open(..., 'w', encoding='utf-8') 等写入模式。"
                ),
                "code": code,
                "llm2code_info": err_info,
            },
            ensure_ascii=False,
        )

    allowed_basenames = {Path(p).name for p in path_names if p}
    code, canonical_path_rewrites = _canonicalize_authorized_input_literals(
        code,
        allowed_basenames,
    )
    if canonical_path_rewrites:
        print(
            "  [LLM2Code] 程序级路径规范化：" + "；".join(canonical_path_rewrites),
            flush=True,
        )
    code, path_repair_note = _repair_single_input_read_path(
        code,
        allowed_basenames=allowed_basenames,
    )
    if path_repair_note:
        print(f"  [LLM2Code] 程序级修正：{path_repair_note}", flush=True)
    rogue_files = list(dict.fromkeys(_read_api_unauthorized_basenames(code, allowed_basenames)))
    if rogue_files:
        detail = "、".join(rogue_files)
        print(f"  [LLM2Code] 中止：代码试图读取未授权的文件名: {detail}", flush=True)
        err_info = _build_llm2code_info_summary(
            path_names,
            normalized_sql_queries,
            step_text,
            generated_code_chars=len(code),
            stage="unauthorized_read_api",
        )
        err_info["unauthorized_read_api"] = rogue_files
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"生成的代码使用 read_file/read_json 试图读取未在工作区白名单内的数据文件名（{detail}），禁止执行。"
                    "请仅读取给定文件列表中的纯文件名。落盘保存文件不受限制。"
                ),
                "code": code,
                "llm2code_info": err_info,
            },
            ensure_ascii=False,
        )

    noncanonical_reads = _noncanonical_read_api_args(code, allowed_basenames)
    if noncanonical_reads:
        detail = "；".join(noncanonical_reads)
        print(f"  [LLM2Code] 中止：输入读取路径不是可证明的工作区 basename: {detail}", flush=True)
        err_info = _build_llm2code_info_summary(
            path_names,
            normalized_sql_queries,
            step_text,
            generated_code_chars=len(code),
            stage="noncanonical_read_path",
        )
        err_info["noncanonical_read_args"] = noncanonical_reads
        return json.dumps(
            {
                "success": False,
                "error": (
                    "read_file/read_json 的输入必须直接使用授权列表中的纯文件名，或使用直接赋值为该纯文件名的简单变量；"
                    f"禁止目录拼接、工作区猜测或动态路径表达式：{detail}"
                ),
                "code": code,
                "llm2code_info": err_info,
            },
            ensure_ascii=False,
        )

    if not code.strip():
        print("  [LLM2Code] 中止：LLM 未产出可解析的 Python 代码", flush=True)
        err_info = dict(base_info)
        err_info["stage"] = "llm_no_code"
        return json.dumps({
            "success": False,
            "error": "LLM 未生成可执行 Python 代码",
            "code": code,
            "llm2code_info": err_info,
        }, ensure_ascii=False)

    print(f"  [LLM2Code] 已生成代码，共 {len(code)} 字符", flush=True)
    print("  [LLM2Code] 执行沙盒…", flush=True)

    try:
        # 运行沙盒，执行 LLM 生成的代码
        result = execute_python_sandbox.invoke({
            "code": code,
            "timeout": config.SANDBOX_TIMEOUT,
            "strict_json_output": True,
            "authorized_input_files": path_names,
        })
    except SandboxOutputParseError as e:
        err_info = _build_llm2code_info_summary(
            path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="sandbox_parse_error"
        )
        print(f"  [LLM2Code] 沙盒输出解析失败：{type(e).__name__}", flush=True)
        return json.dumps({
            "success": False,
            "error": str(e) + (f"\n原始 stdout: {e.raw_stdout[:500]}" if getattr(e, "raw_stdout", "") else ""),
            "code": code,
            "llm2code_info": err_info,
        }, ensure_ascii=False)

    # 检查沙盒返回结果类型
    if not isinstance(result, dict):
        err_info = _build_llm2code_info_summary(
            path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="sandbox_bad_result_type"
        )
        print(f"  [LLM2Code] 沙盒返回类型异常：{type(result).__name__}", flush=True)
        return json.dumps({
            "success": False,
            "error": f"沙盒返回类型异常：期望 dict，实际为 {type(result).__name__}",
            "code": code,
            "llm2code_info": err_info,
        }, ensure_ascii=False)

    # 解析沙盒输出内容和状态
    success = bool(result.get("success", False))
    stdout = str(result.get("stdout", "") or "")
    stderr = str(result.get("stderr", "") or "")
    parsed = result.get("parsed")

    if not success:
        err_info = _build_llm2code_info_summary(
            path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="sandbox_process_failed"
        )
        print(f"  [LLM2Code] 沙盒进程失败：{stderr[:300]!r}", flush=True)
        return json.dumps({
            "success": False,
            "error": stderr or "执行失败",
            "code": code,
            "llm2code_info": err_info,
        }, ensure_ascii=False)

    # 如果解析出的内容为字典且约定信息齐全
    if isinstance(parsed, dict):
        script_status = parsed.get("status", "ok")
        answer_text = parsed.get("answer_text", "")
        data_payload = parsed.get("data_payload", {})
        if script_status != "ok":
            err_info = _build_llm2code_info_summary(
                path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="script_status_error"
            )
            print(f"  [LLM2Code] 脚本返回 status={script_status!r}", flush=True)
            return json.dumps({
                "success": False,
                "error": answer_text or stderr or "脚本返回 status=error",
                "code": code,
                "llm2code_info": err_info,
            }, ensure_ascii=False)
        save_code_to_memory(code, slots)
        code_output_str = str(answer_text or "")
        if data_payload:
            code_output_str += "\n\n" + json.dumps(data_payload, ensure_ascii=False, indent=2)
        raw_saved = parsed.get("saved_files")
        saved_files: list[str] = []
        if isinstance(raw_saved, list):
            saved_files = [str(x).strip() for x in raw_saved if str(x).strip()]
        elif isinstance(raw_saved, str) and raw_saved.strip():
            saved_files = [raw_saved.strip()]
        if saved_files and not _step_explicitly_requests_saved_artifact(
            f"{step_text} {question or ''}"
        ):
            # A generated helper CSV must not replace a scalar/aggregate answer
            # in downstream evaluation. The file may remain in the isolated
            # workspace, but it is not part of the semantic tool result unless
            # the blueprint explicitly requested an artifact.
            print(
                "  [LLM2Code:v2] 当前蓝图未要求明细/落盘，忽略辅助 saved_files="
                f"{saved_files}",
                flush=True,
            )
            saved_files = []
        ok_info = _build_llm2code_info_summary(
            path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="ok"
        )
        ok_info["answer_text_preview"] = str(answer_text or "")[:240] + ("…" if len(str(answer_text or "")) > 240 else "")
        ok_info["data_payload_keys"] = list(data_payload.keys()) if isinstance(data_payload, dict) else []
        ok_info["saved_files"] = saved_files
        print(
            f"  [LLM2Code] 完成：answer_text {len(str(answer_text or ''))} 字，"
            f"data_payload 键 {ok_info['data_payload_keys']}，saved_files {saved_files}",
            flush=True,
        )
        out_ok: dict[str, Any] = {
            "success": True,
            "code_output": code_output_str,
            # 保留沙盒真实结构化结果，供结案节点直接封装；不再依赖从自然语言中反解析。
            "answer_text": answer_text,
            "data_payload": data_payload,
            "code": code,
            "llm2code_info": ok_info,
        }
        if saved_files:
            out_ok["saved_files"] = saved_files
        return json.dumps(out_ok, ensure_ascii=False)

    parse_hint = "Python 脚本执行成功，但未按协议输出包含 status, answer_text, data_payload 的标准 JSON。"
    err_info = _build_llm2code_info_summary(
        path_names, normalized_sql_queries, step_text, generated_code_chars=len(code), stage="protocol_parse_mismatch"
    )
    print("  [LLM2Code] 脚本 stdout 未解析为协议 JSON", flush=True)
    return json.dumps({
        "success": False,
        "error": parse_hint + (f"\nstdout: {stdout[:500]}" if stdout else ""),
        "code": code,
        "llm2code_info": err_info,
    }, ensure_ascii=False)


@tool(args_schema=MapRenderingArgs)
def map_rendering_tool(geojson_path: str) -> str:
    """将 GeoJSON 文件渲染为 Folium 交互式 HTML 地图。"""
    if not str(geojson_path).lower().endswith(".geojson"):
        return "跳过：该文件不是 .geojson 格式，为纯属性数据，无需地图渲染。"
    try:
        map_file_path = render_spatial_map.invoke({
            "geojson_path": geojson_path,
            "output_html": "agent_map.html",
            "out_dir": str(config.WORKSPACE_DIR),
        })
        if map_file_path and str(map_file_path).strip().startswith("错误"):
            return f"地图渲染警告：{map_file_path}"
        return f"交互式地图已生成：{map_file_path}"
    except Exception as e:
        return f"地图渲染失败：{e}"


@tool("text2sql_tool", args_schema=Text2SQLArgs)
def text2sql_tool_schema(**kwargs):
    """从业务库提取空间/属性数据。你只需用自然语言在 `question` 中描述需要什么数据；系统底层的 DBA/Text2SQL 模块会自动编写 SQL、在 PostGIS 中执行，并导出 GeoJSON/JSON 到工作区。不要在对话里手写 SQL 代替本工具。"""
    return json.dumps({
        "success": False,
        "queries": [],
        "geojson_paths": [],
        "errors": ["路由泄漏：text2sql_tool 应在 LangGraph 的 text2sql 节点执行，不应在通用 ToolNode 中调用。请检查 route_after_agent 与边配置。"],
    }, ensure_ascii=False)


@tool("python_analysis_tool", args_schema=PythonAnalysisArgs)
def python_analysis_tool_schema(**kwargs):
    """对已导出的 GeoJSON/JSON 做统计、空间拓扑与高级分析。你不需要、也不应在聊天框里写 Python：只需在 `question` 中用自然语言说明分析目标，在 `geojson_paths` 中传入工作区内的纯文件名列表；底层沙盒会自动生成并执行代码，并把结果写回 ToolMessage。"""
    return json.dumps({
        "success": False,
        "code_output": "",
        "error": "路由泄漏：python_analysis_tool 应由 python_analysis 节点执行，不应在通用 ToolNode 中直接调用。请检查图路由。",
    }, ensure_ascii=False)

# ALL_TOOLS / TOOLS_MAP 仅从 agent.tooling.registry 导入（若在 tools 末尾再导出会与 registry 形成循环导入）
