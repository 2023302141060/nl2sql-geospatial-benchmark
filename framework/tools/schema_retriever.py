# -*- coding: utf-8 -*-
"""Schema RAG 检索模块：基于 Chroma 向量数据库实现 Top-K Schema 检索与外键关系补全。"""
import gc
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from collections import deque
from typing import Any


import yaml
from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
import config
from utils.schema_utils import clear_schema_cache, load_schemas_by_table_names

_CHROMA_DIR = config.WORKSPACE_DIR / "chroma_db"
_COLLECTION_NAME = "schema_docs"
_HASH_FILE = _CHROMA_DIR / ".yaml_hash"
_EMBEDDING_SIG_FILE = _CHROMA_DIR / ".embedding_sig"

_vectorstore: Chroma | None = None
_table_name_to_yaml: dict[str, str] = {}
_vectorstore_lock = threading.Lock()
logger = logging.getLogger(__name__)


def release_schema_vectorstore() -> None:
    """丢弃缓存的 LangChain Chroma 实例并关闭底层 PersistentClient，释放 chroma.sqlite3 句柄。

    在重建向量库或外部删除 chroma_db 前调用，避免 Windows 上文件占用。
    """
    global _vectorstore
    with _vectorstore_lock:
        vs = _vectorstore
        _vectorstore = None
    client = getattr(vs, "_client", None)
    try:
        if client is not None and hasattr(client, "close"):
            client.close()
    except Exception:
        pass
    systems = []
    if client is not None:
        system = getattr(client, "_system", None)
        if system is not None:
            systems.append(system)
    try:
        systems.extend(list(getattr(SharedSystemClient, "_identifier_to_system", {}).values()))
    except Exception:
        pass
    seen_ids: set[int] = set()
    for system in systems:
        sid = id(system)
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        try:
            if hasattr(system, "stop"):
                system.stop()
        except Exception:
            pass
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    try:
        del vs
    except Exception:
        pass
    try:
        del client
    except Exception:
        pass
    gc.collect()


def _normalize_schema_dict(data: dict) -> dict:
    """
    将单表 schema dict 规范化为“表级信息在前、columns 在后”的顺序，
    以降低 Text2SQL 多表场景下的字段归属幻觉。
    """
    if not isinstance(data, dict):
        return data

    ordered: dict = {}
    for key in [
        "table_name",
        "table_description",
        "domain",
        "semantic_aliases",
        "spatiotemporal_properties",
    ]:
        if key in data:
            ordered[key] = data.get(key)
    # 兼容一些 schema 文件可能存在的替代键名
    for key in ["spatial_granularity", "temporal_granularity", "primary_key", "foreign_keys"]:
        if key in data and key not in ordered:
            ordered[key] = data.get(key)

    ordered["columns"] = data.get("columns") or []

    for key, value in data.items():
        if key in ordered:
            continue
        ordered[key] = value
    return ordered


def _compute_schemas_hash() -> str:
    """对所有 YAML 文件内容取摘要，用于判断是否需要重新索引。"""
    h = hashlib.sha256()
    for p in sorted(config.SCHEMAS_DIR.glob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()


def _embedding_signature(embeddings) -> str:
    """当前向量模型 + 输出维度（一次 probe），用于与持久化库一致；切换模型/维度后自动重建 Chroma。"""
    vec = embeddings.embed_query("__chroma_embedding_dim_probe__")
    return f"{config.EMBEDDING_MODEL}:{len(vec)}"


def _stored_signature_uses_current_model(stored_sig: str) -> bool:
    """同名模型复用已验证维度，避免每次加载 Chroma 都发起额外远程 probe。"""
    stored_model, separator, stored_dimension = str(stored_sig or "").partition(":")
    return bool(
        separator
        and stored_model.strip() == str(config.EMBEDDING_MODEL).strip()
        and stored_dimension.strip().isdigit()
    )


def _build_table_yaml_map() -> dict[str, str]:
    """构建 table_name -> YAML 字符串 的全局映射，供外键补全时快速查找。"""
    mapping: dict[str, str] = {}
    for p in config.SCHEMAS_DIR.glob("*.yaml"):
        try:
            raw = p.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
            if data and data.get("table_name"):
                data = _normalize_schema_dict(data)
                yaml_str = yaml.dump(
                    data,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
                mapping[data["table_name"]] = yaml_str
        except Exception:
            continue
    return mapping


def _semantic_compact_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("pm2.5", "pm25").replace("pm2_5", "pm25")
    return re.sub(r"[\s_\-.:/\\()\[\]{}，。；、]+", "", text)


def _schema_map_snapshot() -> dict[str, str]:
    """Return Schema metadata without opening Chroma or issuing an embedding call."""
    if _table_name_to_yaml:
        return dict(_table_name_to_yaml)
    return _build_table_yaml_map()


def _schema_domain(yaml_str: str) -> str:
    """Return the mounted dataset domain declared by one Schema document."""
    try:
        return str((yaml.safe_load(yaml_str) or {}).get("domain") or "").strip()
    except Exception:
        return ""


def _domain_surface_forms(domain: str) -> set[str]:
    """从 Schema 声明的数据域生成保守的常见表面形式。"""
    text = str(domain or "").strip()
    forms = {text} if text else set()
    # 两字国家名常以“全 + 首字”指代全国范围，例如“美国”→“全美”。
    # 只在完整的“全X”短语出现时使用，避免单字误命中。
    if len(text) == 2 and text.endswith("国"):
        forms.add(f"全{text[0]}")
    return forms


def _schema_temporal_granularity(yaml_str: str) -> str:
    try:
        data = yaml.safe_load(yaml_str) or {}
        return str((data.get("spatiotemporal_properties") or {}).get("temporal_granularity") or "").strip()
    except Exception:
        return ""


def _requested_temporal_granularities(question: str) -> set[str]:
    """识别问题明确要求的多个时间粒度，不根据年份数量猜测。"""
    text = str(question or "").casefold()
    requested: set[str] = set()
    if re.search(r"月均|月度|逐月|每月|monthly|per month", text):
        requested.add("monthly")
    if re.search(r"年均|年度|逐年|每年|annual|yearly|per year", text):
        requested.add("annual")
    return requested


def _temporal_granularity_role(granularity: str) -> str:
    text = str(granularity or "").casefold()
    if "月" in text or "month" in text:
        return "monthly"
    if "年" in text or "annual" in text or "year" in text:
        return "annual"
    return "other"


def _as_term_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _description_term_variants(description: str, column_name: str) -> list[str]:
    """Derive conservative lexical variants from Schema column metadata.

    The variants keep the physical statistic semantics: only ``*_mean`` fields
    receive mean/average wording.  This lets phrases such as ``平均气温`` and
    ``白天地表平均温度`` bind to a mean column without making mean/sum/count
    interchangeable.
    """
    text = str(description or "").strip()
    if not text:
        return []
    variants = [text]
    cleaned = re.sub(r"[“”‘’\"']", "", text)
    if cleaned != text:
        variants.append(cleaned)
    for prefix in ("行政区内", "城市", "省级", "州级", "县级", "网格", "空间单元"):
        if text.startswith(prefix) and len(text) - len(prefix) >= 3:
            variants.append(text[len(prefix):])
        if cleaned.startswith(prefix) and len(cleaned) - len(prefix) >= 3:
            variants.append(cleaned[len(prefix):])
    for term in list(variants):
        if term.startswith("存在") and len(term) > 4:
            variants.append(term[2:])
        if "的面积比例" in term:
            variants.append(term.replace("的面积比例", "面积比例"))
        normalized = re.sub(r"^存在", "", term).replace("的面积比例", "面积比例")
        if normalized != term:
            variants.append(normalized)
        # “平均海拔高度”与“平均海拔”是同一物理量的完整/省略表达；仅去掉
        # 描述性后缀，不改变 mean/sum/count 等统计语义。
        if term.endswith("海拔高度"):
            variants.append(term[:-2])
    if str(column_name or "").casefold().endswith("_mean"):
        if text.startswith("年平均"):
            variants.append(text[1:])
        if text.startswith("月平均"):
            variants.append(text[1:])
        base = re.sub(r"(?:年均值|月均值|平均值|均值)$", "", text).strip()
        if base and base != text:
            variants.append(base)
        for term in list(variants):
            if "平均" not in term and "均值" not in term:
                variants.append("平均" + term)
            if "地表温度" in term and "地表平均温度" not in term:
                variants.append(term.replace("地表温度", "地表平均温度"))
    return list(dict.fromkeys(item for item in variants if len(item) >= 2))


def _iter_schema_semantic_entries(
    table_name: str,
    yaml_str: str,
    *,
    ambiguous_base_tokens: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a data-driven semantic catalog from YAML aliases and column metadata."""
    try:
        data = yaml.safe_load(yaml_str) or {}
    except Exception:
        return []
    entries: list[dict[str, Any]] = []
    semantic_aliases = data.get("semantic_aliases") or {}
    if isinstance(semantic_aliases, dict):
        for concept, raw in semantic_aliases.items():
            if isinstance(raw, dict):
                terms = _as_term_list(raw.get("terms") or raw.get("aliases"))
                columns = _as_term_list(raw.get("columns"))
            else:
                terms = _as_term_list(raw)
                columns = []
            if terms:
                entries.append({
                    "concept": str(concept),
                    "terms": terms,
                    "table": table_name,
                    "columns": columns,
                    "source": "semantic_alias",
                })

    columns = [column for column in (data.get("columns") or []) if isinstance(column, dict)]
    base_token_counts: dict[str, int] = {}
    for column in columns:
        column_name = str(column.get("name") or column.get("column_name") or "").strip()
        first_token = re.split(r"_+", column_name.casefold(), maxsplit=1)[0]
        if re.fullmatch(r"[a-z][a-z0-9.]{2,}", first_token or ""):
            base_token_counts[first_token] = base_token_counts.get(first_token, 0) + 1

    for column in columns:
        if not isinstance(column, dict):
            continue
        column_name = str(column.get("name") or column.get("column_name") or "").strip()
        if not column_name:
            continue
        terms = _as_term_list(column.get("aliases"))
        description = str(column.get("description") or "").strip()
        if description and len(description) <= 48:
            terms.extend(_description_term_variants(description, column_name))
        if column_name.casefold().endswith("_mean"):
            for alias in list(terms):
                terms.extend(_description_term_variants(alias, column_name))
        # Only keep the complete physical column name.  Splitting names such as
        # ``lst_night_mean`` into ``lst``/``night`` causes cross-metric leakage:
        # a daytime-LST question would otherwise bind the night-time field merely
        # because both contain the generic token ``lst``.
        terms.append(column_name)
        # A base token is safe only when it identifies exactly one column in the
        # table.  Thus ``ndvi_mean`` may expose ``ndvi``, while the shared ``lst``
        # token of day/night fields is deliberately withheld.
        first_token = re.split(r"_+", column_name.casefold(), maxsplit=1)[0]
        if (
            base_token_counts.get(first_token, 0) == 1
            and first_token not in (ambiguous_base_tokens or set())
        ):
            terms.append(first_token)
        if terms:
            entries.append({
                "concept": column_name,
                "terms": list(dict.fromkeys(terms)),
                "table": table_name,
                "columns": [column_name],
                "source": "column",
            })
    return entries


def _query_years(question: str, slots_dict: dict[str, Any] | None = None) -> list[str]:
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(question or ""))
    slots = slots_dict if isinstance(slots_dict, dict) else {}
    for value in [slots.get("time"), *(slots.get("time_range") or [])]:
        years.extend(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(value or "")))
    return list(dict.fromkeys(years))


def _schema_time_compatibility(
    yaml_str: str,
    *,
    years: list[str],
    monthly_query: bool,
) -> float:
    try:
        data = yaml.safe_load(yaml_str) or {}
    except Exception:
        return 1.0
    st = data.get("spatiotemporal_properties") or {}
    granularity = str(st.get("temporal_granularity") or "").strip()
    if granularity in {"静态", "static"}:
        return 1.0
    available_years: set[str] = set()
    available_years.update(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", str(data.get("table_description") or "")))
    for column in data.get("columns") or []:
        if str(column.get("name") or "").casefold() != "year":
            continue
        available_years.update(str(item) for item in (column.get("enums") or []) if str(item).strip())
    if years and available_years and not set(years).intersection(available_years):
        return 0.0
    if monthly_query:
        return 1.35 if "月" in granularity or "month" in granularity.casefold() else 0.75
    return 1.0


def _semantic_schema_matches(
    question: str,
    slots_dict: dict[str, Any] | None = None,
    table_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Match question concepts to Schema aliases without an LLM call."""
    slots = slots_dict if isinstance(slots_dict, dict) else {}
    normalized_question = _semantic_compact_text(question)
    years = _query_years(question, slots)
    monthly_query = bool(re.search(r"(?:\d{1,2}\s*[、,，和及至到\-]?\s*)+月|月度|逐月|每月|夏季|冬季", str(question or "")))
    target_domain = str(slots.get("region_scope") or "").strip()
    matches: list[dict[str, Any]] = []
    schema_map = table_map or _schema_map_snapshot()
    eligible_tables: list[tuple[str, str, float]] = []
    for table_name, yaml_str in schema_map.items():
        try:
            table_domain = str((yaml.safe_load(yaml_str) or {}).get("domain") or "").strip()
        except Exception:
            table_domain = ""
        if target_domain and table_domain and table_domain != target_domain:
            continue
        compatibility = _schema_time_compatibility(
            yaml_str,
            years=years,
            monthly_query=monthly_query,
        )
        if compatibility <= 0:
            continue
        eligible_tables.append((table_name, yaml_str, compatibility))

    # A shorthand such as ``landcover`` is safe only if it identifies one
    # physical column across the domain/time-compatible candidate Schema, not
    # merely one column inside one table.  This prevents a precise IGBP request
    # from also binding an unrelated ESA field in another table.
    global_base_counts: dict[str, int] = {}
    for _table_name, yaml_str, _compatibility in eligible_tables:
        try:
            columns = (yaml.safe_load(yaml_str) or {}).get("columns") or []
        except Exception:
            columns = []
        for column in columns:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or column.get("column_name") or "").strip()
            first_token = re.split(r"_+", column_name.casefold(), maxsplit=1)[0]
            if re.fullmatch(r"[a-z][a-z0-9.]{2,}", first_token or ""):
                global_base_counts[first_token] = global_base_counts.get(first_token, 0) + 1
    ambiguous_base_tokens = {
        token for token, count in global_base_counts.items() if count > 1
    }

    for table_name, yaml_str, compatibility in eligible_tables:
        for entry in _iter_schema_semantic_entries(
            table_name,
            yaml_str,
            ambiguous_base_tokens=ambiguous_base_tokens,
        ):
            matched_terms: list[str] = []
            matched_spans: list[tuple[int, int]] = []
            for term in entry["terms"]:
                compact_term = _semantic_compact_text(term)
                if len(compact_term) < 2:
                    continue
                start = normalized_question.find(compact_term)
                if start < 0:
                    continue
                matched_terms.append(term)
                while start >= 0:
                    matched_spans.append((start, start + len(compact_term)))
                    start = normalized_question.find(compact_term, start + 1)
            if not matched_terms:
                continue
            matches.append({
                **entry,
                "matched_terms": matched_terms,
                "matched_spans": list(dict.fromkeys(matched_spans)),
                "time_compatibility": compatibility,
            })
    # A precise table-level concept (for example ``cropland_proportion``) owns
    # its declared physical columns.  Suppress a simultaneous generic column
    # hit such as ``proportion``/``占比`` from the same table; otherwise one user
    # metric becomes two execution requirements and SCGA is over-constrained.
    claimed_columns: dict[str, set[str]] = {}
    for item in matches:
        if item.get("source") != "semantic_alias":
            continue
        claimed_columns.setdefault(str(item.get("table") or ""), set()).update(
            str(column) for column in (item.get("columns") or []) if str(column).strip()
        )
    filtered = [
        item
        for item in matches
        if not (
            item.get("source") == "column"
            and set(str(column) for column in (item.get("columns") or []))
            and set(str(column) for column in (item.get("columns") or [])).issubset(
                claimed_columns.get(str(item.get("table") or ""), set())
            )
        )
    ]

    # Resolve lexical alternatives before they become hard SCGA requirements.
    # A specific surface form owns the text span that contains it.  Separate
    # mentions remain independent, so average/sum/count semantics are not
    # collapsed merely because they share a base metric name.
    surviving: list[dict[str, Any]] = []
    for candidate in filtered:
        candidate_spans = [tuple(span) for span in candidate.get("matched_spans") or []]
        has_unshadowed_span = False
        for start, end in candidate_spans:
            shadowed = False
            for other in filtered:
                if other is candidate:
                    continue
                for other_start, other_end in other.get("matched_spans") or []:
                    contains = other_start <= start and other_end >= end
                    strictly_more_specific = (other_end - other_start) > (end - start)
                    same_span_better_source = (
                        other_start == start
                        and other_end == end
                        and other.get("source") == "semantic_alias"
                        and candidate.get("source") != "semantic_alias"
                    )
                    if contains and (strictly_more_specific or same_span_better_source):
                        shadowed = True
                        break
                if shadowed:
                    break
            if not shadowed:
                has_unshadowed_span = True
                break
        if has_unshadowed_span or not candidate_spans:
            surviving.append(candidate)

    # If no region was supplied, agreement among independent Schema hits can
    # identify a coherent mounted dataset.  Only a strong, multi-binding lead
    # is used; balanced cross-domain questions remain untouched.
    if not target_domain:
        domain_scores: dict[str, float] = {}
        domain_counts: dict[str, int] = {}
        for item in surviving:
            domain = _schema_domain(schema_map.get(str(item.get("table") or ""), ""))
            if not domain:
                continue
            longest = max(
                (len(_semantic_compact_text(term)) for term in item.get("matched_terms") or []),
                default=0,
            )
            source_weight = 2.0 if item.get("source") == "semantic_alias" else 1.0
            domain_scores[domain] = domain_scores.get(domain, 0.0) + source_weight + min(longest, 12) / 12.0
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        ordered_domains = sorted(domain_scores, key=domain_scores.get, reverse=True)
        if ordered_domains:
            best_domain = ordered_domains[0]
            runner_up = domain_scores.get(ordered_domains[1], 0.0) if len(ordered_domains) > 1 else 0.0
            if (
                domain_counts.get(best_domain, 0) >= 2
                and (runner_up == 0.0 or domain_scores[best_domain] > 2.0 * runner_up)
            ):
                surviving = [
                    item
                    for item in surviving
                    if _schema_domain(schema_map.get(str(item.get("table") or ""), "")) == best_domain
                ]

    for item in surviving:
        item.pop("matched_spans", None)
    return surviving


def enrich_intention_slots_from_schema(
    question: str,
    slots_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill stable intent fields from the question and mounted Schema metadata.

    This is the low-cost first stage of Intent Understanding.  It never rewrites
    the user question and never replaces an already populated LLM slot.
    """
    slots = dict(slots_dict or {})
    question_text = str(question or "")
    table_map = _schema_map_snapshot()
    mounted_domains: list[str] = []
    for yaml_str in table_map.values():
        try:
            domain = str((yaml.safe_load(yaml_str) or {}).get("domain") or "").strip()
        except Exception:
            domain = ""
        if domain and domain not in mounted_domains:
            mounted_domains.append(domain)
    explicit_domains = [
        domain
        for domain in mounted_domains
        if any(surface and surface in question_text for surface in _domain_surface_forms(domain))
    ]
    if explicit_domains and not str(slots.get("region_scope") or "").strip():
        slots["region_scope"] = explicit_domains[0]
    if explicit_domains and not str(slots.get("region") or "").strip():
        slots["region"] = explicit_domains[0]

    years = _query_years(question_text, slots)
    if len(years) == 1 and not str(slots.get("time") or "").strip():
        slots["time"] = years[0]
    elif len(years) >= 2 and not (slots.get("time_range") or []):
        slots["time_range"] = years

    matches = _semantic_schema_matches(question_text, slots, table_map)
    if not str(slots.get("region_scope") or "").strip():
        inferred_domains = {
            _schema_domain(table_map.get(str(match.get("table") or ""), ""))
            for match in matches
        }
        inferred_domains.discard("")
        if len(inferred_domains) == 1 and len(matches) >= 2:
            slots["region_scope"] = next(iter(inferred_domains))
    existing_metrics = [str(item) for item in (slots.get("metric_set") or []) if str(item).strip()]
    for match in matches:
        concept = str(match.get("concept") or "").strip()
        if concept and concept not in existing_metrics:
            existing_metrics.append(concept)
    if existing_metrics:
        slots["metric_set"] = existing_metrics
        if not str(slots.get("metric") or "").strip():
            slots["metric"] = existing_metrics[0]

    # Preserve every explicit comparison as a compact checklist.  This is not
    # a benchmark rule and does not attempt to translate it into SQL; downstream
    # Planner/SCGA use the original wording to ensure that a multi-condition
    # question does not silently lose its second predicate.
    if not (slots.get("condition") or {}):
        clauses = [
            fragment.strip(" \t\r\n，。；;？?")
            for fragment in re.split(r"[，。；;？?]+", question_text)
            if re.search(
                r"不超过|不少于|不低于|不高于|高出|低于|高于|超过|大于|小于|等于|至少|至多",
                fragment,
            )
        ]
        if clauses:
            slots["condition"] = {"clauses": list(dict.fromkeys(clauses))}

    analytical_rules = (
        ("correlation", ("皮尔逊", "相关系数", "correlation", "pearson")),
        ("regression", ("回归", "regression")),
        ("clustering", ("聚类", "k-means", "kmeans", "cluster")),
        ("statistical_test", ("正态性检验", "统计检验", "t 检验", "t检验", "shapiro", "mann-whitney", "anova")),
        ("temporal_change", ("相邻月份", "环比", "同比", "时序差分", "变化检测", "发生变化")),
        ("minimum_bounding", ("最小外接矩形", "bounding box", "bounding rectangle")),
        ("bootstrap", ("bootstrap", "自助法", "有放回")),
    )
    q_lower = question_text.casefold()
    if not str(slots.get("analytical_method") or "").strip():
        for method, terms in analytical_rules:
            if any(term in q_lower for term in terms):
                slots["analytical_method"] = method
                break

    spatial_rules = (
        ("within_distance", ("距离", "公里内", "km内", "缓冲区")),
        ("contains", ("完全落在", "包含", "within")),
        ("touches", ("接壤", "相邻", "touches")),
        ("intersects", ("相交", "intersect")),
    )
    if not str(slots.get("spatial_predicate") or "").strip():
        for predicate, terms in spatial_rules:
            if any(term in q_lower for term in terms):
                slots["spatial_predicate"] = predicate
                break
    if not str(slots.get("spatial_threshold") or "").strip():
        threshold = re.search(r"\d+(?:\.\d+)?\s*(?:km|公里|千米|m|米)", question_text, re.IGNORECASE)
        if threshold:
            slots["spatial_threshold"] = threshold.group(0)
    return slots


def init_or_load_vectorstore() -> Chroma:
    """初始化或加载 Chroma。YAML 未变且向量模型/维度签名一致则复用；否则自动删除重建（避免 1536/1024 维混用报错）。"""
    global _vectorstore, _table_name_to_yaml

    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore

        _table_name_to_yaml = _build_table_yaml_map()

        embeddings = config.get_schema_embeddings()

        current_hash = _compute_schemas_hash()
        need_rebuild = True

        if _CHROMA_DIR.exists() and _HASH_FILE.exists():
            try:
                stored_hash = _HASH_FILE.read_text(encoding="utf-8").strip()
                if stored_hash == current_hash:
                    if not _EMBEDDING_SIG_FILE.exists():
                        logger.info(
                            "[RAG] 未找到 .embedding_sig（旧版索引），将按当前向量模型重建 chroma_db"
                        )
                    else:
                        stored_sig = _EMBEDDING_SIG_FILE.read_text(encoding="utf-8").strip()
                        if _stored_signature_uses_current_model(stored_sig):
                            need_rebuild = False
                        else:
                            current_sig = _embedding_signature(embeddings)
                            if stored_sig != current_sig:
                                logger.info(
                                    "[RAG] 向量签名已变更 (%s -> %s)，重建 chroma_db",
                                    stored_sig,
                                    current_sig,
                                )
                            need_rebuild = stored_sig != current_sig
                # else: YAML 变更，保持 need_rebuild=True
            except Exception:
                logger.exception("[RAG] Failed to validate the persisted embedding signature")
                raise

        if not need_rebuild:
            try:
                _vectorstore = Chroma(
                    collection_name=_COLLECTION_NAME,
                    persist_directory=str(_CHROMA_DIR),
                    embedding_function=embeddings,
                )
                return _vectorstore
            except Exception as e:
                err = str(e).lower()
                if (
                    "dimension" in err
                    or "expecting embedding" in err
                    or "invalidargument" in err.replace("_", "")
                ):
                    logger.warning(
                        "[RAG] 加载 Chroma 失败（%s），将删除并重建", type(e).__name__
                    )
                    need_rebuild = True
                else:
                    raise

        documents: list[Document] = []
        for table_name, yaml_str in _table_name_to_yaml.items():
            data = yaml.safe_load(yaml_str)
            desc = data.get("table_description", "")
            domain = data.get("domain", "")
            st = data.get("spatiotemporal_properties", {}) or {}
            spatial_g = st.get("spatial_granularity", "")
            temporal_g = st.get("temporal_granularity", "")

            has_geom = st.get("has_geometry", False)
            semantic_aliases = data.get("semantic_aliases") or {}

            # ── 1. 表级文档（1个）：包含表的全局语义信息 ──
            table_page_content = (
                f"table_name: {table_name}\n"
                f"table_description: {desc}\n"
                f"domain: {domain}\n"
                f"spatial_granularity: {spatial_g}\n"
                f"temporal_granularity: {temporal_g}\n"
                f"has_geometry: {has_geom}\n"
                f"semantic_aliases: {json.dumps(semantic_aliases, ensure_ascii=False)}"
            )
            documents.append(Document(
                page_content=table_page_content,
                metadata={
                    "parent_table": table_name,
                    "doc_type": "table",
                    "yaml_content": yaml_str,
                },
            ))

            # ── 2. 列级文档（每列1个）：让每个指标字段独立参与语义竞争 ──
            for col in data.get("columns", []):
                col_name = col.get("name") or col.get("column_name", "")
                col_desc = col.get("description", "")
                aliases = _as_term_list(col.get("aliases"))
                col_type = col.get("type", "")
                is_spatial = col.get("is_spatial_column", False)
                is_temporal = col.get("is_temporal_column", False)
                fk = col.get("foreign_key")
                fk_str = str(fk) if fk and str(fk).strip().lower() != "null" else ""

                enums = col.get("enums") or col.get("value_mapping") or {}
                if isinstance(enums, dict):
                    enums_str = ", ".join(f"{k}={v}" for k, v in enums.items())
                elif isinstance(enums, list):
                    enums_str = ", ".join(str(e) for e in enums)
                else:
                    enums_str = str(enums) if enums else ""

                col_parts = [f"column_name: {col_name}", f"description: {col_desc}"]
                if aliases:
                    col_parts.append(f"aliases: {', '.join(aliases)}")
                if col_type:
                    col_parts.append(f"type: {col_type}")
                if is_spatial:
                    col_parts.append("is_spatial_column: true")
                if is_temporal:
                    col_parts.append("is_temporal_column: true")
                if fk_str:
                    col_parts.append(f"foreign_key: {fk_str}")
                if enums_str:
                    col_parts.append(f"values: {enums_str}")
                col_parts.append(f"(belongs to table: {table_name}, domain: {domain})")
                col_page_content = "\n".join(col_parts)

                if col_name or col_desc:
                    documents.append(Document(
                        page_content=col_page_content,
                        metadata={
                            "parent_table": table_name,
                            "doc_type": "column",
                            "column_name": col_name,
                        },
                    ))

        # 与 schema_utils 内存缓存对齐，并彻底删除旧 Chroma 持久化目录，避免 from_documents 在已有目录上追加导致幽灵重复文档
        clear_schema_cache()
        if _CHROMA_DIR.exists():
            shutil.rmtree(_CHROMA_DIR)
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=_COLLECTION_NAME,
            persist_directory=str(_CHROMA_DIR),
        )

        _HASH_FILE.write_text(current_hash, encoding="utf-8")
        _EMBEDDING_SIG_FILE.write_text(_embedding_signature(embeddings), encoding="utf-8")
        return _vectorstore


def _log_rag_scores(table_scores: dict[str, float]) -> None:
    """输出各表聚合得分日志。"""
    if not table_scores or not logger.isEnabledFor(logging.INFO):
        return
    logger.info("[RAG] 各表聚合得分（表级+列级加权）：")
    for tname in sorted(table_scores, key=lambda t: table_scores[t], reverse=True):
        logger.info("    %s: %.4f", tname, table_scores[tname])


def _log_selected_columns(top_tables: list[str], selected_columns_by_table: dict[str, list[str]]) -> None:
    """输出 Top 表及其保留列的日志。"""
    if not top_tables or not logger.isEnabledFor(logging.INFO):
        return
    logger.info("[RAG] Top 表保留的高分列：")
    for table_name in top_tables:
        selected_columns = selected_columns_by_table.get(table_name, [])
        if selected_columns:
            logger.info("    %s: %s", table_name, ", ".join(selected_columns))


def _similarity_search_with_ssl_retry(
    vectorstore: Chroma,
    query_text: str,
    *,
    k: int,
) -> list[tuple[Document, float]]:
    """仅对 DashScope 偶发 TLS EOF 做一次短重试；超时和其他异常直接抛出。"""
    try:
        return vectorstore.similarity_search_with_score(query_text, k=k)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".lower()
        transient_ssl_eof = (
            "ssl" in message
            and (
                "unexpected_eof_while_reading" in message
                or "eof occurred in violation of protocol" in message
            )
        )
        if not transient_ssl_eof:
            raise
        logger.warning("[RAG] Embedding TLS EOF，2 秒后仅重试一次 Schema 检索")
        time.sleep(2)
        return vectorstore.similarity_search_with_score(query_text, k=k)


def _extract_foreign_table_names(yaml_str: str) -> set[str]:
    """从单张表的 YAML 中提取所有外键引用的目标表名。"""
    data = yaml.safe_load(yaml_str)
    fk_tables: set[str] = set()
    for col in data.get("columns", []):
        fk = col.get("foreign_key")
        if not fk or fk == "null" or str(fk).strip().lower() == "null":
            continue
        ref_table = str(fk).split(".")[0].strip()
        if ref_table:
            fk_tables.add(ref_table)
    return fk_tables


def _extract_column_name_from_doc(doc: Document) -> str:
    """从列级检索文档中解析列名。"""
    metadata_name = doc.metadata.get("column_name")
    if metadata_name:
        return str(metadata_name).strip()

    for raw_line in str(doc.page_content or "").splitlines():
        line = raw_line.strip()
        if line.startswith("column_name:"):
            return line.split(":", 1)[1].strip()
    return ""


def _split_identifier_tokens(name: str) -> list[str]:
    """按 snake_case / camelCase / 连字符切分列名，便于识别关键标识列。"""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or ""))
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized)
    return [token.lower() for token in normalized.split("_") if token]


def _is_essential_column(column: dict) -> bool:
    """识别 SQL 生成阶段应始终保留的关键列。"""
    col_name = str(column.get("name") or column.get("column_name") or "").strip()
    if not col_name:
        return False

    if column.get("is_primary_key") or column.get("is_spatial_column") or column.get("is_temporal_column"):
        return True

    fk = str(column.get("foreign_key") or "").strip().lower()
    if fk and fk != "null":
        return True

    tokens = set(_split_identifier_tokens(col_name))
    if tokens & {"id", "name", "code", "iso", "level", "geometry", "geom", "year", "time", "date", "ratio", "proportion", "weight", "area"}:
        return True

    description = str(column.get("description") or "")
    if any(token in description for token in ["名称", "地名", "州名", "编码", "代码", "标识", "编号", "边界", "年份", "时间", "比例", "占比", "权重", "面积"]):
        return True

    lower_description = description.lower()
    return any(token in lower_description for token in ["geometry", "identifier", "boundary", "time", "date", "ratio", "proportion", "weight", "area"])


def _build_filtered_schema_yaml(yaml_str: str, selected_column_names: list[str]) -> str:
    """仅保留高分列及 SQL 必需关键列，减少传入 Text2SQL 的 schema 噪声。"""
    try:
        data = yaml.safe_load(yaml_str) or {}
    except Exception:
        return yaml_str

    if not isinstance(data, dict):
        return yaml_str

    columns = data.get("columns") or []
    if not isinstance(columns, list):
        return yaml_str

    selected_set = {str(name).strip() for name in (selected_column_names or []) if str(name).strip()}
    filtered_columns: list[dict] = []
    filtered_column_names: set[str] = set()
    for column in columns:
        if not isinstance(column, dict):
            continue
        col_name = str(column.get("name") or column.get("column_name") or "").strip()
        if selected_set and col_name in selected_set:
            filtered_columns.append(column)
            filtered_column_names.add(col_name)
            continue
        if _is_essential_column(column):
            filtered_columns.append(column)
            filtered_column_names.add(col_name)

    selected_match_count = sum(
        1 for column in filtered_columns
        if str(column.get("name") or column.get("column_name") or "").strip() in selected_set
    )
    if selected_set and selected_match_count == 0:
        for column in columns:
            if not isinstance(column, dict):
                continue
            col_name = str(column.get("name") or column.get("column_name") or "").strip()
            if not col_name or col_name in filtered_column_names:
                continue
            filtered_columns.append(column)
            filtered_column_names.add(col_name)
            if len(filtered_columns) >= min(20, len(columns)):
                break

    if not filtered_columns:
        filtered_columns = [column for column in columns if isinstance(column, dict)][: min(15, len(columns))]

    filtered_data = dict(data)
    filtered_data["columns"] = filtered_columns
    filtered_data = _normalize_schema_dict(filtered_data)
    return yaml.dump(filtered_data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _collect_mounted_domains(table_yaml_map: dict[str, str]) -> list[str]:
    """从已挂载 schema 的 YAML 中提取非空 domain，去重后按长度降序、字典序升序排列（子串匹配时优先更长、更具体的 domain）。"""
    unique: set[str] = set()
    for yaml_str in table_yaml_map.values():
        try:
            data = yaml.safe_load(yaml_str) or {}
            d = str(data.get("domain", "")).strip()
            if d:
                unique.add(d)
        except Exception:
            continue
    return sorted(unique, key=lambda x: (-len(x), x))


def extract_table_name_from_schema_yaml(schema_yaml: str) -> str:
    """从单段 schema YAML 中提取表名。"""
    for line in str(schema_yaml).splitlines():
        if line.strip().startswith("table_name:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_rag_query(scenario: str, **kwargs: Any) -> str:
    """统一构建 RAG 检索用的自然语言 query。

    scenario:
        - ``text2sql``: ``global_question`` + ``sql_task`` 按行拼接（空段省略）。
        - ``tool``: ``question``。
    """
    if scenario == "text2sql":
        anchor = str(kwargs.get("global_question") or "").strip()
        step = str(kwargs.get("sql_task") or "").strip()
        lines: list[str] = []
        if anchor:
            lines.append(f"全局问题：{anchor}")
        if step:
            lines.append(f"当前 SQL 子任务：{step}")
        return "\n".join(lines)
    if scenario == "tool":
        return str(kwargs.get("question") or "").strip()
    return ""


def build_planner_schemas_yaml_from_rag_list(
    schemas: list[str],
    max_tables: int | None = None,
) -> str:
    """按 RAG 返回顺序，仅保留前 max_tables 个唯一表的 schema YAML（供 Planner 摘要）。"""
    cap = config.RAG_PLANNER_MAX_TABLES if max_tables is None else max_tables
    if not schemas:
        return ""

    selected: list[str] = []
    seen_tables: set[str] = set()
    for schema_yaml in schemas:
        table_name = extract_table_name_from_schema_yaml(schema_yaml)
        if not table_name or table_name in seen_tables:
            continue
        seen_tables.add(table_name)
        selected.append(schema_yaml)
        if len(selected) >= cap:
            break

    return "\n\n".join(selected)


def build_text2sql_schemas_yaml_from_bundle(
    schema_bundle: dict[str, Any],
    final_table_names: list[str],
) -> str:
    """按表顺序拼接传入 Text2SQL 的 YAML：优先 RAG 列裁剪片段，缺表时回退磁盘全量 YAML。"""
    if not final_table_names:
        return load_schemas_by_table_names([])

    bundle_names = schema_bundle.get("table_names") or []
    bundle_schemas = schema_bundle.get("schemas") or []
    by_table: dict[str, str] = {}
    for i, tname in enumerate(bundle_names):
        if not isinstance(tname, str) or not tname:
            continue
        if i >= len(bundle_schemas):
            break
        blob = bundle_schemas[i]
        if isinstance(blob, str) and blob.strip():
            by_table[tname] = blob

    pieces: list[str] = []
    for tname in final_table_names:
        cropped = by_table.get(tname)
        if isinstance(cropped, str) and cropped.strip():
            pieces.append(cropped)
            continue
        fallback = load_schemas_by_table_names([tname]).strip()
        if fallback and fallback != "（无 Schema）":
            pieces.append(fallback)

    merged = "\n\n".join(pieces)
    if not merged.strip() or merged.strip() == "（无 Schema）":
        return load_schemas_by_table_names(final_table_names)
    return merged


def format_schema_yaml_by_exact_table_names(table_names: list[str]) -> str:
    """按表名从缓存加载全量 YAML（无向量检索）；过滤空表名。"""
    cleaned = [str(n).strip() for n in table_names if isinstance(n, str) and str(n).strip()]
    return load_schemas_by_table_names(cleaned)


def retrieve_top_k_schema_bundle(
    slots_dict: dict,
    k: int | None = None,
    natural_language_query: str = "",
    semantic_anchor_query: str | None = None,
) -> dict:
    """
    多向量父子文档检索（RAG Schema 检索主流程）：
        放宽召回碎片文档 → 加权聚合到父表 → Top-K → 外键补全。

    Args:
        slots_dict: 当前意图解析得到的槽位信息（dict）。
        k: 最终返回的主表数量（不含外键补全带入的依赖表）；默认 ``config.RAG_TOP_K_DEFAULT``。
        natural_language_query: 用于向量召回的查询，可包含紧凑槽位关键词。
        semantic_anchor_query: 用于确定性字段绑定的原始用户问题。它与向量召回
            文本分离，避免把上游推断出的物理字段名再次当作用户证据。

    Returns:
        包含裁剪后 schema YAML、拼接文本与表名列表的结构化结果。
    """
    global _vectorstore, _table_name_to_yaml

    top_k = config.RAG_TOP_K_DEFAULT if k is None else k

    # 如果向量库尚未加载，则初始化或从持久化加载
    if _vectorstore is None:
        init_or_load_vectorstore()

    # -------- 第一步：构造召回查询文本，提取 domain 约束 --------
    query_text = ""
    if natural_language_query:
        # 优先拼接自然语言原始查询
        query_text += f"Query: {natural_language_query}\n"
    # 拼接槽位字典（带语义约束）
    query_text += json.dumps(slots_dict, ensure_ascii=False)

    # 尝试识别 domain：仅当槽位综合文本命中已挂载 YAML 中声明的 domain 时生效（数据驱动，无硬编码地域词）
    region_scope = (slots_dict or {}).get("region_scope")
    region = (slots_dict or {}).get("region")
    region_set = (slots_dict or {}).get("region_set") or []

    probe_parts: list[str] = []
    if region_scope:
        s = str(region_scope).strip()
        if s:
            probe_parts.append(s)
    if region:
        s = str(region).strip()
        if s:
            probe_parts.append(s)
    if region_set:
        probe_parts.append(" ".join(str(r) for r in region_set))
    # Natural-language domain mentions remain available even when the fast
    # Intent path did not call an LLM.
    probe_text = " ".join([*probe_parts, str(natural_language_query or "")])

    domain_hint = ""
    for d in _collect_mounted_domains(_table_name_to_yaml):
        if d in probe_text:
            domain_hint = d
            break

    # 如果识别到 domain_hint, 优先加在 query_text 前缀
    if domain_hint:
        query_text = f"domain: {domain_hint}\n{query_text}"

    # 使用向量库检索，高召回列级碎片（k 大）再交由下游表聚合重排
    docs_with_scores = _similarity_search_with_ssl_retry(
        _vectorstore,
        query_text,
        k=150,
    )

    # -------- 第二步：文档聚合打分，汇总到父表 --------
    # Chroma 默认返回 L2 距离（越小越相似），这里转换为相似度分数 1/(1+d)
    # 对于列文档权重设为 1.5，表级设为 1.0，更重视具体字段
    _WEIGHT = {"column": 1.5, "table": 1.0}

    table_doc_scores: dict[str, float] = {}         # 每个表自身文档的最高得分
    table_column_scores: dict[str, list[float]] = {}# 每个表聚合的所有列文档得分列表
    column_scores: dict[str, dict[str, float]] = {} # 每个表每列的累计得分

    for doc, distance in docs_with_scores:
        parent = doc.metadata.get("parent_table", "")
        doc_type = doc.metadata.get("doc_type", "table")  # "table" or "column"
        yaml_str = _table_name_to_yaml.get(parent, "")
        if not parent:
            continue
        sim_score = 1.0 / (1.0 + distance)
        weight = _WEIGHT.get(doc_type, 1.0)
        score = sim_score * weight

        # 域过滤加权（比如仅要中国的表，多加分，否则衰减分数）
        if domain_hint and yaml_str:
            try:
                data = yaml.safe_load(yaml_str) or {}
                table_domain = str(data.get("domain", "")).strip()
                if table_domain == domain_hint:
                    score *= 1.35  # 强 domain 匹配提升
                elif table_domain:
                    score *= 0.05  # 明确域不匹配时近似排除，避免宽表凭字段数量混入
            except Exception:
                pass

        target_metrics = (slots_dict or {}).get("metric_set") or []
        if not target_metrics and (slots_dict or {}).get("metric"):
            target_metrics = [(slots_dict or {}).get("metric")]
        if target_metrics and doc_type == "column":
            col_name = _extract_column_name_from_doc(doc)
            col_text = (doc.page_content or "").lower()
            col_name_l = col_name.lower()
            for tm in target_metrics:
                if tm is None:
                    continue
                tm_s = str(tm).strip().lower()
                if not tm_s:
                    continue
                # 1. 完整包含匹配
                if tm_s in col_name_l or tm_s in col_text:
                    score *= 1.5
                    break
                # 2. 连续英文/下划线 token（≥4）子串匹配，例如 "mixed_forest 占比" → mixed_forest
                eng_tokens = re.findall(r"[a-z_]{4,}", tm_s)
                if any(token in col_name_l or token in col_text for token in eng_tokens):
                    score *= 1.5
                    break

        if doc_type == "table":
            # 保存每张表的最大主文档得分
            table_doc_scores[parent] = max(table_doc_scores.get(parent, 0.0), score)
        elif doc_type == "column":
            # 列文档分数列表
            table_column_scores.setdefault(parent, []).append(score)
            column_name = _extract_column_name_from_doc(doc)
            if column_name:
                column_scores.setdefault(parent, {})
                column_scores[parent][column_name] = column_scores[parent].get(column_name, 0.0) + score

    # -------- 聚合各父表的总分 --------
    table_scores: dict[str, float] = {}
    for parent in set(table_doc_scores.keys()) | set(table_column_scores.keys()):
        t_score = table_doc_scores.get(parent, 0.0)           # 此表的主文档分
        c_scores = table_column_scores.get(parent, [])
        if c_scores:
            # 只看最相关的少量字段，避免宽表因列多获得系统性优势。
            strongest = sorted(c_scores, reverse=True)[:3]
            max_col_score = strongest[0]
            c_score_agg = max_col_score + 0.15 * sum(strongest[1:])
        else:
            c_score_agg = 0.0
        table_scores[parent] = t_score + c_score_agg

    query_years = _query_years(natural_language_query, slots_dict)
    monthly_query = bool(re.search(
        r"(?:\d{1,2}\s*[、,，和及至到\-]?\s*)+月|月度|逐月|每月|夏季|冬季",
        str(natural_language_query or ""),
    ))
    for table_name in list(table_scores):
        yaml_str = _table_name_to_yaml.get(table_name, "")
        time_factor = _schema_time_compatibility(
            yaml_str,
            years=query_years,
            monthly_query=monthly_query,
        )
        table_scores[table_name] *= 0.02 if time_factor <= 0 else time_factor

    # 输出各父表聚合分数（可选debug）
    _log_rag_scores(table_scores)

    # -------- 第三步：语义覆盖优先，再按向量分数补齐 Top-K 父表 --------
    semantic_matches = _semantic_schema_matches(
        semantic_anchor_query if semantic_anchor_query is not None else natural_language_query,
        slots_dict,
        _table_name_to_yaml,
    )
    matches_by_concept: dict[str, list[dict[str, Any]]] = {}
    for match in semantic_matches:
        table_name = str(match.get("table") or "")
        if not table_name:
            continue
        compatibility = float(match.get("time_compatibility") or 1.0)
        # Alias hit is a precise Schema signal.  Add a bounded boost rather
        # than replacing vector similarity altogether.
        table_scores[table_name] = table_scores.get(table_name, 0.0) + 1.5 * compatibility
        matches_by_concept.setdefault(str(match.get("concept") or table_name), []).append(match)

    semantic_bindings: list[dict[str, Any]] = []
    forced_tables: list[str] = []
    requested_granularities = _requested_temporal_granularities(
        semantic_anchor_query if semantic_anchor_query is not None else natural_language_query
    )
    for concept, candidates in matches_by_concept.items():
        def _binding_rank(item: dict[str, Any]) -> tuple[float, float, int]:
            return (
                float(item.get("time_compatibility") or 1.0),
                table_scores.get(str(item.get("table") or ""), 0.0),
                max((len(str(term)) for term in item.get("matched_terms") or []), default=0),
            )

        selected: list[dict[str, Any]] = []
        # 同一物理指标若在问题中被明确要求月度与年度两个粒度，保留各自最佳表。
        # 普通单粒度问题仍只选一个绑定，避免扩大 Schema 上下文。
        if len(requested_granularities) > 1:
            for role in sorted(requested_granularities):
                role_candidates = [
                    item for item in candidates
                    if _temporal_granularity_role(
                        _schema_temporal_granularity(
                            _table_name_to_yaml.get(str(item.get("table") or ""), "")
                        )
                    ) == role
                ]
                if role_candidates:
                    selected.append(max(role_candidates, key=_binding_rank))
        if not selected:
            selected = [max(candidates, key=_binding_rank)]

        for best in selected:
            table_name = str(best.get("table") or "")
            if table_name and table_name not in forced_tables:
                forced_tables.append(table_name)
            semantic_bindings.append({
                "concept": concept,
                "table": table_name,
                "columns": list(best.get("columns") or []),
                "matched_terms": list(best.get("matched_terms") or []),
            })

    requested_metrics = [
        str(item).strip() for item in ((slots_dict or {}).get("metric_set") or []) if str(item).strip()
    ]
    bound_concepts = {_semantic_compact_text(item["concept"]) for item in semantic_bindings}
    uncovered_metrics = [
        metric for metric in requested_metrics if _semantic_compact_text(metric) not in bound_concepts
    ]
    coverage_complete = not uncovered_metrics

    ranked_tables = sorted(table_scores, key=lambda t: table_scores[t], reverse=True)
    if domain_hint:
        ranked_tables = [
            table_name
            for table_name in ranked_tables
            if _schema_domain(_table_name_to_yaml.get(table_name, "")) in {"", domain_hint}
        ]
    top_tables = list(forced_tables)
    # Once all requested concepts have an explicit Schema binding, those tables
    # are the pre-filtering result.  Entity/join tables are added by FK traversal
    # below; an unrelated vector fallback only enlarges the prompt and can tempt
    # SCGA to use a semantically wrong field.
    needs_entity_or_geometry_context = bool(
        str((slots_dict or {}).get("spatial_predicate") or "").strip()
        or str((slots_dict or {}).get("spatial_threshold") or "").strip()
    )
    selection_limit = (
        top_k
        if not forced_tables or not coverage_complete or needs_entity_or_geometry_context
        else len(forced_tables)
    )
    for table_name in ranked_tables:
        if table_name not in top_tables:
            top_tables.append(table_name)
        if len(top_tables) >= selection_limit:
            break
    top_tables = top_tables[:max(selection_limit, len(forced_tables))]

    # 每个表选出Top20高分字段用于下游过滤
    selected_columns_by_table: dict[str, list[str]] = {}
    for table_name, scores in column_scores.items():
        ranked_columns = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected_columns_by_table[table_name] = [name for name, _ in ranked_columns[:20]]

    # 输出字段选择（可选debug）
    _log_selected_columns(top_tables, selected_columns_by_table)

    # -------- 第四步：外键依赖 BFS 全补齐 --------
    # 依次读取每张主表，并做必要的外键查找（BFS方式扩展相关依赖表，避免漏依赖但限制深度和总表数）
    retrieved_tables: dict[str, str] = {}   # 表名到裁剪后schema文本
    ordered_table_names: list[str] = []     # 顺序保存表名，便于拼接输出
    for tname in top_tables:
        yaml_str = _table_name_to_yaml.get(tname, "")
        if yaml_str:
            retrieved_tables[tname] = _build_filtered_schema_yaml(
                yaml_str,
                selected_columns_by_table.get(tname, []),
            )
            ordered_table_names.append(tname)

    # BFS 外键依赖拓展
    max_total = max(config.RAG_FK_MAX_TOTAL_TABLES, len(ordered_table_names))  # 最大表数限制
    max_depth = max(0, config.RAG_FK_MAX_DEPTH)                                # 外键最大深度
    fk_queue: deque[str] = deque(retrieved_tables.keys())                      # 从主表开始广度优先
    visited = set(retrieved_tables.keys())
    table_depth: dict[str, int] = {t: 0 for t in retrieved_tables.keys()}

    while fk_queue:
        if len(ordered_table_names) >= max_total:
            break
        current = fk_queue.popleft()
        dc = table_depth.get(current, 0)   # 当前 BFS 已达深度
        if dc >= max_depth:
            continue
        yaml_str = retrieved_tables.get(current) or _table_name_to_yaml.get(current, "")
        if not yaml_str:
            continue
        # 提取所有外键指向的表
        for ref_table in _extract_foreign_table_names(yaml_str):
            if len(ordered_table_names) >= max_total:
                break
            if ref_table in visited:
                continue
            visited.add(ref_table)
            dep_yaml = _table_name_to_yaml.get(ref_table, "")
            if dep_yaml:
                # 引用表裁剪字段
                retrieved_tables[ref_table] = _build_filtered_schema_yaml(
                    dep_yaml,
                    selected_columns_by_table.get(ref_table, []),
                )
                ordered_table_names.append(ref_table)
                table_depth[ref_table] = dc + 1  # 依赖表深度+1
                fk_queue.append(ref_table)

    # -------- 结果组装，返回包含所有表的 YAML 和顺序列表 --------
    schemas = [retrieved_tables[table_name] for table_name in ordered_table_names if table_name in retrieved_tables]
    return {
        "schemas": schemas,                            # 个别主表/依赖表的yaml列表
        "schemas_yaml": "\n\n".join(schemas),          # 拼接合并文本
        "table_names": ordered_table_names,            # 表名顺序列表
        "semantic_bindings": semantic_bindings,
        "schema_coverage": {
            "matched_concepts": [item["concept"] for item in semantic_bindings],
            "uncovered_metrics": uncovered_metrics,
            "complete": coverage_complete,
        },
    }
