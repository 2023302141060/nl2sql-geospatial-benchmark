# -*- coding: utf-8 -*-
"""Question-derived answer projection contracts.

The contract contains no answer values and no benchmark-specific field names.
It gives downstream SQL/Python workers a stable description of output shape,
statistical roles and information that must not be lost during ranking.
"""
from __future__ import annotations

import re
from typing import Any


_STATISTIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"皮尔逊|pearson|斯皮尔曼|spearman|肯德尔|kendall|相关系数|correlation", "correlation"),
    (r"斜率|slope", "slope"),
    (r"p\s*(?:[ -]?value|值)|显著性", "p_value"),
    (r"r[ -]?squared|r\s*\^?2|决定系数", "r_squared"),
    (r"w\s*(?:统计量|值)|w[ -]?statistic", "w_statistic"),
    (r"t\s*(?:统计量|值)|t[ -]?statistic", "t_statistic"),
    (r"平均|均值|mean|average", "mean"),
    (r"总和|合计|sum", "sum"),
    (r"数量|个数|多少个|计数|count", "count"),
    (r"标准差|std|standard deviation", "std"),
    (r"方差|variance", "variance"),
    (r"轮廓系数|silhouette", "silhouette_score"),
    (r"面积|area", "area"),
)


# Stable interchange names for statistics whose notation is otherwise highly
# variable across LLMs (for example ``pearson_corr`` versus ``correlation``).
# These are semantic protocol fields, not benchmark answer values.  Descriptive
# aliases may still be emitted alongside them.
_CANONICAL_STATISTIC_FIELDS: dict[str, str] = {
    "correlation": "pearson_r",
    "slope": "slope",
    "p_value": "p_value",
    "r_squared": "r_squared",
    "w_statistic": "w_statistic",
    "t_statistic": "t_statistic",
}


def build_answer_projection_contract(
    question: str,
    slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(question or "").strip()
    low = text.lower()
    slots_d = slots if isinstance(slots, dict) else {}

    top_match = re.search(
        r"(?:前\s*|top\s*)(\d+)|(?:最高|最低|最大|最小|最多|最少)的?\s*(\d+)\s*个",
        low,
        re.IGNORECASE,
    )
    top_k = int(next(g for g in top_match.groups() if g)) if top_match else None
    asks_count = bool(re.search(r"多少个|数量|个数|计数|网格数|单元数|count", low))
    asks_multiple = bool(
        re.search(
            r"哪些|所有|各(?:[^，。；,;]{0,8})?(?:组|类|州|网格|空间单元)|分别|逐个",
            low,
        )
        or re.search(
            r"按[^，。；,;]{1,40}?(?:分类|分组|类别|类型)[^，。；,;]{0,12}?(?:统计|计算|汇总|排序)",
            low,
        )
    )
    asks_rank = bool(top_k or re.search(r"排名|排行|排序|最高|最低|最大|最小|最多|最少", low))

    statistics = [name for pattern, name in _STATISTIC_PATTERNS if re.search(pattern, low)]
    statistics = list(dict.fromkeys(statistics))
    canonical_statistic_fields: list[str] = []
    for name in statistics:
        if name == "correlation":
            if re.search(r"kendall|肯德尔", low, re.IGNORECASE):
                canonical_statistic_fields.append("kendall_tau")
            elif re.search(r"spearman|斯皮尔曼", low, re.IGNORECASE):
                canonical_statistic_fields.append("spearman_rho")
            elif re.search(r"pearson|皮尔逊", low, re.IGNORECASE):
                canonical_statistic_fields.append("pearson_r")
            else:
                canonical_statistic_fields.append("correlation")
        elif name in _CANONICAL_STATISTIC_FIELDS:
            canonical_statistic_fields.append(_CANONICAL_STATISTIC_FIELDS[name])
    if "silhouette_score" in statistics:
        cluster_counts = list(
            dict.fromkeys(
                int(value)
                for value in re.findall(r"\bk\s*=\s*(\d+)\b", low, re.IGNORECASE)
            )
        )
        canonical_statistic_fields.extend(
            f"silhouette_k{count}" for count in cluster_counts
        )
        if len(cluster_counts) > 1 or re.search(r"较优|最佳|最优|better|best", low):
            canonical_statistic_fields.append("best_k")
        canonical_statistic_fields = list(dict.fromkeys(canonical_statistic_fields))

    if top_k:
        output_shape = "ranked_records"
    elif asks_count and not asks_multiple and len(statistics) <= 1:
        output_shape = "scalar"
    elif asks_rank and asks_multiple:
        # 排序后的多实体结果必须保留为完整有序记录；只有“哪个最高/最低”
        # 这类单一极值问题才收缩为 single_record。
        output_shape = "ranked_records"
    elif asks_rank:
        output_shape = "single_record"
    elif asks_multiple:
        output_shape = "records"
    else:
        output_shape = "single_record_or_scalar"

    grouping_dimension = ""
    group_match = re.search(
        r"(?:按|根据)(.{1,24}?)(?:把|将|等频|分为|分成|分组|统计各组)",
        text,
    )
    if group_match:
        grouping_dimension = group_match.group(1).strip(" ，、：:")

    preserve_signed = bool(
        re.search(r"绝对值|绝对相关|绝对.*系数|absolute", low)
        and any(name in statistics for name in ("correlation", "slope"))
    )

    metric_set = slots_d.get("metric_set")
    if isinstance(metric_set, list):
        requested_metrics = [str(x) for x in metric_set if str(x).strip()]
    else:
        metric = str(slots_d.get("metric") or "").strip()
        requested_metrics = [metric] if metric else []

    return_each_requested_metric = bool(
        re.search(r"分别.{0,10}(?:输出|列出|给出|返回)", text)
        or re.search(r"(?:输出|列出|给出|返回).{0,12}(?:分别|各个|所有).{0,8}数值", text)
        or re.search(r"标准化后.{0,16}(?:两个|各个|所有).{0,8}数值", text)
    )

    return {
        "output_shape": output_shape,
        "top_k": top_k,
        "statistics": statistics,
        "canonical_statistic_fields": canonical_statistic_fields,
        "requested_metrics": requested_metrics,
        "return_each_requested_metric": return_each_requested_metric,
        "grouping_dimension": grouping_dimension,
        "preserve_signed_statistic": preserve_signed,
        "rules": [
            "字段名表达实体、指标、统计量和单位，不使用无法解释的通用键",
            "排名记录同时保留实体标识和排序指标",
            "若按绝对值排序，保留原始带符号统计量，并可另附绝对值排序键",
            "分组记录的组键体现实际分组维度，不使用无语义的 group_id",
        ],
    }
