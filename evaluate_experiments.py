#!/usr/bin/env python3
"""
evaluate_experiments.py  ·  v5（严格结构化评估）
=================================================
主指标只评价通过 StrictFinalAnswer JSON Schema/Pydantic 校验的完整结构化答案。
执行失败、超时、缺少结构化答案、Schema 违规均直接记为失败；自然语言文本不得补救主指标。

主指标
------
  strict_structured_accuracy：严格结构化正确率。实体与数值必须在同一记录中配对；
                              实体/记录列表要求无额外行且无缺失行，排名要求顺序一致；
                              字典/记录允许预测含额外字段，但 Gold 字段必须按语义键匹配；
                              若排名题的 Gold 截断结果全部并列，则允许同值的另一组 Top-K
                              结构化记录作为并列等价答案。
  entity_precision/recall/f1：实体列表或记录列表的行级宏观指标。
  exact_set_match：无序实体/记录列表的精确集合匹配。
  ranking_exact_match：排名列表的顺序精确匹配；并列截断题按 tie-aware Top-K 口径计入。

路由指标
--------
  experiment_results_detail.csv 含 used_python_tool 与 routing_tp/fp/fn/tn；summary/overall
         在 routing_accuracy（全样本：(TP+TN)/N）之外，另含 routing_precision / routing_recall /
         routing_f1 及 routing_fp_rate_among_negatives（FP/(FP+TN)，即 gold 负例中的误调用 Python 占比）。

数据来源
--------
  默认基准：`dev_plus.yaml`（200 题公开基准）；可用 `--benchmark_path` 覆盖。
  主指标结构化：`results.strict_answer` + `results.answer_schema_valid`；
                同时核对其 data_payload 与完整 `results.parsed_payload` 一致。
  难度：   基准中历史上的 very_hard 在报表中统一记为 hard。
  步数：   报表字段 graph_steps / avg_graph_steps 使用 trace_summary.tool_routing_path

  first_execution_path_success：辅助诊断指标。仅当最终严格结构化答案正确，且执行过程中
  guardrail_retries=0、failed_calls=0 时记为 1。该指标用于说明一次执行路径的稳定性，
  不替代 strict_structured_accuracy，也不把经过合理重试后得到的正确答案改判为错误。
           的长度（与集成测试中每条 ToolMessage 一条记录一致），不再使用
           execution_metrics.total_graph_steps（LangGraph stream chunk 计数，
           在去除 Planner / LLM2Code 等消融之间不可比）。

用法
----
    python evaluate_experiments.py --benchmark_path benchmark/dev_plus.yaml \\
        --run_dirs benchmark/agent_runs/<本次完整实验目录> \\
        --output_dir results/

    # 统一评价已归档到 benchmark/agent_runs 下的 7 个 qwen3.7-plus 消融/基线：
    python evaluate_experiments.py --ablation_suite

    # 统一评价已归档到 benchmark/agent_runs 下的 5 个主模型：
    python evaluate_experiments.py --model_suite

    # 统一评价 5 个主模型 + 7 个消融/基线，生成最终总表：
    python evaluate_experiments.py --final_suite

默认按完整基准评价：结果目录中缺失的 question_id 会作为执行失败补入分母。
仅在冒烟或子集诊断时显式增加 ``--allow_partial``。
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Tuple

import yaml
import pandas as pd

from agent.answer_schema import coerce_strict_answer


# ══════════════════════════════════════════════════════════════════════════════
# 0.  评估配置
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_ROOT = Path(__file__).resolve().parent

# 默认基准为完整 dev_plus.yaml；结果目录必须通过 --run_dirs 显式指定，
# 避免无意中混入历史模型或旧版消融结果。
BENCHMARK_RELPATH = "dev_plus.yaml"
OUTPUT_RELPATH = "results"

ABLATION_SUITE_RUN_DIRS = [
    "benchmark/agent_runs/ablation_standard_plan_execute_qwen3.7_plus",
    "benchmark/agent_runs/ablation_standard_react_qwen3.7_plus",
    "benchmark/agent_runs/ablation_no_llm2code_qwen3.7_plus",
    "benchmark/agent_runs/ablation_no_llm2code_postgis_qwen3.7_plus",
    "benchmark/agent_runs/ablation_no_planner_qwen3.7_plus",
    "benchmark/agent_runs/ablation_postgis_qwen3.7_plus",
    "benchmark/agent_runs/baseline_llm_geo_qwen3.7_plus",
]

ABLATION_SUITE_OUTPUT_RELPATH = "results/formal_ablation_qwen3.7_plus_evaluation"

MODEL_SUITE_RUN_DIRS = [
    "benchmark/agent_runs/formal_v2_qwen3.7_max_2026_05_17",
    "benchmark/agent_runs/formal_v2_qwen3.7_plus",
    "benchmark/agent_runs/formal_v2_qwen3.6_flash_2026_04_16",
    "benchmark/agent_runs/formal_v2_deepseek_v4_pro",
    "benchmark/agent_runs/formal_v2_deepseek_v4_flash",
]

MODEL_SUITE_OUTPUT_RELPATH = "results/formal_model_suite_evaluation"
FINAL_SUITE_OUTPUT_RELPATH = "results/formal_final_all_experiments_evaluation"

# 固定数值容差：所有实验统一使用，避免按题或按模型调整阈值。
STRICT_NUMERIC_ABS_TOL = 1e-3
STRICT_NUMERIC_REL_TOL = 1e-3
STRICT_P_VALUE_ABS_TOL = 1e-10
STRICT_DISTANCE_ABS_TOL_M = 10.0
STRICT_AREA_ABS_TOL_M2 = 10_000.0
STRICT_CONFIDENCE_INTERVAL_ABS_TOL = 1e-3
STRICT_BOUNDING_BOX_ABS_TOL_DEGREES = 1e-6


def normalize_difficulty_label(label: Any) -> str:
    """
    基准难度档位：原 very_hard 已并入 hard，与 benchmark/dev*.yaml 一致。
    """
    if label is None:
        return "unknown"
    s = str(label).strip()
    if s == "very_hard":
        return "hard"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 1.  标量与键
# ══════════════════════════════════════════════════════════════════════════════

def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  严格结构化主指标（不使用正文兜底、不使用跨记录叶子搜索）
# ══════════════════════════════════════════════════════════════════════════════

_STRICT_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([^0-9\s]*)\s*$"
)
_RANKING_QUESTION_RE = re.compile(
    r"(?:\btop\s*\d*\b|排名|排行|排序|最高|最低|最大|最小|前\s*\d+\s*(?:个|名|项)?)",
    re.IGNORECASE,
)

_UNIT_RULES: dict[str, tuple[str, float]] = {
    "%": ("ratio", 0.01),
    "percent": ("ratio", 0.01),
    "percentage": ("ratio", 0.01),
    "百分比": ("ratio", 0.01),
    "m": ("length_m", 1.0),
    "米": ("length_m", 1.0),
    "km": ("length_m", 1000.0),
    "公里": ("length_m", 1000.0),
    "cm": ("length_m", 0.01),
    "厘米": ("length_m", 0.01),
    "m2": ("area_m2", 1.0),
    "m²": ("area_m2", 1.0),
    "平方米": ("area_m2", 1.0),
    "km2": ("area_m2", 1_000_000.0),
    "km²": ("area_m2", 1_000_000.0),
    "平方公里": ("area_m2", 1_000_000.0),
    "ha": ("area_m2", 10_000.0),
    "公顷": ("area_m2", 10_000.0),
    "μg/m³": ("concentration_ug_m3", 1.0),
    "μg/m3": ("concentration_ug_m3", 1.0),
    "ug/m3": ("concentration_ug_m3", 1.0),
    "m/s": ("speed_m_s", 1.0),
    "米/秒": ("speed_m_s", 1.0),
}


def _strict_norm_string(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return " ".join(text.strip().casefold().split())


def _strict_norm_key(value: Any) -> str:
    return _strict_norm_string(value)


def _parse_number_and_unit(value: Any) -> tuple[float, str | None, float] | None:
    """返回（换算后的基准值、规范单位、原始数值）；非纯数值/单位标量返回 None。"""
    if _is_numeric(value):
        raw = float(value)
        return raw, None, raw
    if not isinstance(value, str):
        return None
    match = _STRICT_NUMBER_RE.match(value)
    if not match or not match.group(1):
        return None
    try:
        raw = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit_raw = _strict_norm_string(match.group(2))
    if not unit_raw:
        return raw, None, raw
    rule = _UNIT_RULES.get(unit_raw)
    if rule is None:
        return None
    canonical, factor = rule
    return raw * factor, canonical, raw


def _parse_plain_integer_exact(value: Any) -> int | None:
    """Parse unitless integer identifiers without routing through float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and abs(value) <= 2**53:
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if re.fullmatch(r"[+-]?\d+", text):
            try:
                return int(text)
            except ValueError:
                return None
        if re.fullmatch(r"[+-]?\d+\.0+", text):
            try:
                dec = Decimal(text)
            except InvalidOperation:
                return None
            if dec == dec.to_integral_value():
                return int(dec)
    return None


def _is_integer_like_value(value: Any, raw_number: float) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        return re.fullmatch(r"[+-]?\d+", value.strip().replace(",", "")) is not None
    return raw_number.is_integer()


def _strict_scalar_equal(expected: Any, actual: Any) -> bool:
    """标量严格等价：字符串完整相等；数值按固定容差与常见单位归一化。"""
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual

    exp_int = _parse_plain_integer_exact(expected)
    act_int = _parse_plain_integer_exact(actual)
    if exp_int is not None and act_int is not None:
        return exp_int == act_int

    exp_num = _parse_number_and_unit(expected)
    act_num = _parse_number_and_unit(actual)
    if exp_num is not None and act_num is not None:
        exp_base, exp_unit, exp_raw = exp_num
        act_base, act_unit, act_raw = act_num
        if (
            exp_unit is None
            and act_unit is None
            and _is_integer_like_value(expected, exp_raw)
            and _is_integer_like_value(actual, act_raw)
        ):
            return exp_raw == act_raw
        if exp_unit is not None and act_unit is not None:
            if exp_unit != act_unit:
                return False
            return math.isclose(
                exp_base,
                act_base,
                rel_tol=STRICT_NUMERIC_REL_TOL,
                abs_tol=STRICT_NUMERIC_ABS_TOL,
            )
        if exp_unit is None and act_unit is None:
            return math.isclose(
                exp_raw,
                act_raw,
                rel_tol=STRICT_NUMERIC_REL_TOL,
                abs_tol=STRICT_NUMERIC_ABS_TOL,
            )
        # Gold 常不显式写单位：先比较显示数值；百分数再兼容 0.1 ↔ 10%。
        if math.isclose(
            exp_raw,
            act_raw,
            rel_tol=STRICT_NUMERIC_REL_TOL,
            abs_tol=STRICT_NUMERIC_ABS_TOL,
        ):
            return True
        if exp_unit == "ratio" or act_unit == "ratio":
            return math.isclose(
                exp_base if exp_unit == "ratio" else exp_raw,
                act_base if act_unit == "ratio" else act_raw,
                rel_tol=STRICT_NUMERIC_REL_TOL,
                abs_tol=STRICT_NUMERIC_ABS_TOL,
            )
        return False

    if isinstance(expected, str) and isinstance(actual, str):
        return (
            _strict_norm_string(expected) == _strict_norm_string(actual)
            or _semantic_label_equal(expected, actual)
        )
    return type(expected) is type(actual) and expected == actual


def _semantic_label_equal(expected: Any, actual: Any) -> bool:
    def _parse_bound_value(text: str, token: str) -> float | None:
        try:
            return float(token)
        except ValueError:
            return None

    def _elevation_band_label(value: Any) -> str | None:
        text = _strict_norm_string(value)
        compact = re.sub(r"\s+", "", text).replace("—", "-").replace("–", "-").replace("至", "-").replace("到", "-")
        if compact in {"low_lt_100", "lt_100", "below_100"}:
            return "elev_low_lt_100"
        if compact in {"high_ge_500", "ge_500", "above_500"}:
            return "elev_high_ge_500"
        if compact in {"mid_100_500", "100_500", "100m_500m", "100-500m"}:
            return "elev_mid_100_500"

        range_match = re.search(
            r"(?<!\d)(100(?:\.0+)?)\s*(?:m|米)?\s*[-~]\s*(500(?:\.0+)?)\s*(?:m|米)?(?!\d)",
            compact,
        )
        if range_match:
            return "elev_mid_100_500"
        between_match = re.search(
            r"(?:above|over|greaterthan|>=|≥|大于|高于|以上)(100(?:\.0+)?)(?:m|米)?.*"
            r"(?:below|under|lessthan|<|＜|低于|小于)(500(?:\.0+)?)(?:m|米)?",
            compact,
        )
        if between_match:
            return "elev_mid_100_500"
        chinese_between = re.search(
            r"(100(?:\.0+)?)(?:m|米)?(?:以上|及以上|到|至|-).*"
            r"(?:低于|小于|以下)(500(?:\.0+)?)(?:m|米)?",
            compact,
        )
        if chinese_between:
            return "elev_mid_100_500"

        lt_match = re.search(r"(?:<|＜|below|under|lessthan|低于|小于)(100(?:\.0+)?)(?:m|米)?(?!\d)", compact)
        if lt_match:
            return "elev_low_lt_100"
        ge_match = re.search(r"(?:>=|≥|above|over|greaterthan|高于|不低于|以上|及以上)(500(?:\.0+)?)(?:m|米)?(?!\d)", compact)
        suffix_ge_match = re.search(r"(?<!\d)(500(?:\.0+)?)(?:m|米)?(?:以上|及以上)", compact)
        if ge_match or suffix_ge_match:
            return "elev_high_ge_500"
        return None

    exp_label = _elevation_band_label(expected)
    return exp_label is not None and exp_label == _elevation_band_label(actual)


def _strict_key_lookup(key: Any, actual: dict) -> tuple[bool, Any]:
    """键仅允许 Unicode/大小写/空白归一化，不允许子串模糊匹配。"""
    expected_key = _strict_norm_key(key)
    for actual_key, actual_value in actual.items():
        if _strict_norm_key(actual_key) == expected_key:
            return True, actual_value
    return False, None


_FIELD_ALIAS_GROUPS: tuple[set[str], ...] = (
    {
        "shapename",
        "state",
        "state_name",
        "state_with_max_fluctuation",
        "州",
        "州名",
    },
    {"province", "province_name", "省", "省份"},
    {"city", "city_name", "fullname", "nearest_city", "城市", "城市名称"},
    {"county", "county_name", "县", "区县"},
    {"cell_id", "grid_id", "fishnet_id", "网格id", "网格_id", "网格编号"},
    {"asdf_id"},
    {"code", "city_code", "admin_code", "行政区划代码"},
    {"shapeid", "shape_id"},
    {"shapeiso", "shape_iso"},
    {"gqid"},
    {"cluster", "cluster_label", "cluster_id", "类别", "分类"},
    {"elevation_band", "elevation_interval", "elevation_range", "高程区间", "海拔区间"},
    {"pm25_mean", "pm2_5", "pm2.5", "annual_avg_pm2.5", "annual_avg_pm25", "年均pm2.5"},
    {"ndvi", "ndvi_mean", "avg_ndvi", "mean_ndvi", "平均ndvi"},
    {"total_ndvi", "ndvi_sum", "ndvi_total", "ndvi总和"},
    {"evi", "evi_mean", "avg_evi", "mean_evi", "平均evi"},
    {"evi_sum", "total_evi", "evi总和"},
    {"population", "pop", "人口"},
    {"pop_sum", "population_sum", "total_population", "人口总和", "总人口"},
    {"precipitation", "precip", "降水量"},
    {"precip_sum", "precipitation_sum", "total_precipitation", "降水总和"},
    {"avg_precip", "avg_precipitation", "mean_precipitation", "平均降水量"},
    {"temperature", "temp", "air_temp_mean", "era5_temp", "气温"},
    {"avg_temp", "avg_temperature", "mean_temperature", "平均气温"},
    {"soil_moisture", "平均土壤湿度", "土壤湿度"},
    {"built_surface", "urban_built_surface", "built_area", "建成区", "城市建设用地"},
    {"slope", "slope_mean", "平均坡度"},
    {"elevation", "elevation_mean", "高程", "海拔"},
    {"windspeed_mean", "wind_speed", "平均风速"},
    {"pvout_mean", "pvout", "光伏潜力"},
    {"xco2", "xco2_mean"},
    {"nightlight", "ntl", "ntl_mean", "夜间灯光"},
    {"count", "数量"},
    {"category_count", "类别数"},
    {"positive_category_count", "正值类别数"},
    {
        "unit_count",
        "filtered_unit_count",
        "urban_builtup_spatial_unit_count",
        "state_count",
        "states_count",
        "total_states",
        "num_states",
        "number_of_states",
        "空间单元数量",
        "筛选区域数量",
        "州数量",
        "州数",
        "高于平均值的州数量",
        "符合条件的州数量",
    },
    {
        "mean_abs_fluctuation",
        "avg_temp_diff",
        "max_avg_temp_diff",
        "max_annual_abs_temp_diff",
        "mean_abs_temp_diff",
        "mean_abs_monthly_temp_diff",
        "mean_abs_diff",
        "平均绝对温差",
        "平均月际温差",
    },
    {
        "ci95_lower",
        "lower_bound",
        "lower_bound_2.5%",
        "ci_lower",
        "ci_lower_2.5%",
        "2.5%",
        "2.5%分位数",
        "置信区间下界",
        "置信区间下限",
        "95%置信区间下界",
    },
    {
        "ci95_upper",
        "upper_bound",
        "upper_bound_97.5%",
        "ci_upper",
        "ci_upper_97.5%",
        "97.5%",
        "97.5%分位数",
        "置信区间上界",
        "置信区间上限",
        "95%置信区间上界",
    },
    {"class_2012", "from_type", "start_type", "start_landcover", "source_type", "起始类型"},
    {"class_2022", "to_type", "end_type", "end_landcover", "target_type", "结束类型"},
    {
        "grid_count",
        "filtered_grid_count",
        "selected_grid_count",
        "intersecting_grid_count",
        "transition_count",
        "transfer_count",
        "网格数",
        "网格数量",
        "筛选网格数",
        "相交网格数",
        "转移数量",
        "符合条件的网格数量",
    },
    {
        "extreme_bright_grids_count",
        "nightlight_outlier_count",
        "异常高值网格数量",
        "夜间灯光异常高值网格数量",
    },
    {
        "changed_twice_count",
        "two_stage_changed_grid_count",
        "两阶段均发生变化的网格数",
    },
    {
        "t_statistic",
        "t_value",
        "t_stat",
        "t值",
        "t统计量",
        "配对t统计量",
    },
    {
        "temp_z",
        "temperature_z",
        "z_temperature",
        "temperature_zscore",
        "standardized_temperature",
        "标准化气温",
        "气温z_score",
    },
    {
        "effective_area_km2",
        "effective_area_sum_km2",
        "杭州市内网格area_ratio总和",
    },
    {
        "total_changed_area_km2",
        "changed_effective_area_km2",
        "total_effective_area_sqkm",
        "total_effective_area_sq_km",
        "变化网格有效面积总和",
    },
    {
        "cv",
        "coefficient_of_variation",
        "pop_cv",
        "population_cv",
        "变异系数",
        "人口变异系数",
    },
    {"pm_rank", "pm25_rank", "rank_pm25", "pm2_5_rank"},
    {"urban_rank", "rank_urban", "builtup_rank", "urban_proportion_rank"},
    {"pop_rank", "population_rank", "rank_population"},
    {"light_rank", "nightlight_rank", "ntl_rank", "rank_nightlight"},
    {"built_rank", "built_surface_rank", "rank_built_surface"},
    {"mean_rank", "avg_rank", "average_rank", "rank_mean", "avg_rank_score"},
    {"correlation", "pearson_correlation", "pearson_r", "r", "相关系数", "皮尔逊相关系数"},
    {"minx", "min_x", "min_lon", "min_longitude", "bbox_minx"},
    {"miny", "min_y", "min_lat", "min_latitude", "bbox_miny"},
    {"maxx", "max_x", "max_lon", "max_longitude", "bbox_maxx"},
    {"maxy", "max_y", "max_lat", "max_latitude", "bbox_maxy"},
    {"value", "answer", "result", "scalar"},
)


def _compact_key(value: Any) -> str:
    compact = _strict_norm_key(value).replace(" ", "_").replace("-", "_")
    compact = compact.replace("pm2_5", "pm25").replace("pm2.5", "pm25")
    return compact


def _field_alias_group(norm_key: str) -> int | None:
    compact = _compact_key(norm_key)
    for idx, group in enumerate(_FIELD_ALIAS_GROUPS):
        normalized_group = {
            _compact_key(member)
            for member in group
        }
        if compact in normalized_group:
            return idx
    return None


def _field_stat_semantics(key: Any) -> str | None:
    compact = _compact_key(key)
    # The benchmark column ``era5_temp`` is an annual mean even though its
    # physical column name does not contain an explicit ``mean`` suffix.
    if compact == "era5_temp":
        return "mean"
    if re.fullmatch(r"(?:max_|min_|target_)?year|年份|最高年份|最低年份", compact):
        return None
    if compact in {"p", "p_value", "p值"} or compact.endswith("_p_value") or compact.endswith("_pvalue"):
        return "p_value"
    if "feature" in compact or "特征" in compact or "因子" in compact:
        return None
    if "z_score" in compact or "zscore" in compact or compact.endswith("_z") or "_z_" in compact:
        return "zscore"
    if "variance_explained" in compact or "explained_variance" in compact or "方差解释率" in compact:
        return "ratio"
    if "range" in compact or "极差" in compact:
        return "range"
    if "gini" in compact or "基尼" in compact:
        return "gini"
    if (
        "correlation" in compact
        or "相关系数" in compact
        or "pearson" in compact
        or "spearman" in compact
        or "rho" in compact
        or compact in {"r", "rho"}
    ):
        return "correlation"
    if "median" in compact or "中位数" in compact:
        return "median"
    if re.search(r"(^|_)(count|counts|num|number)($|_)", compact) or "数量" in compact or "个数" in compact or "类别数" in compact:
        return "count"
    if "cagr" in compact or "compound_annual_growth" in compact:
        return "ratio"
    # ``proportion_increase`` is a change *of* a ratio, not a static ratio.
    # Preserve the operation semantics so it can only match another change
    # field under the record-level disambiguation rules below.
    if any(token in compact for token in ("delta", "diff", "difference", "change", "growth", "increment", "increase", "变化", "增量", "增长")) and any(
        token in compact for token in ("proportion", "ratio", "rate", "percent", "percentage", "占比", "比例", "百分比")
    ):
        return "change"
    if (
        "proportion" in compact
        or "ratio" in compact
        or "rate" in compact
        or "percent" in compact
        or "percentage" in compact
        or "占比" in compact
        or "比例" in compact
        or "百分比" in compact
    ):
        return "ratio"
    if (
        "delta" in compact
        or "diff" in compact
        or "difference" in compact
        or "change" in compact
        or "growth" in compact
        or "increment" in compact
        or "increase" in compact
        or "amplitude" in compact
        or "变化" in compact
        or "增量" in compact
        or "增长" in compact
        or "温差" in compact
    ):
        return "change"
    if (
        "statistic" in compact
        or compact.endswith("_value")
        or compact.endswith("_stat")
        or "统计量" in compact
        or "chi_square" in compact
        or "chi2" in compact
        or "卡方" in compact
    ):
        return "statistic"
    if re.search(r"(^|_)(avg|mean|average)($|_)", compact) or "平均" in compact or "均值" in compact or "年均" in compact:
        return "mean"
    if re.search(r"(^|_)(sum|total)($|_)", compact) or "总和" in compact or "总数" in compact or "总" in compact:
        return "sum"
    if re.search(r"(^|_)(max|maximum)($|_)", compact) or "最大" in compact or "最高" in compact:
        return "max"
    if re.search(r"(^|_)(min|minimum)($|_)", compact) or "最小" in compact or "最低" in compact:
        return "min"
    return None


def _field_base_metric(key: Any) -> str | None:
    compact = _compact_key(key)
    if any(token in compact for token in ("outlier", "extreme", "异常")):
        if any(token in compact for token in ("nightlight", "ntl", "bright", "夜间灯光")):
            return "nightlight_outlier"
        return "outlier"
    if re.search(r"(^|_)pc\s*1($|_)", compact) or "pc1" in compact:
        return "pc1"
    if re.search(r"(^|_)pc\s*2($|_)", compact) or "pc2" in compact:
        return "pc2"
    if re.search(r"(^|_)pc\s*3($|_)", compact) or "pc3" in compact:
        return "pc3"
    if "cell_id" in compact or "fishnet_id" in compact or "网格id" in compact or "网格_id" in compact:
        return "cell_id"
    if "asdf_id" in compact:
        return "asdf_id"
    if "hii" in compact:
        return "hii"
    if re.fullmatch(r"(?:max_|min_|target_)?year|年份|最高年份|最低年份", compact):
        return "year"
    if "chi_square" in compact or "chi2" in compact or "卡方" in compact:
        return "chi_square"
    if compact in {"dof", "df", "degrees_of_freedom", "自由度"} or "degree_of_freedom" in compact:
        return "degrees_of_freedom"
    if "cramers_v" in compact or "cramer" in compact:
        return "cramers_v"
    if compact in {"p", "p_value", "p值"} or compact.endswith("_p_value") or compact.endswith("_pvalue"):
        return "p_value"
    if "u_statistic" in compact or "u统计" in compact or "mann" in compact:
        return "u_test"
    if "silhouette" in compact or "轮廓系数" in compact:
        k_match = re.search(r"k[_-]?(\d+)", compact)
        return f"silhouette_k{k_match.group(1)}" if k_match else "silhouette"
    if "stability" in compact or "stable" in compact or "稳定性" in compact:
        return "stability"
    if "quartile" in compact or "四分位" in compact or "等频" in compact:
        return "elevation_quartile" if "elevation" in compact or "海拔" in compact else "quartile"
    if any(token in compact for token in ("elevation", "海拔", "高程")) and any(
        token in compact for token in ("group", "分组", "组别")
    ):
        return "group_label"
    if compact in {"group", "group_name", "组名", "分组"}:
        return "group_label"
    if "cluster" in compact or "聚类" in compact:
        return "cluster"
    token_map = {
        "population": ("population", "pop", "人口"),
        "ndvi": ("ndvi",),
        "evi": ("evi",),
        "precipitation": ("precipitation", "precip", "降水"),
        "temperature": ("temperature", "temp", "气温", "era5_temp", "air_temp", "lst_day", "lst_night", "lst"),
        "pm25": ("pm25", "pm2_5", "pm2.5"),
        "built_surface": ("built_surface", "built_area", "建成区", "城市建设用地"),
        "nightlight": ("nightlight", "ntl", "夜间灯光"),
        "soil_moisture": ("soil_moisture", "土壤湿度"),
        "slope": ("slope", "坡度"),
        "elevation": ("elevation", "dem", "高程", "海拔"),
        "xco2": ("xco2",),
        "windspeed": ("windspeed", "wind_speed", "wind", "风速"),
        "pvout": ("pvout", "solar", "pv", "光伏"),
        "area": ("area", "面积"),
        "ratio": ("ratio", "proportion", "占比", "比例"),
        "rank": ("rank", "排名"),
        "correlation": ("correlation", "相关系数", "pearson", "spearman", "rho"),
        "gini": ("gini", "基尼"),
        "shapiro_w": ("shapiro", "w_statistic", "w_stat", "w统计"),
        "t_test": ("t_statistic", "t_value", "t_stat", "t值"),
        "u_test": ("u_statistic", "u_value", "u_stat", "u检验"),
        "p_value": ("p_value", "p值"),
        "r_squared": ("r_squared", "r2", "r^2"),
        "feature": ("feature", "特征", "因子"),
        "outlier": ("outlier", "extreme", "异常"),
        "cluster": ("cluster", "聚类", "类别"),
        "grid": ("grid", "网格"),
        "cagr": ("cagr", "compound_annual_growth"),
        "changed": ("changed", "change", "变化", "发生变化"),
        "city": ("city", "城市"),
        "state": ("state", "州"),
        "border": ("border", "边界", "接壤"),
        "nearest_city": ("nearest_city", "最近城市"),
    }
    for base, tokens in token_map.items():
        if any(token in compact for token in tokens):
            return base
    return None


def _field_semantics_compatible(expected_key: Any, actual_key: Any) -> bool:
    exp_norm = _strict_norm_key(expected_key)
    act_norm = _strict_norm_key(actual_key)
    if exp_norm == act_norm:
        return True
    if _distance_unit_from_key(expected_key) is not None or _distance_unit_from_key(actual_key) is not None:
        return (
            _distance_unit_from_key(expected_key) is not None
            and _distance_unit_from_key(actual_key) is not None
            and _field_base_metric(expected_key) == _field_base_metric(actual_key)
        )
    exp_base = _field_base_metric(expected_key)
    act_base = _field_base_metric(actual_key)
    exp_group = _field_alias_group(exp_norm)
    act_group = _field_alias_group(act_norm)
    if exp_group is not None and exp_group == act_group:
        return True
    if exp_base is not None and exp_base == act_base and exp_base in {
        "cell_id",
        "asdf_id",
        "year",
        "chi_square",
        "degrees_of_freedom",
        "cramers_v",
        "p_value",
        "u_test",
        "silhouette_k2",
        "silhouette_k3",
    }:
        return True
    if {exp_base, act_base} == {"elevation_quartile", "group_label"}:
        return True
    if _is_rate_like_key(expected_key) and _is_rate_like_key(actual_key):
        return exp_base == act_base
    if exp_base == act_base == "area" and "effective" in _compact_key(expected_key) and "effective" in _compact_key(actual_key):
        return True
    exp_stat = _field_stat_semantics(expected_key)
    act_stat = _field_stat_semantics(actual_key)
    if exp_stat != act_stat:
        return False
    if exp_base is not None and exp_base == act_base:
        return True
    if exp_stat == "correlation" and (exp_base == "correlation" or act_base == "correlation"):
        return True
    if exp_stat in {"range", "ratio", "change", "zscore"} and (exp_base is None or act_base is None):
        return True
    if exp_stat == "ratio" and {exp_base, act_base} in (
        {"population", "cagr"},
        {"pc1", "ratio"},
        {"pc2", "ratio"},
        {"pc3", "ratio"},
    ):
        return True
    return exp_group is not None and exp_group == act_group


def _is_rate_like_key(key: Any) -> bool:
    norm = _strict_norm_key(key)
    return any(
        token in norm
        for token in (
            "rate",
            "ratio",
            "pct",
            "percent",
            "percentage",
            "growth",
            "increase",
            "比例",
            "占比",
            "百分比",
            "增幅",
            "增长",
        )
    )


def _distance_unit_from_key(key: Any) -> str | None:
    norm = _strict_norm_key(key)
    if not any(token in norm for token in ("distance", "dist", "距离", "距")):
        return None
    if "km" in norm or "公里" in norm or "千米" in norm:
        return "km"
    if re.search(r"(^|[_\s])m($|[_\s])", norm) or "米" in norm:
        return "m"
    return None


def _area_unit_from_key(key: Any) -> str | None:
    norm = _strict_norm_key(key).replace("²", "2")
    if not any(token in norm for token in ("area", "面积", "hull")):
        return None
    if any(token in norm for token in ("km2", "km^2", "sqkm", "sq_km", "square_km", "平方公里", "平方千米")):
        return "km2"
    if any(token in norm for token in ("m2", "m^2", "sqm", "sq_m", "square_m", "平方米", "平方米")):
        return "m2"
    return None


def _strict_distance_equal_by_key(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
) -> bool:
    exp_unit = _distance_unit_from_key(expected_key)
    act_unit = _distance_unit_from_key(actual_key)
    if exp_unit is None or act_unit is None:
        return False
    if _field_base_metric(expected_key) != _field_base_metric(actual_key):
        return False
    exp_num = _parse_number_and_unit(expected_value)
    act_num = _parse_number_and_unit(actual_value)
    if exp_num is None or act_num is None:
        return False
    _, exp_value_unit, exp_raw = exp_num
    _, act_value_unit, act_raw = act_num
    if exp_value_unit is not None or act_value_unit is not None:
        return False

    exp_m = exp_raw if exp_unit == "m" else exp_raw * 1000.0
    act_m = act_raw if act_unit == "m" else act_raw * 1000.0
    return math.isclose(exp_m, act_m, rel_tol=0.0, abs_tol=STRICT_DISTANCE_ABS_TOL_M)


def _strict_area_equal_by_key(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
) -> bool:
    exp_unit = _area_unit_from_key(expected_key)
    act_unit = _area_unit_from_key(actual_key)
    if exp_unit is None or act_unit is None:
        return False
    if _field_base_metric(expected_key) != "area" or _field_base_metric(actual_key) != "area":
        return False
    explicit_alias_match = (
        _field_alias_group(_strict_norm_key(expected_key)) is not None
        and _field_alias_group(_strict_norm_key(expected_key))
        == _field_alias_group(_strict_norm_key(actual_key))
    )
    if (
        _field_stat_semantics(expected_key) != _field_stat_semantics(actual_key)
        and not (
            "effective" in _compact_key(expected_key)
            and "effective" in _compact_key(actual_key)
        )
        and not explicit_alias_match
    ):
        return False
    exp_num = _parse_number_and_unit(expected_value)
    act_num = _parse_number_and_unit(actual_value)
    if exp_num is None or act_num is None:
        return False
    _, exp_value_unit, exp_raw = exp_num
    _, act_value_unit, act_raw = act_num
    if exp_value_unit is not None or act_value_unit is not None:
        return False
    exp_m2 = exp_raw if exp_unit == "m2" else exp_raw * 1_000_000.0
    act_m2 = act_raw if act_unit == "m2" else act_raw * 1_000_000.0
    return math.isclose(exp_m2, act_m2, rel_tol=0.0, abs_tol=STRICT_AREA_ABS_TOL_M2)


def _parse_quartile_ordinal(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    parsed = _parse_plain_integer_exact(value)
    if parsed in {1, 2, 3, 4}:
        return parsed
    text = _strict_norm_string(value)
    compact = text.replace(" ", "")
    match = re.search(r"(?:^|[^0-9])q([1-4])(?:[^0-9]|$)", compact)
    if match:
        return int(match.group(1))
    chinese_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
    }
    for token, ordinal in chinese_map.items():
        if f"第{token}" in compact or f"{token}分位" in compact:
            return ordinal
    return None


def _strict_quartile_label_equal(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
) -> bool:
    if { _field_base_metric(expected_key), _field_base_metric(actual_key) } != {
        "elevation_quartile",
        "group_label",
    }:
        return False
    exp_ord = _parse_quartile_ordinal(expected_value)
    act_ord = _parse_quartile_ordinal(actual_value)
    return exp_ord is not None and exp_ord == act_ord


def _strict_keyed_scalar_equal(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
) -> bool:
    keys_compatible = _field_semantics_compatible(expected_key, actual_key)
    exp_alias_group = _field_alias_group(_strict_norm_key(expected_key))
    act_alias_group = _field_alias_group(_strict_norm_key(actual_key))
    explicit_alias_match = exp_alias_group is not None and exp_alias_group == act_alias_group
    if (
        _distance_unit_from_key(expected_key) is not None
        or _distance_unit_from_key(actual_key) is not None
    ):
        return _strict_distance_equal_by_key(
            expected_key,
            expected_value,
            actual_key,
            actual_value,
        )
    if _area_unit_from_key(expected_key) is not None or _area_unit_from_key(actual_key) is not None:
        # A few database-derived effective-area fields encode km2 in the
        # schema/alias rather than in the emitted display key.  Only an exact
        # explicit alias group may bridge that missing key-level unit.
        if (
            keys_compatible
            and explicit_alias_match
            and (
                _area_unit_from_key(expected_key) is None
                or _area_unit_from_key(actual_key) is None
            )
        ):
            return _strict_scalar_equal(expected_value, actual_value)
        return _strict_area_equal_by_key(
            expected_key,
            expected_value,
            actual_key,
            actual_value,
        )
    if keys_compatible and _strict_quartile_label_equal(
        expected_key,
        expected_value,
        actual_key,
        actual_value,
    ):
        return True
    if (
        keys_compatible
        and _field_base_metric(expected_key) == "p_value"
        and _field_base_metric(actual_key) == "p_value"
    ):
        exp_num = _parse_number_and_unit(expected_value)
        act_num = _parse_number_and_unit(actual_value)
        if exp_num is None or act_num is None:
            return False
        if exp_num[1] is not None or act_num[1] is not None:
            return False
        return math.isclose(
            exp_num[2],
            act_num[2],
            rel_tol=STRICT_NUMERIC_REL_TOL,
            abs_tol=STRICT_P_VALUE_ABS_TOL,
        )
    if (
        keys_compatible
        and _field_base_metric(expected_key) == "feature"
        and _field_base_metric(actual_key) == "feature"
    ):
        expected_feature = _field_base_metric(expected_value)
        actual_feature = _field_base_metric(actual_value)
        if expected_feature is not None and expected_feature == actual_feature:
            return True
    if keys_compatible and _strict_value_match(expected_value, actual_value):
        return True
    if not (keys_compatible and _is_rate_like_key(expected_key) and _is_rate_like_key(actual_key)):
        return False
    exp_num = _parse_number_and_unit(expected_value)
    act_num = _parse_number_and_unit(actual_value)
    if exp_num is None or act_num is None:
        return False
    _, exp_unit, exp_raw = exp_num
    _, act_unit, act_raw = act_num
    if exp_unit is not None or act_unit is not None:
        return False
    return (
        math.isclose(exp_raw * 100.0, act_raw, rel_tol=STRICT_NUMERIC_REL_TOL, abs_tol=STRICT_NUMERIC_ABS_TOL)
        or math.isclose(exp_raw, act_raw * 100.0, rel_tol=STRICT_NUMERIC_REL_TOL, abs_tol=STRICT_NUMERIC_ABS_TOL)
    )


def _strict_keyed_value_equal(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
) -> bool:
    """Compare values only after field semantics are compatible.

    Scalar fields use unit/rate-aware comparison.  Nested dict/list values are
    allowed for semantically equivalent wrapper keys such as
    ``cluster_counts`` ↔ ``聚类数量统计`` or ``top_3_grids`` ↔ ``稳定性指数TOP3``;
    no cross-key leaf search is performed unless the enclosing keys already
    describe the same metric.
    """
    if not _field_semantics_compatible(expected_key, actual_key):
        return False
    if _is_scalar_payload(expected_value) and _is_scalar_payload(actual_value):
        return _strict_keyed_scalar_equal(expected_key, expected_value, actual_key, actual_value)
    return _strict_value_match(expected_value, actual_value)


def _scalar_entity_matches_record(expected: Any, actual: dict) -> bool:
    return any(
        _strict_scalar_equal(expected, value)
        for key, value in actual.items()
        if _is_scalar_payload(value) and _is_entity_like_key(key)
    )


def _strict_list_item_match(
    expected: Any,
    actual: Any,
    *,
    answer_fields: Iterable[Any] | None = None,
    tie_candidates: Any = None,
) -> bool:
    if _is_scalar_payload(expected) and isinstance(actual, dict):
        return _scalar_entity_matches_record(expected, actual)
    return _strict_value_match(
        expected,
        actual,
        answer_fields=answer_fields,
        tie_candidates=tie_candidates,
    )


def _question_allows_subset_record(question: str) -> bool:
    text = _strict_norm_string(question)
    compact = text.replace(" ", "")
    if any(token in compact for token in ("各州", "各城市", "各网格", "每个", "所有", "全部")):
        return True
    if "并列出" in compact or "并输出" in compact or "列出" in compact:
        return True
    return False


def _nested_record_match_exists(expected: dict, actual: Any) -> bool:
    if isinstance(actual, dict):
        if _strict_record_match(expected, actual):
            return True
        return any(_nested_record_match_exists(expected, value) for value in actual.values())
    if isinstance(actual, list):
        return any(_nested_record_match_exists(expected, item) for item in actual)
    return False


def _maximum_bipartite_matches(
    expected_items: list,
    actual_items: list,
    *,
    answer_fields: Iterable[Any] | None = None,
    tie_candidates: Any = None,
) -> int:
    """按严格项目匹配求最大双射，禁止一个实际项重复覆盖多个期望项。"""
    adjacency: list[list[int]] = [
        [
            j
            for j, actual in enumerate(actual_items)
            if _strict_list_item_match(
                expected,
                actual,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            )
        ]
        for expected in expected_items
    ]
    actual_to_expected: dict[int, int] = {}

    def _augment(exp_idx: int, visited: set[int]) -> bool:
        for act_idx in adjacency[exp_idx]:
            if act_idx in visited:
                continue
            visited.add(act_idx)
            if act_idx not in actual_to_expected or _augment(actual_to_expected[act_idx], visited):
                actual_to_expected[act_idx] = exp_idx
                return True
        return False

    return sum(1 for i in range(len(expected_items)) if _augment(i, set()))


def _project_expected_record(
    expected: dict,
    answer_fields: Iterable[Any] | None,
) -> dict:
    """Apply benchmark-declared output fields without weakening unannotated records."""
    fields = [field for field in (answer_fields or []) if field is not None]
    if not fields:
        return expected
    projected = {
        key: value
        for key, value in expected.items()
        if any(
            _compact_key(key) == _compact_key(field)
            or _field_semantics_compatible(key, field)
            for field in fields
        )
    }
    return projected or expected


def _record_has_transition_endpoints(record: dict) -> bool:
    groups = {
        _field_alias_group(_strict_norm_key(key))
        for key in record
    }
    source_group = _field_alias_group("class_2012")
    target_group = _field_alias_group("class_2022")
    return source_group in groups and target_group in groups


def _contextual_transition_count_match(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
    expected_record: dict,
    actual_record: dict,
) -> bool:
    """Allow generic ``count`` only inside a validated transition row."""
    if _field_alias_group(_strict_norm_key(expected_key)) != _field_alias_group("grid_count"):
        return False
    if _compact_key(actual_key) not in {"count", "数量"}:
        return False
    if not (
        _record_has_transition_endpoints(expected_record)
        and _record_has_transition_endpoints(actual_record)
    ):
        return False
    return _strict_scalar_equal(expected_value, actual_value)


def _contextual_unique_change_match(
    expected_key: Any,
    expected_value: Any,
    actual_key: Any,
    actual_value: Any,
    expected_record: dict,
    actual_record: dict,
) -> bool:
    """Match a generic change column only when the row disambiguates it.

    Generated SQL often aliases the only requested delta as ``increase`` or
    ``proportion_increase``.  Such a key must not globally become an alias of
    every domain metric.  We therefore require exactly one change-valued field
    on each side plus a separately matching entity anchor in the same record.
    """
    if _field_stat_semantics(expected_key) != "change" or _field_stat_semantics(actual_key) != "change":
        return False
    expected_changes = [
        key for key in expected_record if _field_stat_semantics(key) == "change"
    ]
    actual_changes = [
        key for key in actual_record if _field_stat_semantics(key) == "change"
    ]
    if len(expected_changes) != 1 or len(actual_changes) != 1:
        return False
    entity_anchor = any(
        _strict_keyed_value_equal(exp_key, exp_value, act_key, act_value)
        for exp_key, exp_value in expected_record.items()
        if _is_entity_like_key(exp_key) and _is_scalar_payload(exp_value)
        for act_key, act_value in actual_record.items()
        if _is_entity_like_key(act_key) and _is_scalar_payload(act_value)
    )
    return entity_anchor and _strict_scalar_equal(expected_value, actual_value)


def _compound_labeled_count_record_match(expected: dict, actual: dict) -> bool:
    """Accept ``杭州市落入网格数量: 1554`` as a labelled entity-count pair."""
    if len(actual) != 1 or len(expected) != 2:
        return False
    entity_items = [
        (key, value)
        for key, value in expected.items()
        if _is_entity_like_key(key) and isinstance(value, str)
    ]
    count_items = [
        (key, value)
        for key, value in expected.items()
        if _field_stat_semantics(key) == "count" and _is_scalar_payload(value)
    ]
    if len(entity_items) != 1 or len(count_items) != 1:
        return False
    actual_key, actual_value = next(iter(actual.items()))
    if _field_stat_semantics(actual_key) != "count" or not _is_scalar_payload(actual_value):
        return False
    entity_text = _strict_norm_string(entity_items[0][1]).replace(" ", "")
    actual_label = _strict_norm_string(actual_key).replace(" ", "")
    return bool(entity_text and entity_text in actual_label) and _strict_scalar_equal(
        count_items[0][1], actual_value
    )


def _strict_record_match(
    expected: dict,
    actual: dict,
    *,
    answer_fields: Iterable[Any] | None = None,
) -> bool:
    """同一记录内逐键匹配；别名键需属于显式语义组，不得跨字段仅按同值兜底。

    若 actual 存在同名键，则该键的值必须正确，不能用其它列的同值兜底；仅对缺失的
    Gold 键（例如 shapeName ↔ state）使用显式字段别名、距离或比例语义匹配。
    """
    expected = _project_expected_record(expected, answer_fields)
    if _compound_labeled_count_record_match(expected, actual):
        return True
    unmatched_items: list[tuple[Any, Any]] = []
    used_actual_keys: set[Any] = set()
    for expected_key, expected_value in expected.items():
        expected_norm = _strict_norm_key(expected_key)
        matched_key = None
        matched_value = None
        for actual_key, actual_value in actual.items():
            if _strict_norm_key(actual_key) == expected_norm:
                matched_key = actual_key
                matched_value = actual_value
                break
        if matched_key is None:
            for actual_key, actual_value in actual.items():
                if actual_key in used_actual_keys:
                    continue
                if _strict_keyed_value_equal(
                    expected_key,
                    expected_value,
                    actual_key,
                    actual_value,
                ) or _contextual_transition_count_match(
                    expected_key,
                    expected_value,
                    actual_key,
                    actual_value,
                    expected,
                    actual,
                ) or _contextual_unique_change_match(
                    expected_key,
                    expected_value,
                    actual_key,
                    actual_value,
                    expected,
                    actual,
                ):
                    matched_key = actual_key
                    matched_value = actual_value
                    break
        if matched_key is None:
            unmatched_items.append((expected_key, expected_value))
            continue
        if not (
            _strict_keyed_value_equal(expected_key, expected_value, matched_key, matched_value)
            or _contextual_transition_count_match(
                expected_key,
                expected_value,
                matched_key,
                matched_value,
                expected,
                actual,
            )
            or _contextual_unique_change_match(
                expected_key,
                expected_value,
                matched_key,
                matched_value,
                expected,
                actual,
            )
        ):
            return False
        used_actual_keys.add(matched_key)

    if not unmatched_items:
        return True
    if used_actual_keys and all(
        _is_optional_redundant_identifier_key(key)
        for key, _ in unmatched_items
    ):
        return True
    return False


def _is_scalar_payload(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _is_optional_redundant_identifier_key(key: Any) -> bool:
    """Identifiers often stored in Gold for disambiguation but not requested."""
    return _strict_norm_key(key) in {
        "code",
        "shapeid",
        "shapeiso",
        "gqid",
        "asdf_id",
    }


_SCALAR_ANSWER_KEYS = {
    "value",
    "answer",
    "result",
    "scalar",
}


def _normalized_answer_fields(answer_fields: Iterable[Any] | None = None) -> set[str]:
    return {_compact_key(field) for field in (answer_fields or []) if field is not None}


def _is_scalar_answer_key(key: Any, answer_fields: Iterable[Any] | None = None) -> bool:
    compact = _compact_key(key)
    if compact in _SCALAR_ANSWER_KEYS:
        return True
    return compact in _normalized_answer_fields(answer_fields)


def _record_contains_scalar_answer_value(
    record: dict,
    expected: Any,
    *,
    answer_fields: Iterable[Any] | None = None,
) -> bool:
    return any(
        _strict_scalar_equal(expected, value)
        for key, value in record.items()
        if _is_scalar_payload(value) and _is_scalar_answer_key(key, answer_fields)
    )


def _record_contains_values(record: dict, expected_values: list[Any]) -> bool:
    if not expected_values or any(not _is_scalar_payload(value) for value in expected_values):
        return False
    actual_values = [value for value in record.values() if _is_scalar_payload(value)]
    if len(actual_values) < len(expected_values):
        return False
    adjacency: list[list[int]] = [
        [j for j, actual in enumerate(actual_values) if _strict_scalar_equal(expected, actual)]
        for expected in expected_values
    ]
    actual_to_expected: dict[int, int] = {}

    def _augment(exp_idx: int, visited: set[int]) -> bool:
        for act_idx in adjacency[exp_idx]:
            if act_idx in visited:
                continue
            visited.add(act_idx)
            if act_idx not in actual_to_expected or _augment(actual_to_expected[act_idx], visited):
                actual_to_expected[act_idx] = exp_idx
                return True
        return False

    return sum(1 for i in range(len(expected_values)) if _augment(i, set())) == len(expected_values)


def _nested_record_contains_values(expected_values: list[Any], actual: Any) -> bool:
    if isinstance(actual, dict):
        if _record_contains_values(actual, expected_values):
            return True
        return any(_nested_record_contains_values(expected_values, value) for value in actual.values())
    if isinstance(actual, list):
        return any(_nested_record_contains_values(expected_values, item) for item in actual)
    return False


_RANK_META_KEYS = {
    "rank",
    "ranking",
    "order",
    "index",
    "position",
    "序号",
    "排名",
    "名次",
}

_ENTITY_LIKE_KEYS = {
    "shapename",
    "state",
    "province",
    "city",
    "county",
    "name",
    "code",
    "cell_id",
    "asdf_id",
    "gqid",
    "shapeid",
    "shapeiso",
    "geometry",
}


def _is_rank_meta_key(key: Any) -> bool:
    return _strict_norm_key(key) in _RANK_META_KEYS


def _is_entity_like_key(key: Any) -> bool:
    norm = _strict_norm_key(key)
    return (
        norm in _ENTITY_LIKE_KEYS
        or norm.endswith("_id")
        or norm.endswith(" id")
        or "name" in norm
        or "名称" in norm
    )


def _entity_field_type(key: Any) -> str | None:
    compact = _compact_key(key)
    if compact in {"shapename", "state", "state_name", "州", "州名"}:
        return "state"
    if compact in {"province", "province_name", "省", "省份"}:
        return "province"
    if compact in {"city", "city_name", "fullname", "nearest_city", "城市", "城市名称"}:
        return "city"
    if compact in {"county", "county_name", "县", "区县"}:
        return "county"
    if compact in {"cell_id", "grid_id", "fishnet_id", "网格id", "网格_id", "网格编号"}:
        return "cell_id"
    if compact in {"asdf_id"}:
        return "asdf_id"
    if compact in {"gqid"}:
        return "gqid"
    if compact in {"shapeid", "shape_id"}:
        return "shapeid"
    if compact in {"shapeiso", "shape_iso"}:
        return "shapeiso"
    if compact in {"code", "city_code", "admin_code", "行政区划代码"}:
        return "code"
    if compact.endswith("_id") or compact.endswith("id"):
        return compact
    return None


def _record_entity_signature(record: dict) -> tuple[str, str] | None:
    signatures: list[tuple[str, str]] = []
    for key, value in record.items():
        entity_type = _entity_field_type(key)
        if entity_type is None or not _is_scalar_payload(value):
            continue
        signatures.append((entity_type, _strict_norm_string(value)))
    if len(signatures) != 1:
        return None
    return signatures[0]


def _entity_signature_candidates(values: Any) -> set[tuple[str, str]]:
    if values is None:
        return set()
    if isinstance(values, dict):
        row_values = values.get("tie_candidates")
        if row_values is not None:
            return _entity_signature_candidates(row_values)
        sig = _record_entity_signature(values)
        return {sig} if sig is not None else set()
    if isinstance(values, list):
        signatures: set[tuple[str, str]] = set()
        for item in values:
            if isinstance(item, dict):
                sig = _record_entity_signature(item)
                if sig is not None:
                    signatures.add(sig)
            elif _is_scalar_payload(item):
                signatures.add(("scalar", _strict_norm_string(item)))
        return signatures
    if _is_scalar_payload(values):
        return {("scalar", _strict_norm_string(values))}
    return set()


def _rank_metric_signature(record: dict) -> list[tuple[tuple[Any, Any, Any], Any]]:
    """Numeric, non-entity fields that define a ranked tie group, with field semantics."""
    values: list[tuple[tuple[Any, Any, Any], Any]] = []
    for key, value in record.items():
        if _is_rank_meta_key(key) or _is_entity_like_key(key):
            continue
        if isinstance(value, (dict, list)):
            continue
        if _parse_number_and_unit(value) is not None:
            signature = (
                _field_base_metric(key),
                _field_stat_semantics(key),
                _field_alias_group(_strict_norm_key(key)),
            )
            values.append((signature, value))
    return values


def _numeric_value_multiset_match(
    expected_values: list[tuple[tuple[Any, Any, Any], Any]],
    actual_values: list[tuple[tuple[Any, Any, Any], Any]],
    *,
    allow_count_alias: bool = False,
) -> bool:
    if len(expected_values) != len(actual_values):
        return False
    adjacency: list[list[int]] = [
        [
            j
            for j, actual in enumerate(actual_values)
            if (
                expected[0] == actual[0]
                or (
                    allow_count_alias
                    and expected[0][1] == actual[0][1] == "count"
                )
            )
            and _strict_scalar_equal(expected[1], actual[1])
        ]
        for expected in expected_values
    ]
    actual_to_expected: dict[int, int] = {}

    def _augment(exp_idx: int, visited: set[int]) -> bool:
        for act_idx in adjacency[exp_idx]:
            if act_idx in visited:
                continue
            visited.add(act_idx)
            if act_idx not in actual_to_expected or _augment(actual_to_expected[act_idx], visited):
                actual_to_expected[act_idx] = exp_idx
                return True
        return False

    return sum(1 for i in range(len(expected_values)) if _augment(i, set())) == len(expected_values)


def _ranked_tie_truncated_match(expected: list, actual: list, *, tie_candidates: Any = None) -> bool:
    """Accept another valid Top-K sample only when the rank metric is fully tied.

    This handles benchmarks whose Gold stores one arbitrary K-row slice from a
    larger tie group, e.g. "top 5 states" where many states share the same
    maximum count.  It is intentionally limited to ranked lists with the same
    length and a non-empty numeric tie signature.
    """
    if len(expected) != len(actual) or not expected:
        return False
    if not all(isinstance(row, dict) for row in expected + actual):
        return False

    expected_metric_signature = _rank_metric_signature(expected[0])
    if not expected_metric_signature:
        return False
    expected_entities = [_record_entity_signature(row) for row in expected]
    actual_entities = [_record_entity_signature(row) for row in actual]
    if any(sig is None for sig in expected_entities + actual_entities):
        return False
    if len(set(actual_entities)) != len(actual_entities):
        return False
    expected_entity_types = {sig[0] for sig in expected_entities if sig is not None}
    actual_entity_types = {sig[0] for sig in actual_entities if sig is not None}
    if len(expected_entity_types) != 1 or expected_entity_types != actual_entity_types:
        return False
    allowed_entities = _entity_signature_candidates(tie_candidates) or {
        sig for sig in expected_entities if sig is not None
    }
    if not set(sig for sig in actual_entities if sig is not None).issubset(allowed_entities):
        return False
    for row in expected[1:]:
        if not _numeric_value_multiset_match(
            expected_metric_signature,
            _rank_metric_signature(row),
            allow_count_alias=tie_candidates is not None,
        ):
            return False

    for row in actual:
        actual_values = _rank_metric_signature(row)
        if not actual_values:
            return False
        if not _numeric_value_multiset_match(
            expected_metric_signature,
            actual_values,
            allow_count_alias=tie_candidates is not None,
        ):
            return False
    return True


def _same_record_contains_pair(record: dict, entity: Any, value: Any) -> bool:
    """兼容 Gold 的 {实体: 数值} 映射，但实体和值必须在同一实际记录中。"""
    scalar_values = [v for v in record.values() if not isinstance(v, (dict, list))]
    entity_ok = any(_strict_scalar_equal(entity, candidate) for candidate in scalar_values)
    value_ok = any(_strict_scalar_equal(value, candidate) for candidate in scalar_values)
    return entity_ok and value_ok


def _strict_flat_entity_map_vs_rows(expected: dict, actual: list) -> bool:
    if not expected or not actual or not all(isinstance(row, dict) for row in actual):
        return False
    if any(isinstance(v, (dict, list)) for v in expected.values()):
        return False
    pairs = list(expected.items())
    adjacency = [
        [j for j, row in enumerate(actual) if _same_record_contains_pair(row, entity, value)]
        for entity, value in pairs
    ]
    actual_to_expected: dict[int, int] = {}

    def _augment(exp_idx: int, visited: set[int]) -> bool:
        for act_idx in adjacency[exp_idx]:
            if act_idx in visited:
                continue
            visited.add(act_idx)
            if act_idx not in actual_to_expected or _augment(actual_to_expected[act_idx], visited):
                actual_to_expected[act_idx] = exp_idx
                return True
        return False

    matched = sum(1 for i in range(len(pairs)) if _augment(i, set()))
    return matched == len(pairs) == len(actual)


def _iter_nested_lists(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list):
                yield value
            yield from _iter_nested_lists(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_nested_lists(value)


def _actual_list_candidates(expected: list, actual: Any) -> list[list]:
    """Return structure-only list candidates for a list-shaped gold answer.

    The benchmark stores many one-row answers as ``[{"field": value}]``, while
    strict model outputs may validly use a scalar or wrap the table under a
    descriptive JSON key.  The synthetic ``[actual]`` candidate is only added
    for one-row Gold answers; if ``actual`` is already a multi-item list, that
    nested candidate still has to match the single Gold record and will not make
    extra wrong elements pass.  This helper only inspects structured payloads;
    it never parses natural-language fallback text.
    """
    candidates: list[list] = []
    seen: set[int] = set()

    def _add(candidate: Any) -> None:
        if isinstance(candidate, list) and id(candidate) not in seen:
            candidates.append(candidate)
            seen.add(id(candidate))

    _add(actual)
    if isinstance(actual, dict):
        for nested in _iter_nested_lists(actual):
            _add(nested)
    if len(expected) == 1 or (
        isinstance(actual, dict)
        and _pivoted_record_matches_list(expected, actual)
    ):
        candidates.append([actual])
    return candidates


def _label_token_for_pivot(value: Any) -> str:
    token = _strict_norm_string(value)
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", token)


def _label_tokens_for_pivot(value: Any) -> set[str]:
    """Return exact category-label tokens plus narrow bilingual aliases."""
    token = _label_token_for_pivot(value)
    aliases = {
        "summer": {"summer", "夏季"},
        "夏季": {"summer", "夏季"},
        "winter": {"winter", "冬季"},
        "冬季": {"winter", "冬季"},
        "spring": {"spring", "春季"},
        "春季": {"spring", "春季"},
        "autumn": {"autumn", "fall", "秋季"},
        "fall": {"autumn", "fall", "秋季"},
        "秋季": {"autumn", "fall", "秋季"},
    }
    return aliases.get(token, {token} if token else set())


def _pivoted_record_matches_list(expected: list, actual_record: dict) -> bool:
    """Match category-row Gold against a single wide/pivoted record.

    Example Gold:
      [{"season": "Summer", "avg_temp": 312}, {"season": "Winter", "avg_temp": 286}]
    may be returned as:
      {"summer_avg_lst": 312, "winter_avg_lst": 286}
    Labels must appear in the actual field name and numeric fields still use
    strict keyed semantic comparison.
    """
    if not expected or not all(isinstance(row, dict) for row in expected):
        return False
    if len(actual_record) <= 1:
        return False
    used_actual_keys: set[Any] = set()
    for row in expected:
        label_values = [
            value
            for key, value in row.items()
            if _is_scalar_payload(value)
            and isinstance(value, str)
            and not _is_rank_meta_key(key)
            and _parse_number_and_unit(value) is None
        ]
        metric_items = [
            (key, value)
            for key, value in row.items()
            if _is_scalar_payload(value)
            and _parse_number_and_unit(value) is not None
            and not _is_rank_meta_key(key)
        ]
        if len(label_values) != 1 or not metric_items:
            return False
        label_tokens = _label_tokens_for_pivot(label_values[0])
        if not label_tokens:
            return False
        for expected_key, expected_value in metric_items:
            matched_key = None
            for actual_key, actual_value in actual_record.items():
                if actual_key in used_actual_keys or not _is_scalar_payload(actual_value):
                    continue
                actual_key_token = _label_token_for_pivot(actual_key)
                if not any(token in actual_key_token for token in label_tokens):
                    continue
                if _strict_keyed_scalar_equal(expected_key, expected_value, actual_key, actual_value):
                    matched_key = actual_key
                    break
            if matched_key is None:
                return False
            used_actual_keys.add(matched_key)
    return True


def _strict_list_match(
    expected: list,
    actual: list,
    *,
    ordered: bool = False,
    answer_fields: Iterable[Any] | None = None,
    tie_candidates: Any = None,
) -> bool:
    if len(expected) != len(actual):
        if len(actual) == 1 and isinstance(actual[0], dict):
            return _pivoted_record_matches_list(expected, actual[0])
        return False
    if ordered:
        if all(
            _strict_list_item_match(
                e,
                a,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            )
            for e, a in zip(expected, actual)
        ):
            return True
        return _ranked_tie_truncated_match(expected, actual, tie_candidates=tie_candidates)
    return _maximum_bipartite_matches(
        expected,
        actual,
        answer_fields=answer_fields,
        tie_candidates=tie_candidates,
    ) == len(expected)


def _best_actual_items_for_entity_metrics(
    expected: list,
    actual: Any,
    *,
    ordered: bool = False,
    answer_fields: Iterable[Any] | None = None,
    tie_candidates: Any = None,
) -> tuple[list, int, bool]:
    candidates = _actual_list_candidates(expected, actual)
    if not candidates:
        return [], 0, False

    best_items: list = []
    best_matched = -1
    best_len_delta = sys.maxsize
    best_tie_equivalent = False
    for candidate in candidates:
        tie_equivalent = bool(
            ordered and _ranked_tie_truncated_match(expected, candidate, tie_candidates=tie_candidates)
        )
        pivot_equivalent = bool(
            len(candidate) == 1
            and isinstance(candidate[0], dict)
            and _pivoted_record_matches_list(expected, candidate[0])
        )
        strict_equivalent = _strict_list_match(
            expected,
            candidate,
            ordered=ordered,
            answer_fields=answer_fields,
            tie_candidates=tie_candidates,
        )
        matched = len(expected) if (tie_equivalent or strict_equivalent) else _maximum_bipartite_matches(
            expected,
            candidate,
            answer_fields=answer_fields,
            tie_candidates=tie_candidates,
        )
        # A single wide/pivoted record can encode multiple category rows.
        # Entity metric denominators must use the represented row count rather
        # than the physical wrapper-row count, otherwise precision can exceed 1.
        metric_items = list(expected) if pivot_equivalent else candidate
        len_delta = abs(len(metric_items) - len(expected))
        if (
            matched > best_matched
            or (matched == best_matched and len_delta < best_len_delta)
        ):
            best_items = metric_items
            best_matched = matched
            best_len_delta = len_delta
            best_tie_equivalent = tie_equivalent
    return best_items, max(best_matched, 0), best_tie_equivalent


def _cluster_count_vector(value: Any) -> list[float] | None:
    """Extract cluster sizes while treating cluster labels as arbitrary names."""
    if isinstance(value, dict):
        if value and all(_is_scalar_payload(item) for item in value.values()):
            key_tokens = [_compact_key(key) for key in value]
            if all(
                re.fullmatch(r"(?:cluster_|簇|类别)?\d+", token) is not None
                for token in key_tokens
            ):
                numbers = [_parse_number_and_unit(item) for item in value.values()]
                if all(number is not None for number in numbers):
                    return sorted(float(number[2]) for number in numbers if number is not None)
        for key, nested in value.items():
            compact = _compact_key(key)
            if any(token in compact for token in ("cluster", "聚类", "簇", "类别")):
                vector = _cluster_count_vector(nested)
                if vector is not None:
                    return vector
        return None
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        counts: list[float] = []
        for row in value:
            has_cluster = any(_field_base_metric(key) == "cluster" for key in row)
            count_values = [
                item
                for key, item in row.items()
                if _field_stat_semantics(key) == "count"
                and _is_scalar_payload(item)
                and _parse_number_and_unit(item) is not None
            ]
            if not has_cluster or len(count_values) != 1:
                return None
            parsed = _parse_number_and_unit(count_values[0])
            counts.append(float(parsed[2]))
        return sorted(counts)
    return None


def _cluster_count_permutation_match(expected: Any, actual: Any) -> bool:
    expected_counts = _cluster_count_vector(expected)
    actual_counts = _cluster_count_vector(actual)
    if expected_counts is None or actual_counts is None:
        return False
    if len(expected_counts) != len(actual_counts):
        return False
    return all(
        _strict_scalar_equal(expected_value, actual_value)
        for expected_value, actual_value in zip(expected_counts, actual_counts)
    )


def _confidence_interval_pair(value: Any) -> tuple[Any, Any] | None:
    if not isinstance(value, dict):
        return None
    lower = upper = None
    lower_group = _field_alias_group("ci95_lower")
    upper_group = _field_alias_group("ci95_upper")
    for key, item in value.items():
        compact = _compact_key(key)
        group = _field_alias_group(_strict_norm_key(key))
        is_ci_key = any(token in compact for token in ("confidence_interval", "ci95", "置信区间"))
        is_lower_key = is_ci_key and any(
            token in compact for token in ("lower", "下限", "下界", "2.5%")
        )
        is_upper_key = is_ci_key and any(
            token in compact for token in ("upper", "上限", "上界", "97.5%")
        )
        if (group == lower_group or is_lower_key) and _is_scalar_payload(item):
            lower = item
        elif (group == upper_group or is_upper_key) and _is_scalar_payload(item):
            upper = item
        elif isinstance(item, list) and len(item) == 2:
            if (
                any(token in compact for token in ("confidence_interval", "ci95", "置信区间"))
                and all(_parse_number_and_unit(part) is not None for part in item)
            ):
                lower, upper = item
    if lower is not None and upper is not None:
        return lower, upper
    for nested in value.values():
        if isinstance(nested, dict):
            pair = _confidence_interval_pair(nested)
            if pair is not None:
                return pair
    return None


def _confidence_interval_match(expected: Any, actual: Any) -> bool:
    expected_pair = _confidence_interval_pair(expected)
    actual_pair = _confidence_interval_pair(actual)
    if expected_pair is None or actual_pair is None:
        return False

    def _bound_equal(expected_value: Any, actual_value: Any) -> bool:
        expected_number = _parse_number_and_unit(expected_value)
        actual_number = _parse_number_and_unit(actual_value)
        if expected_number is None or actual_number is None:
            return False
        return math.isclose(
            expected_number[2],
            actual_number[2],
            rel_tol=0.0,
            abs_tol=STRICT_CONFIDENCE_INTERVAL_ABS_TOL,
        )

    return bool(
        _bound_equal(expected_pair[0], actual_pair[0])
        and _bound_equal(expected_pair[1], actual_pair[1])
    )


_BOUNDING_BOX_CONTEXT_KEYS = {
    "bbox",
    "bounds",
    "boundingbox",
    "minimumboundingrectangle",
    "minimumenclosingrectangle",
    "extent",
    "最小外接矩形",
    "最小包围矩形",
    "外接矩形",
    "边界框",
    "包围盒",
}
_BOUNDING_BOX_AXIS_KEYS = {
    "minx": {"minx", "xmin", "minlon", "minlongitude", "最小x", "最小经度"},
    "miny": {"miny", "ymin", "minlat", "minlatitude", "最小y", "最小纬度"},
    "maxx": {"maxx", "xmax", "maxlon", "maxlongitude", "最大x", "最大经度"},
    "maxy": {"maxy", "ymax", "maxlat", "maxlatitude", "最大y", "最大纬度"},
}


def _bounding_box_compact_key(value: Any) -> str:
    """Normalize only separators/case; keep the complete field meaning."""
    return re.sub(r"[\s_\-.:/\\()\[\]{}]+", "", _strict_norm_key(value))


def _is_bounding_box_context_key(key: Any) -> bool:
    compact = _bounding_box_compact_key(key)
    return compact in _BOUNDING_BOX_CONTEXT_KEYS


def _bounding_box_axis_from_key(key: Any) -> tuple[str | None, bool]:
    """Return ``(axis, explicitly_bbox_qualified)`` for one coordinate key."""
    compact = _bounding_box_compact_key(key)
    for axis, aliases in _BOUNDING_BOX_AXIS_KEYS.items():
        if compact in aliases:
            return axis, False
        for context in _BOUNDING_BOX_CONTEXT_KEYS:
            if compact == f"{context}{axis}" or (
                compact.startswith(context) and compact[len(context):] in aliases
            ):
                return axis, True
    return None, False


def _extract_bounding_box(value: Any, *, bbox_context: bool = False) -> dict[str, Any] | None:
    """Extract one complete bounding box from nested or flattened payloads.

    Accepted forms include ``{"BoundingBox": {"minX": ...}}`` and flattened
    fields such as ``最小外接矩形_minx``.  A match is produced only when all four
    coordinate axes are present in the same dictionary; unrelated leaf values
    are never combined across records.
    """
    if isinstance(value, list):
        if len(value) != 1:
            return None
        return _extract_bounding_box(value[0], bbox_context=bbox_context)
    if not isinstance(value, dict):
        return None

    coordinates: dict[str, Any] = {}
    has_explicit_bbox_key = bbox_context
    for key, item in value.items():
        if not _is_scalar_payload(item):
            continue
        axis, explicitly_qualified = _bounding_box_axis_from_key(key)
        if axis is None or axis in coordinates:
            continue
        coordinates[axis] = item
        has_explicit_bbox_key = has_explicit_bbox_key or explicitly_qualified

    if set(coordinates) == {"minx", "miny", "maxx", "maxy"}:
        # Four conventional bare axis names are already an unambiguous bbox;
        # flattened non-standard names must carry an explicit bbox qualifier.
        if has_explicit_bbox_key or len(coordinates) == 4:
            return coordinates

    for key, nested in value.items():
        if not isinstance(nested, (dict, list)):
            continue
        candidate = _extract_bounding_box(
            nested,
            bbox_context=bbox_context or _is_bounding_box_context_key(key),
        )
        if candidate is not None:
            return candidate
    return None


def _bounding_box_match(expected: Any, actual: Any) -> bool:
    expected_bbox = _extract_bounding_box(expected)
    actual_bbox = _extract_bounding_box(actual)
    if expected_bbox is None or actual_bbox is None:
        return False
    for axis in ("minx", "miny", "maxx", "maxy"):
        expected_number = _parse_number_and_unit(expected_bbox[axis])
        actual_number = _parse_number_and_unit(actual_bbox[axis])
        if expected_number is None or actual_number is None:
            return False
        if expected_number[1] is not None or actual_number[1] is not None:
            return False
        if not math.isclose(
            expected_number[2],
            actual_number[2],
            rel_tol=0.0,
            abs_tol=STRICT_BOUNDING_BOX_ABS_TOL_DEGREES,
        ):
            return False
    return True


def _strict_value_match(
    expected: Any,
    actual: Any,
    *,
    ordered: bool = False,
    answer_fields: Iterable[Any] | None = None,
    tie_candidates: Any = None,
) -> bool:
    if isinstance(expected, dict):
        expected_items = _entity_metric_gold_items(expected)
        if expected_items is not None:
            list_equivalent = any(
                _strict_list_match(
                    expected_items,
                    candidate,
                    ordered=ordered,
                    answer_fields=answer_fields,
                    tie_candidates=tie_candidates,
                )
                for candidate in _actual_list_candidates(expected_items, actual)
            )
            if ordered or list_equivalent:
                return list_equivalent
        if ordered:
            return False
        if (
            len(expected) == 1
            and isinstance(next(iter(expected.values())), dict)
            and isinstance(actual, dict)
        ):
            nested_expected = next(iter(expected.values()))
            if _strict_value_match(
                nested_expected,
                actual,
                ordered=ordered,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            ):
                return True
        if isinstance(actual, dict):
            if len(actual) == 1 and isinstance(next(iter(actual.values())), dict):
                if _strict_value_match(
                    expected,
                    next(iter(actual.values())),
                    ordered=ordered,
                    answer_fields=answer_fields,
                    tie_candidates=tie_candidates,
                ):
                    return True
            if _strict_record_match(expected, actual, answer_fields=answer_fields):
                return True
            return any(
                _strict_flat_entity_map_vs_rows(expected, nested)
                for nested in _iter_nested_lists(actual)
            )
        if isinstance(actual, list):
            if len(actual) == 1 and isinstance(actual[0], dict):
                if _strict_record_match(expected, actual[0], answer_fields=answer_fields):
                    return True
            return _strict_flat_entity_map_vs_rows(expected, actual)
        # 显式 answer_fields 可把 Gold 记录投影成用户实际要求的单一标识符。
        projected = _project_expected_record(expected, answer_fields)
        if len(projected) == 1:
            return _strict_scalar_equal(next(iter(projected.values())), actual)
        return False
    if isinstance(expected, list):
        return any(
            _strict_list_match(
                expected,
                candidate,
                ordered=ordered,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            )
            for candidate in _actual_list_candidates(expected, actual)
        )
    if isinstance(actual, dict):
        if len(actual) == 1:
            return _strict_scalar_equal(expected, next(iter(actual.values())))
        return _record_contains_scalar_answer_value(actual, expected, answer_fields=answer_fields)
    if isinstance(actual, list):
        return any(
            _strict_value_match(
                expected,
                item,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            )
            for item in actual
        )
    return _strict_scalar_equal(expected, actual)


def _zero_or_nan_entity_metrics(expected: Any) -> tuple[float, float, float, float]:
    if isinstance(expected, list):
        zero_or_one = 1.0 if len(expected) == 0 else 0.0
        return zero_or_one, zero_or_one, zero_or_one, zero_or_one
    return float("nan"), float("nan"), float("nan"), float("nan")


def _entity_metric_gold_items(expected: Any) -> list | None:
    if isinstance(expected, list):
        return expected
    if isinstance(expected, dict):
        list_values = [value for value in expected.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]
    return None


def _allowed_identifier_keys_for_question(
    question: str,
    expected_record: dict,
    *,
    answer_fields: Iterable[Any] | None = None,
    answer_entity_level: Any = None,
) -> set[str]:
    """Infer which identifier/name field can be returned as a scalar answer."""
    explicit_fields = _normalized_answer_fields(answer_fields)
    if explicit_fields:
        return explicit_fields
    if answer_entity_level:
        level = _compact_key(answer_entity_level)
        level_map = {
            "state": {"shapename", "state", "state_name"},
            "province": {"province", "province_name"},
            "city": {"fullname", "city", "city_name", "nearest_city"},
            "county": {"county", "county_name"},
            "cell_id": {"cell_id", "grid_id", "fishnet_id"},
            "asdf_id": {"asdf_id"},
            "gqid": {"gqid"},
            "code": {"code", "city_code", "admin_code"},
        }
        if level in level_map:
            return level_map[level]
    q = _strict_norm_string(question)
    compact_q = q.replace(" ", "")
    allowed: set[str] = set()
    if (
        "cell_id" in q
        or "grid id" in q
        or "grid_id" in q
        or ("网格" in compact_q and ("id" in compact_q or "编号" in compact_q))
    ):
        allowed.add("cell_id")
    if "asdf_id" in q or "asdf id" in q:
        allowed.add("asdf_id")
    if "gqid" in q:
        allowed.add("gqid")
    if "shapeid" in q or "shape id" in q:
        allowed.add("shapeid")
    if "shapeiso" in q or "shape iso" in q:
        allowed.add("shapeiso")
    if (
        "which code" in q
        or "what code" in q
        or "code?" in q
        or "代码是什么" in compact_q
        or "代码是多少" in compact_q
        or "编码是什么" in compact_q
        or "编码是多少" in compact_q
    ):
        allowed.add("code")
    if (
        "哪个州" in compact_q
        or "state" in q
        or "which state" in q
    ):
        allowed.update({"shapename", "state", "state_name"})
    if (
        "哪个城市" in compact_q
        or "city" in q
        or "which city" in q
    ):
        allowed.update({"fullname", "city", "city_name", "nearest_city"})
    if not allowed:
        entity_keys = [
            _strict_norm_key(key)
            for key in expected_record
            if _is_entity_like_key(key)
        ]
        if len(entity_keys) == 1:
            allowed.add(entity_keys[0])
    return allowed


def _single_record_identifier_answer_match(
    expected: Any,
    actual: Any,
    *,
    question: str = "",
    answer_fields: Iterable[Any] | None = None,
    answer_entity_level: Any = None,
) -> bool:
    """Allow a scalar identifier only when the question implies that field.

    Example: the question asks "which grid ID", while Gold stores
    ``[{"cell_id": 8586, "population": 381026}]``.  A scalar ``8586`` is a
    complete user-facing answer.  Returning a different entity-like field from
    the same Gold record is not accepted.
    """
    if isinstance(expected, list) and len(expected) == 1 and isinstance(expected[0], dict):
        expected_record = expected[0]
    elif isinstance(expected, dict):
        expected_record = expected
    else:
        return False
    if isinstance(actual, list):
        if len(actual) != 1:
            return False
        actual = actual[0]
    if not _is_scalar_payload(actual):
        return False
    allowed_keys = _allowed_identifier_keys_for_question(
        question,
        expected_record,
        answer_fields=answer_fields,
        answer_entity_level=answer_entity_level,
    )
    return any(
        _strict_norm_key(key) in allowed_keys and _strict_scalar_equal(value, actual)
        for key, value in expected_record.items()
        if _is_scalar_payload(value)
    )


def _extract_validated_strict_answer(results: dict) -> tuple[bool, str | None, Any, str]:
    """校验严格 envelope，并以 typed ``parsed_payload`` 作为答案依据。

    ``artifact_payload`` 是可选明细附件，可能与最终问题处于不同粒度，例如最终答案
    是各聚类数量，而附件保存每个实体的聚类标签。附件不能覆盖已经通过 Schema 校验
    的最终 payload，否则正确的聚合答案会被明细文件制造为假阴性。
    """
    if results.get("answer_schema_valid") is not True:
        return False, None, None, "schema_flag_false"
    raw_answer = results.get("strict_answer")
    if raw_answer is None:
        return False, None, None, "missing_strict_answer"
    try:
        answer = coerce_strict_answer(raw_answer)
    except Exception as exc:
        return False, None, None, f"schema_violation:{type(exc).__name__}"
    if "parsed_payload" not in results:
        return False, None, None, "missing_full_payload"
    payload = answer.model_dump(mode="json")["data_payload"]
    if results.get("parsed_payload") != payload:
        return False, None, None, "payload_mismatch"
    return True, answer.answer_type, payload, "schema_valid"


def compute_strict_structured_metrics(
    *,
    expected_subset: Any,
    actual_payload: Any,
    answer_type: str | None,
    schema_valid: bool,
    execution_success: bool,
    question: str,
    answer_fields: Iterable[Any] | None = None,
    answer_entity_level: Any = None,
    tie_candidates: Any = None,
) -> dict[str, Any]:
    """计算主指标；任何运行或 Schema 失败都不能由 final_answer 文本兜底。"""
    entity_gold_items = _entity_metric_gold_items(expected_subset)
    has_entity_gold = entity_gold_items is not None
    strict_ordered = bool(
        has_entity_gold
        and (answer_type == "ranked_list" or _RANKING_QUESTION_RE.search(str(question or "")))
    )
    metric_ordered = bool(
        has_entity_gold
        and (answer_type == "ranked_list" or _RANKING_QUESTION_RE.search(str(question or "")))
    )
    if not execution_success:
        p, r, f1, exact_set = _zero_or_nan_entity_metrics(entity_gold_items)
        return {
            "strict_structured_accuracy": 0,
            "exact_set_match": exact_set if has_entity_gold and not metric_ordered else float("nan"),
            "entity_precision": p,
            "entity_recall": r,
            "entity_f1": f1,
            "ranking_exact_match": 0.0 if metric_ordered else float("nan"),
            "strict_match_method": "execution_failed",
        }
    if not schema_valid:
        p, r, f1, exact_set = _zero_or_nan_entity_metrics(entity_gold_items)
        return {
            "strict_structured_accuracy": 0,
            "exact_set_match": exact_set if has_entity_gold and not metric_ordered else float("nan"),
            "entity_precision": p,
            "entity_recall": r,
            "entity_f1": f1,
            "ranking_exact_match": 0.0 if metric_ordered else float("nan"),
            "strict_match_method": "schema_invalid",
        }
    if expected_subset is None:
        return {
            "strict_structured_accuracy": 0,
            "exact_set_match": float("nan"),
            "entity_precision": float("nan"),
            "entity_recall": float("nan"),
            "entity_f1": float("nan"),
            "ranking_exact_match": float("nan"),
            "strict_match_method": "missing_expectation",
        }

    cluster_label_equivalent = _cluster_count_permutation_match(
        expected_subset,
        actual_payload,
    )
    confidence_interval_equivalent = _confidence_interval_match(
        expected_subset,
        actual_payload,
    )
    bounding_box_equivalent = _bounding_box_match(
        expected_subset,
        actual_payload,
    )
    strict_ok = (
        cluster_label_equivalent
        or confidence_interval_equivalent
        or bounding_box_equivalent
        or _strict_value_match(
            expected_subset,
            actual_payload,
            ordered=strict_ordered,
            answer_fields=answer_fields,
            tie_candidates=tie_candidates,
        )
    )
    subset_record_equivalent = False
    if (
        not strict_ok
        and isinstance(expected_subset, dict)
        and _question_allows_subset_record(question)
        and _nested_record_match_exists(expected_subset, actual_payload)
    ):
        subset_record_equivalent = True
        strict_ok = True
    identifier_equivalent = False
    if not strict_ok:
        identifier_equivalent = _single_record_identifier_answer_match(
            expected_subset,
            actual_payload,
            question=question,
            answer_fields=answer_fields,
            answer_entity_level=answer_entity_level,
        )
        strict_ok = identifier_equivalent
    entity_precision = entity_recall = entity_f1 = float("nan")
    exact_set_match = ranking_exact_match = float("nan")
    tie_equivalent = False

    if has_entity_gold:
        if identifier_equivalent:
            actual_items = actual_payload if isinstance(actual_payload, list) else [actual_payload]
            matched = 1
        else:
            actual_items, matched, tie_equivalent = _best_actual_items_for_entity_metrics(
                entity_gold_items,
                actual_payload,
                ordered=metric_ordered,
                answer_fields=answer_fields,
                tie_candidates=tie_candidates,
            )
        entity_precision = matched / len(actual_items) if actual_items else (1.0 if not entity_gold_items else 0.0)
        entity_recall = matched / len(entity_gold_items) if entity_gold_items else (1.0 if not actual_items else 0.0)
        if entity_precision + entity_recall:
            entity_f1 = 2.0 * entity_precision * entity_recall / (entity_precision + entity_recall)
        else:
            entity_f1 = 0.0
        if metric_ordered:
            ranking_exact_match = float(strict_ok)
        else:
            exact_set_match = float(strict_ok)

    return {
        "strict_structured_accuracy": int(strict_ok),
        "exact_set_match": exact_set_match,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "ranking_exact_match": ranking_exact_match,
        "strict_match_method": (
            "cluster_label_permutation"
            if cluster_label_equivalent and strict_ok
            else (
                "confidence_interval_equivalent"
                if confidence_interval_equivalent and strict_ok
                else (
                    "bounding_box_equivalent"
                    if bounding_box_equivalent and strict_ok
                    else (
                        "ranked_tie_equivalent"
                        if tie_equivalent and strict_ok
                        else (
                            "single_record_identifier"
                            if identifier_equivalent
                            else (
                                "subset_record_equivalent"
                                if subset_record_equivalent
                                else ("ordered_exact" if strict_ordered else "structured_exact")
                            )
                        )
                    )
                )
            )
        ),
    }


def compute_best_strict_structured_metrics(
    *,
    expected_subset: Any,
    acceptable_expected_subsets: Iterable[Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate explicit alternative Gold payloads and keep the best valid match."""
    candidates = [expected_subset]
    candidates.extend(list(acceptable_expected_subsets or []))
    evaluated = [
        compute_strict_structured_metrics(
            expected_subset=candidate,
            **kwargs,
        )
        for candidate in candidates
    ]

    def _score(metrics: dict[str, Any]) -> tuple[float, ...]:
        def _number(name: str) -> float:
            value = metrics.get(name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return -1.0
            return number if not math.isnan(number) else -1.0

        return (
            _number("strict_structured_accuracy"),
            _number("entity_f1"),
            _number("ranking_exact_match"),
            _number("exact_set_match"),
        )

    best_index, best_metrics = max(
        enumerate(evaluated),
        key=lambda pair: (_score(pair[1]), -pair[0]),
    )
    if best_index > 0 and best_metrics["strict_structured_accuracy"]:
        best_metrics = dict(best_metrics)
        best_metrics["strict_match_method"] = (
            f"alternative_expected:{best_metrics['strict_match_method']}"
        )
    return best_metrics


def _missing_result_list_metrics(expected_subset: Any, question: str = "") -> dict[str, float]:
    entity_gold_items = _entity_metric_gold_items(expected_subset)
    has_entity_gold = entity_gold_items is not None
    metric_ordered = bool(has_entity_gold and _RANKING_QUESTION_RE.search(str(question or "")))
    if not has_entity_gold:
        return {
            "exact_set_match": float("nan"),
            "entity_precision": float("nan"),
            "entity_recall": float("nan"),
            "entity_f1": float("nan"),
            "ranking_exact_match": float("nan"),
        }
    p, r, f1, exact_set = _zero_or_nan_entity_metrics(entity_gold_items)
    return {
        "exact_set_match": exact_set if not metric_ordered else float("nan"),
        "entity_precision": p,
        "entity_recall": r,
        "entity_f1": f1,
        "ranking_exact_match": 0.0 if metric_ordered else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  路由 / 效率 / 鲁棒性
# ══════════════════════════════════════════════════════════════════════════════


def used_python_from_tool_trace(tool_routing_path: Any) -> bool:
    """工具轨迹中是否出现过 python_analysis_tool。"""
    if tool_routing_path is None or not isinstance(tool_routing_path, list):
        return False
    return any(
        isinstance(step, dict) and step.get("tool") == "python_analysis_tool"
        for step in tool_routing_path
    )


def routing_confusion_cells(requires_sandbox: bool, used_python: bool) -> Tuple[int, int, int, int]:
    """二分类混淆矩阵单样本计数，返回 (tp, fp, fn, tn)，各项为 0 或 1。
    真值正类：标注 requires_sandbox（需要 Python 侧计算）。
    预测正类：轨迹中实际调用 python_analysis_tool。
    """
    req = bool(requires_sandbox)
    use = bool(used_python)
    if req and use:
        return 1, 0, 0, 0
    if req and not use:
        return 0, 0, 1, 0
    if not req and use:
        return 0, 1, 0, 0
    return 0, 0, 0, 1


def compute_routing(requires_sandbox: bool, tool_routing_path) -> int:
    """全样本路由准确率：requires_sandbox 与是否调用 Python 一致则 1（等价于 TP+TN）。"""
    used = used_python_from_tool_trace(tool_routing_path)
    return 1 if (bool(requires_sandbox) == used) else 0


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return float("nan")
    return a / b


def finalize_routing_aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    """在已有 routing_*_total 列的聚合表上注入 precision / recall / F1 / FP 率。"""
    out = df.copy()
    tp = out["routing_tp_total"].astype(float)
    fp = out["routing_fp_total"].astype(float)
    fn = out["routing_fn_total"].astype(float)
    tn = out["routing_tn_total"].astype(float)

    denom_p = tp + fp
    denom_r = tp + fn
    out["routing_precision"] = pd.Series(
        [_safe_div(tpv, pv) for tpv, pv in zip(tp, denom_p)],
        index=out.index,
    )
    out["routing_recall"] = pd.Series(
        [_safe_div(tpv, rv) for tpv, rv in zip(tp, denom_r)],
        index=out.index,
    )
    pr = out["routing_precision"]
    rc = out["routing_recall"]
    f1_vals: list[float] = []
    for p_i, r_i in zip(pr, rc):
        if p_i != p_i or r_i != r_i:
            f1_vals.append(float("nan"))
        elif p_i == 0 and r_i == 0:
            f1_vals.append(0.0)
        else:
            f1_vals.append(2.0 * p_i * r_i / (p_i + r_i))
    out["routing_f1"] = f1_vals

    neg = fp + tn
    out["routing_fp_rate_among_negatives"] = pd.Series(
        [_safe_div(fpv, nv) for fpv, nv in zip(fp, neg)],
        index=out.index,
    )

    for col in (
        "routing_precision",
        "routing_recall",
        "routing_f1",
        "routing_fp_rate_among_negatives",
        "routing_accuracy",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def graph_steps_from_tool_trace(tool_routing_path) -> int:
    """
    可跨消融对比的「步数」：tool_routing_path 中记录的条数（每条对应一次工具调用轨迹，
    与 run_agent_autotest._append_tool_trace 一致）。不使用 execution_metrics.total_graph_steps。
    """
    if not tool_routing_path or not isinstance(tool_routing_path, list):
        return 0
    return len(tool_routing_path)


def extract_efficiency(run_data: dict) -> dict:
    exec_metrics = run_data.get("execution_metrics") or {}
    token_usage = run_data.get("token_usage") or {}
    return {
        "exec_status":       exec_metrics.get("status", "unknown"),
        "time_sec":          exec_metrics.get("execution_time_seconds"),
        "total_tokens":      token_usage.get("total_tokens"),
        "prompt_tokens":     token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
    }


def extract_robustness(run_data: dict) -> dict:
    exec_metrics = run_data.get("execution_metrics") or {}
    trace = run_data.get("trace_summary") or {}
    routing_path = trace.get("tool_routing_path") or []
    return {
        "guardrail_retries": int(exec_metrics.get("guardrail_retry_count") or 0),
        "failed_calls": sum(
            1 for step in routing_path
            if isinstance(step, dict) and step.get("status") == "failed"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  数据加载与评估循环
# ══════════════════════════════════════════════════════════════════════════════

def load_benchmark(benchmark_path: Path) -> dict:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)
    if not isinstance(items, list):
        raise ValueError(f"基准文件格式错误，预期顶层为列表: {benchmark_path}")
    return {item["question_id"]: item for item in items}


def _benchmark_eval_hints(gt: dict) -> dict[str, Any]:
    expected_exec = gt.get("expected_execution_result") or {}
    return {
        "answer_fields": (
            gt.get("answer_fields")
            or expected_exec.get("answer_fields")
            or gt.get("expected_answer_fields")
            or expected_exec.get("expected_answer_fields")
        ),
        "answer_entity_level": (
            gt.get("answer_entity_level")
            or expected_exec.get("answer_entity_level")
        ),
        "tie_candidates": (
            gt.get("tie_candidates")
            or expected_exec.get("tie_candidates")
            or gt.get("expected_tie_candidates")
            or expected_exec.get("expected_tie_candidates")
        ),
        "acceptable_expected_subsets": (
            gt.get("acceptable_expected_subsets")
            or expected_exec.get("acceptable_expected_subsets")
            or gt.get("alternative_payload_subsets")
            or expected_exec.get("alternative_payload_subsets")
        ),
    }


def evaluate_run_dir(
    run_dir: Path,
    benchmark: dict,
    experiment_name: str,
    *,
    include_missing: bool = True,
) -> list:
    yaml_files = sorted(
        f for f in run_dir.iterdir()
        if f.suffix == ".yaml" and "_console" not in f.name
    )

    rows = []
    seen_question_ids: set[int] = set()
    for yaml_path in yaml_files:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                run_data = yaml.safe_load(f)
        except Exception as exc:
            print(f"  [WARN] 无法加载文件 {yaml_path.name}: {exc}", file=sys.stderr)
            continue

        if not run_data or not isinstance(run_data, dict):
            print(f"  [WARN] 空或非法 YAML: {yaml_path.name}", file=sys.stderr)
            continue

        metadata = run_data.get("metadata") or {}
        raw_question_id = metadata.get("question_id")
        if raw_question_id is None:
            print(f"  [WARN] 缺少 question_id: {yaml_path.name}", file=sys.stderr)
            continue
        try:
            question_id = int(raw_question_id)
        except (TypeError, ValueError):
            print(
                f"  [WARN] 非法 question_id={raw_question_id!r}: {yaml_path.name}",
                file=sys.stderr,
            )
            continue
        if question_id in seen_question_ids:
            raise ValueError(
                f"结果目录 {run_dir} 中 question_id={question_id} 重复；"
                "请确保每个实验目录每题只有一个 YAML，避免重复计入分母。"
            )

        gt = benchmark.get(question_id)
        if gt is None:
            print(
                f"  [WARN] question_id={question_id} 不在基准文件中，跳过: {yaml_path.name}",
                file=sys.stderr,
            )
            continue
        seen_question_ids.add(question_id)

        difficulty = normalize_difficulty_label(gt.get("difficulty", "unknown"))
        requires_sandbox = bool(gt.get("requires_sandbox", False))
        expected_exec = gt.get("expected_execution_result") or {}
        expected_subset = expected_exec.get("data_payload_subset")
        eval_hints = _benchmark_eval_hints(gt)

        results = run_data.get("results") or {}
        trace = run_data.get("trace_summary") or {}
        tool_routing_path = trace.get("tool_routing_path") or []

        used_py = used_python_from_tool_trace(tool_routing_path)
        rt, rf, rn, rz = routing_confusion_cells(requires_sandbox, used_py)
        routing_correct = compute_routing(requires_sandbox, tool_routing_path)
        efficiency = extract_efficiency(run_data)
        robustness = extract_robustness(run_data)
        n_tool_trace_steps = graph_steps_from_tool_trace(tool_routing_path)
        schema_valid, answer_type, strict_payload, schema_status = _extract_validated_strict_answer(results)
        strict_metrics = compute_best_strict_structured_metrics(
            expected_subset=expected_subset,
            acceptable_expected_subsets=eval_hints["acceptable_expected_subsets"],
            actual_payload=strict_payload,
            answer_type=answer_type,
            schema_valid=schema_valid,
            execution_success=efficiency["exec_status"] == "success",
            question=str(gt.get("question") or metadata.get("query") or ""),
            answer_fields=eval_hints["answer_fields"],
            answer_entity_level=eval_hints["answer_entity_level"],
            tie_candidates=eval_hints["tie_candidates"],
        )
        first_execution_path_success = int(
            strict_metrics["strict_structured_accuracy"] == 1
            and robustness["guardrail_retries"] == 0
            and robustness["failed_calls"] == 0
        )

        rows.append({
            "experiment_name":   experiment_name,
            "question_id":       question_id,
            "result_present":    1,
            "difficulty":        difficulty,
            "requires_sandbox":  requires_sandbox,
            "used_python_tool": int(used_py),
            "routing_tp":        rt,
            "routing_fp":        rf,
            "routing_fn":        rn,
            "routing_tn":        rz,
            "strict_structured_accuracy": strict_metrics["strict_structured_accuracy"],
            "first_execution_path_success": first_execution_path_success,
            "answer_schema_valid": int(schema_valid),
            "answer_schema_status": schema_status,
            "answer_type": answer_type or "",
            "exact_set_match": strict_metrics["exact_set_match"],
            "entity_precision": strict_metrics["entity_precision"],
            "entity_recall": strict_metrics["entity_recall"],
            "entity_f1": strict_metrics["entity_f1"],
            "ranking_exact_match": strict_metrics["ranking_exact_match"],
            "strict_match_method": strict_metrics["strict_match_method"],
            "routing_correct":   routing_correct,
            "exec_status":       efficiency["exec_status"],
            "time_sec":          efficiency["time_sec"],
            "total_tokens":      efficiency["total_tokens"],
            "prompt_tokens":     efficiency["prompt_tokens"],
            "completion_tokens": efficiency["completion_tokens"],
            "graph_steps":       n_tool_trace_steps,
            "guardrail_retries": robustness["guardrail_retries"],
            "failed_calls":      robustness["failed_calls"],
        })

    if include_missing:
        for question_id, gt in sorted(benchmark.items()):
            qid = int(question_id)
            if qid in seen_question_ids:
                continue
            difficulty = normalize_difficulty_label(gt.get("difficulty", "unknown"))
            requires_sandbox = bool(gt.get("requires_sandbox", False))
            expected_exec = gt.get("expected_execution_result") or {}
            expected_subset = expected_exec.get("data_payload_subset")
            question_text = str(gt.get("question") or "")
            missing_list_metrics = _missing_result_list_metrics(expected_subset, question_text)
            rows.append({
                "experiment_name": experiment_name,
                "question_id": qid,
                "result_present": 0,
                "difficulty": difficulty,
                "requires_sandbox": requires_sandbox,
                "used_python_tool": 0,
                # 没有实际执行轨迹时不伪造 TP/TN/FN/FP；routing_correct=0
                # 仍保证缺失题进入路由准确率分母。
                "routing_tp": 0,
                "routing_fp": 0,
                "routing_fn": 0,
                "routing_tn": 0,
                "strict_structured_accuracy": 0,
                "first_execution_path_success": 0,
                "answer_schema_valid": 0,
                "answer_schema_status": "missing_result",
                "answer_type": "",
                "exact_set_match": missing_list_metrics["exact_set_match"],
                "entity_precision": missing_list_metrics["entity_precision"],
                "entity_recall": missing_list_metrics["entity_recall"],
                "entity_f1": missing_list_metrics["entity_f1"],
                "ranking_exact_match": missing_list_metrics["ranking_exact_match"],
                "strict_match_method": "missing_result",
                "routing_correct": 0,
                "exec_status": "missing_result",
                "time_sec": None,
                "total_tokens": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "graph_steps": 0,
                "guardrail_retries": 0,
                "failed_calls": 1,
            })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 9.  报告导出
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if "result_present" not in detail_df.columns:
        detail_df = detail_df.copy()
        detail_df["result_present"] = 1
    summary = (
        detail_df.groupby(["experiment_name", "difficulty"], sort=True)
        .agg(
            n_questions           = ("question_id",       "count"),
            result_coverage       = ("result_present",    "mean"),
            missing_results       = ("result_present",    lambda s: int((s == 0).sum())),
            strict_structured_accuracy = ("strict_structured_accuracy", "mean"),
            first_execution_path_success_rate = ("first_execution_path_success", "mean"),
            schema_valid_rate     = ("answer_schema_valid", "mean"),
            exact_set_match       = ("exact_set_match", "mean"),
            entity_precision      = ("entity_precision", "mean"),
            entity_recall         = ("entity_recall", "mean"),
            entity_f1             = ("entity_f1", "mean"),
            ranking_exact_match   = ("ranking_exact_match", "mean"),
            routing_accuracy      = ("routing_correct",   "mean"),
            routing_tp_total      = ("routing_tp",        "sum"),
            routing_fp_total      = ("routing_fp",        "sum"),
            routing_fn_total      = ("routing_fn",        "sum"),
            routing_tn_total      = ("routing_tn",        "sum"),
            avg_time_sec          = ("time_sec",          "mean"),
            avg_total_tokens      = ("total_tokens",      "mean"),
            avg_prompt_tokens     = ("prompt_tokens",     "mean"),
            avg_compl_tokens      = ("completion_tokens", "mean"),
            avg_graph_steps       = ("graph_steps",       "mean"),
            avg_guardrail_retries = ("guardrail_retries", "mean"),
            avg_failed_calls      = ("failed_calls",      "mean"),
        )
        .reset_index()
    )
    summary = finalize_routing_aggregate_df(summary)
    for col in (
        "result_coverage",
        "strict_structured_accuracy",
        "first_execution_path_success_rate",
        "schema_valid_rate",
        "exact_set_match",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "ranking_exact_match",
    ):
        summary[col] = summary[col].round(4)
    return summary


def build_overall(detail_df: pd.DataFrame) -> pd.DataFrame:
    if "result_present" not in detail_df.columns:
        detail_df = detail_df.copy()
        detail_df["result_present"] = 1
    overall = (
        detail_df.groupby("experiment_name", sort=True)
        .agg(
            n_questions      = ("question_id",       "count"),
            result_coverage  = ("result_present",    "mean"),
            missing_results  = ("result_present",    lambda s: int((s == 0).sum())),
            strict_structured_accuracy = ("strict_structured_accuracy", "mean"),
            first_execution_path_success_rate = ("first_execution_path_success", "mean"),
            schema_valid_rate = ("answer_schema_valid", "mean"),
            exact_set_match = ("exact_set_match", "mean"),
            entity_precision = ("entity_precision", "mean"),
            entity_recall = ("entity_recall", "mean"),
            entity_f1 = ("entity_f1", "mean"),
            ranking_exact_match = ("ranking_exact_match", "mean"),
            routing_accuracy = ("routing_correct",   "mean"),
            routing_tp_total = ("routing_tp",        "sum"),
            routing_fp_total = ("routing_fp",        "sum"),
            routing_fn_total = ("routing_fn",       "sum"),
            routing_tn_total = ("routing_tn",        "sum"),
            avg_time_sec     = ("time_sec",          "mean"),
            avg_total_tokens = ("total_tokens",      "mean"),
            avg_graph_steps  = ("graph_steps",       "mean"),
            avg_failed_calls = ("failed_calls",      "mean"),
        )
        .reset_index()
    )
    overall = finalize_routing_aggregate_df(overall)
    for col in (
        "result_coverage",
        "strict_structured_accuracy",
        "first_execution_path_success_rate",
        "schema_valid_rate",
        "exact_set_match",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "ranking_exact_match",
    ):
        overall[col] = overall[col].round(4)
    return overall


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动化批量消融实验评估 — 严格结构化指标（v5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmark_path", default=None, help="Ground truth YAML")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=None,
        help="一个或多个本次实验结果目录；未使用 suite 模式时必须显式指定，避免混入历史结果",
    )
    parser.add_argument(
        "--ablation_suite",
        action="store_true",
        help="评价已统一移动到 benchmark/agent_runs 下的 7 个 qwen3.7-plus 消融/基线目录",
    )
    parser.add_argument(
        "--model_suite",
        action="store_true",
        help="评价已统一移动到 benchmark/agent_runs 下的 5 个主模型结果目录",
    )
    parser.add_argument(
        "--final_suite",
        action="store_true",
        help="同时评价 5 个主模型与 7 个消融/基线目录，生成最终总表",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="仅评价目录中已有题目；只用于冒烟/子集诊断，完整论文实验请勿使用",
    )
    parser.add_argument("--output_dir", default=None, help="输出 CSV 目录")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path) if args.benchmark_path else (
        _SCRIPT_ROOT / BENCHMARK_RELPATH
    )
    suite_flags = [
        args.ablation_suite,
        args.model_suite,
        args.final_suite,
    ]
    if sum(bool(flag) for flag in suite_flags) > 1:
        parser.error("--ablation_suite、--model_suite、--final_suite 只能选择一个")

    if args.ablation_suite or args.model_suite or args.final_suite:
        if args.run_dirs:
            parser.error("suite 模式会自动选择统一目录，请不要同时传 --run_dirs")
        if args.ablation_suite:
            run_dir_list = [_SCRIPT_ROOT / p for p in ABLATION_SUITE_RUN_DIRS]
            default_output_relpath = ABLATION_SUITE_OUTPUT_RELPATH
        elif args.model_suite:
            run_dir_list = [_SCRIPT_ROOT / p for p in MODEL_SUITE_RUN_DIRS]
            default_output_relpath = MODEL_SUITE_OUTPUT_RELPATH
        else:
            run_dir_list = [
                _SCRIPT_ROOT / p
                for p in (MODEL_SUITE_RUN_DIRS + ABLATION_SUITE_RUN_DIRS)
            ]
            default_output_relpath = FINAL_SUITE_OUTPUT_RELPATH
        output_dir = Path(args.output_dir) if args.output_dir else (
            _SCRIPT_ROOT / default_output_relpath
        )
    else:
        if not args.run_dirs:
            parser.error("必须传 --run_dirs，或使用 --ablation_suite / --model_suite / --final_suite")
        run_dir_list = [Path(p) for p in args.run_dirs]
        output_dir = Path(args.output_dir) if args.output_dir else (
            _SCRIPT_ROOT / OUTPUT_RELPATH
        )

    missing_run_dirs = [p for p in run_dir_list if not p.exists()]
    if missing_run_dirs:
        print("[ERROR] 以下结果目录不存在：", file=sys.stderr)
        for p in missing_run_dirs:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 加载基准文件: {benchmark_path}")
    benchmark = load_benchmark(benchmark_path)
    print(f"[INFO] 共加载 {len(benchmark)} 个基准问题")
    print(
        "[INFO] 主指标：strict_structured_accuracy（仅合法结构化答案；运行/Schema 失败记 0）；"
        "实体列表同时报告 Exact Set Match 与 P/R/F1，排名保序；"
        "first_execution_path_success_rate 仅作为无失败调用、无门禁重试的一次路径稳定性说明。"
    )
    all_rows: list = []
    for run_dir in run_dir_list:
        if not run_dir.is_dir():
            print(f"[WARN] 非目录，已跳过: {run_dir}", file=sys.stderr)
            continue
        experiment_name = run_dir.name
        print(f"[INFO] 评估实验: {experiment_name}")
        rows = evaluate_run_dir(
            run_dir,
            benchmark,
            experiment_name,
            include_missing=not args.allow_partial,
        )
        present_count = sum(int(row.get("result_present", 0)) for row in rows)
        print(
            f"       读取 {present_count} 个结果；评价分母 {len(rows)} 题"
            + ("（允许子集）" if args.allow_partial else "（缺失题记失败）")
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] 没有可处理的结果，退出。")
        sys.exit(1)

    detail_df = pd.DataFrame(all_rows)
    detail_df.sort_values(["experiment_name", "question_id"], inplace=True, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "experiment_results_detail.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 详细报告已保存: {detail_path}")

    summary_df = build_summary(detail_df)
    summary_path = output_dir / "experiment_results_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 汇总报告已保存: {summary_path}")

    overall_df = build_overall(detail_df)
    overall_path = output_dir / "experiment_results_overall.csv"
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 总体报告已保存: {overall_path}")

    print("\n" + "═" * 90)
    print("  总体实验对比（Overall per-Experiment）")
    print("═" * 90)
    print(overall_df.to_string(index=False))

    print("\n" + "═" * 90)
    print("  分难度汇总（Summary by Difficulty）")
    print("═" * 90)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
