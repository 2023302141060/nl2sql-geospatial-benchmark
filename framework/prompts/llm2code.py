# -*- coding: utf-8 -*-
"""LLM2Code 节点 Prompt：根据意图与工作区 GeoJSON 路径生成 Python 空间分析脚本。"""
from pathlib import Path

LLM2CODE_SYSTEM = """你是沙盒内 Python 分析师。仅输出独立可执行的纯代码（禁止 Markdown 块与冗余文本）。

【编码契约】
1. 读取文件仅限 `geojson_paths` 列表给出的文件名。沙盒当前目录就是数据工作区，代码必须直接使用列表中的纯文件名，禁止添加 `/workspace`、`workspace/` 等目录前缀或切换工作目录。`.geojson` 必须用 `gpd.read_file()`，`.json` 必须用 `pd.read_json()`。绝对禁止使用 `open()` 读取未授权数据。
2. 处理空值/无穷值时，绝对禁止使用 `pd.isinf()`！必须使用 `df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})` 替换为空进行 JSON 安全清洗。不要对 numpy.ndarray 调用 replace。
3. 🚨【数据落盘与返回铁律】：
   - 【重型数据】：如果分析结果包含长列表、海量网格详情或大型聚类结果（>15条记录），你必须使用 `df.to_csv("analysis_result.csv")` 等方式落盘。仅当用户要求明细或示例时，才在 `data_payload` 中提供前 5-10 行预览；若用户只要求计数/汇总，payload 只能保留所问的计数/汇总，不得强塞样本 ID。
   - 【轻型数据】：如果最终结果或记录数很少（<=15条，如仅有几个州），无需落盘，【必须直接将包含具体名称/ID的完整结果】放入 `data_payload` 中！
4. 返回结果必须使用标准输出（确保上层 Agent 能看懂结论）：
   `print(json.dumps({"status": "ok", "answer_text": "不超过3句话的分析摘要。如果有落盘文件，必须在这里明确说明文件名称！", "data_payload": {"极值": "...", "聚类明细预览": [{"州名": "A", "聚类": 0}, ...]}, "saved_files": ["..."]}, ensure_ascii=False))`
5. 若遇异常，在 except 块中必须输出：
   `print(json.dumps({"status": "error", "answer_text": f"执行失败: {str(e)}", "data_payload": {"traceback": traceback.format_exc()}}, ensure_ascii=False))`
6. 🚨【实体标识保留铁律】：在进行机器学习（如 K-Means 聚类）、透视表转换（pivot）或矩阵运算时，【绝对禁止】在预处理阶段永久丢弃人类可读的实体标识列（如 shapeName、fullname、州名、城市名等）。计算完成后，【必须】将聚类标签（labels）或计算结果与原始的实体名称重新合并（merge/join）。在输出的 payload 中，禁止使用无意义的数字自动索引（如 "ID": 0, 1, 2），必须输出真实的实体名称！
7. 【答案范围与精度】：`data_payload` 只保留用户问题要求的结果以及理解结果所必需的实体标识、单位和分组键；不要附带未被询问的诊断量或样例。浮点结果保留计算精度，不要先格式化再写入 payload。
8. 【可追溯性】：筛选、排序、分组或汇总后，结果必须保留足以判断“对象是谁、数值是什么、单位或分组是什么”的信息；不得用无意义的自动索引替代真实实体标识。
9. 【语义字段】：payload 字段名应由“原始指标/实体 + 统计操作 + 单位”组成，并沿用输入数据中的指标名；有明确语义时不得只写 `value`、`count`、`result`。极值与排名同时返回实体标识和比较指标；先筛选一组实体再聚合时，同时返回入选实体数量（小集合也可返回完整标识列表）。
10. 【答案投影契约】：优先遵守步骤契约中的 `answer_projection`。`output_shape=scalar` 时 data_payload 直接返回单个值；`single_record` 只返回所问的单条结果，不附带完整序列；排名使用记录列表。按绝对值排序时必须同时输出原始带符号统计量与绝对值排序键；分组键必须体现实际分组维度。问题要求“每个/所有/全部”记录时，payload 必须包含完整记录，preview 或仅落盘文件不能替代答案。
"""

LLM2CODE_SYSTEM += """
11. 【统计量字段协议】：如果步骤契约的 `answer_projection.canonical_statistic_fields`
非空，`data_payload` 必须包含这些精确字段名；可以同时保留更具体的业务字段名。
该协议只统一统计量的接口名称，不规定或暗示任何答案值。
12. 【坐标参考系可复现性】：用户明确给出 EPSG/CRS 时必须原样使用，禁止替换为“类似”投影；
用户只要求“等积投影”但没有指定 CRS 时，统一使用 EPSG:6933。质心、面积等投影运算必须先在该
投影中计算，再按问题要求转换回经纬度或其他单位。
"""

LLM2CODE_USER = """【当前任务】（脚本仅完成此项）
{current_plan_step}

【步骤契约】
{analysis_contract}

【业务背景】
{question}

槽位：{slots_json}

【工作区文件】（下列文件名是唯一合法输入源；代码中字符串字面量只能使用这些名字，不得自造）
{geojson_paths}

上一阶段 SQL（含导出文件名与类型，请与上表对应）：
{sql_context}

{optional_templates_section}{optional_error_section}
请输出完整可运行脚本（无 markdown 围栏）。"""


LLM2CODE_PERFORMANCE_RULE = """性能硬约束：读入 GeoDataFrame 后，如果同一实体（如 asdf_id/cell_id/shapeName/fullname）因月份或年份明细重复出现，而空间拓扑只需按实体判断一次，必须先按实体去重 geometry，并把动态指标用 groupby 聚合后再做 intersects/touches/within/distance。禁止对重复月份行逐行执行拓扑判断。"""


def get_llm2code_messages(
    question: str,
    slots_json: str,
    current_plan_step: str,
    analysis_contract_json: str,
    geojson_paths: list[str],
    sql_queries: list[dict],
    code_templates: list[str] | None = None,
    error_trace: str | None = None,
) -> tuple[str, str]:
    """返回 (system_message, user_message) 用于 LLM2Code。"""
    path_type_map: dict[str, str] = {}
    for p in geojson_paths:
        lower = str(p).lower()
        if lower.endswith(".geojson"):
            path_type_map[p] = "geojson"
        elif lower.endswith(".json"):
            path_type_map[p] = "json"
        else:
            path_type_map[p] = "json"

    for q in (sql_queries or []):
        if not isinstance(q, dict):
            continue
        normalized_path = str(q.get("normalized_path") or "").strip()
        file_type = str(q.get("file_type") or "").strip()
        if normalized_path and file_type:
            path_type_map[normalized_path] = file_type

    def _line_for_workspace_path(p: str) -> str:
        ft = path_type_map.get(p, "json")
        base = Path(str(p)).name
        hint_parts: list[str] = [f"file_type={ft}"]
        for q in sql_queries or []:
            if not isinstance(q, dict):
                continue
            out_fn = str(q.get("output_filename") or "").strip()
            if not out_fn or Path(out_fn).name != base:
                continue
            sql_t = str(q.get("sql") or "").strip().replace("\n", " ")
            if len(sql_t) > 200:
                sql_t = sql_t[:200] + "…"
            if sql_t:
                hint_parts.append(f"对应 SQL: {sql_t}")
            break
        return f"- {Path(p).name}（{'；'.join(hint_parts)}）"

    paths_text = "\n".join(_line_for_workspace_path(p) for p in geojson_paths)

    sql_context_lines = []
    for i, q in enumerate(sql_queries or [], 1):
        sql_text = q.get("sql", "") if isinstance(q, dict) else getattr(q, "sql", "")
        out_name = q.get("output_filename", "") if isinstance(q, dict) else getattr(q, "output_filename", "")
        file_type = q.get("file_type", "") if isinstance(q, dict) else getattr(q, "file_type", "")
        normalized_path = q.get("normalized_path", "") if isinstance(q, dict) else getattr(q, "normalized_path", "")
        has_geometry = q.get("has_geometry", "") if isinstance(q, dict) else getattr(q, "has_geometry", "")
        if sql_text or out_name:
            extras = []
            if normalized_path:
                extras.append(f"标准化路径: {normalized_path}")
            if file_type:
                extras.append(f"file_type: {file_type}")
            if has_geometry != "":
                extras.append(f"has_geometry: {has_geometry}")
            extra_text = f"；{'；'.join(extras)}" if extras else ""
            sql_context_lines.append(f"SQL {i}（导出文件: {out_name}{extra_text}）:\n{sql_text}")
    sql_context = "\n\n".join(sql_context_lines) if sql_context_lines else "（无）"

    optional_templates_section = ""
    if code_templates:
        optional_templates_section = "参考历史脚本（风格/逻辑）：\n" + "\n---\n".join(code_templates) + "\n\n"
    optional_error_section = ""
    if error_trace:
        optional_error_section = f"""上次报错：
{error_trace}
请修正后输出完整脚本。"""
    return (
        LLM2CODE_PERFORMANCE_RULE + "\n" + LLM2CODE_SYSTEM,
        LLM2CODE_USER.format(
            question=question,
            slots_json=slots_json,
            current_plan_step=current_plan_step or "（未显式提供当前步骤）",
            analysis_contract=analysis_contract_json or "{}",
            geojson_paths=paths_text,
            sql_context=sql_context,
            optional_templates_section=optional_templates_section,
            optional_error_section=optional_error_section,
        ),
    )
