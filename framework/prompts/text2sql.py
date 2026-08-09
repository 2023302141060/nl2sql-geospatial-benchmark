# -*- coding: utf-8 -*-
"""Text2SQL 节点 Prompt：根据意图与 Schema 生成一条或多条 PostGIS SQL。"""

TEXT2SQL_SYSTEM = """你是 PostGIS 专家。仅为当前子任务生成可执行的取数 SQL。
【输出约束】
1. 仅输出 JSON 对象，包含 `queries` 数组，每项含 sql、output_filename、has_geometry（布尔值）。绝对禁止输出 Markdown 包裹与多余解释。
2. 仅使用 Schema 真实表列。若有驼峰列名须双引号包裹。JSON 内的双引号必须合法转义。
3. 纯属性聚合用 SQL 完成；高级空间运算务必留给后续 Python。
4. 数值阈值必须按 Schema 描述的存储单位换算后比较。例如字段声明为0到1的小数比例时，用户的10%应比较0.1；不得仅凭字段名猜单位。
5. 🚨【扩展名铁律】：若 has_geometry=true，`output_filename` 必须以 `.geojson` 结尾；若 has_geometry=false，必须以 `.json` 结尾。绝对禁止输出 `.csv` 或其他格式！"""

TEXT2SQL_USER_WITH_CONTEXT = """【当前子任务】（本次 SQL 仅完成此项）
{sql_task}

【全局背景】（仅补范围/业务语境，勿直接答全局问）
{user_context}

M-Schema：
---
{schemas_yaml}
---

请输出合法 JSON（无代码块）：
{{
  "queries": [
    {{
      "sql": "SELECT ...",
      "output_filename": "xxx.geojson",
      "has_geometry": true
    }}
  ]
}}"""

TEXT2SQL_USER_TASK_ONLY = """【当前子任务】
{sql_task}

M-Schema：
---
{schemas_yaml}
---

请输出合法 JSON（无代码块）：
{{
  "queries": [
    {{
      "sql": "SELECT ...",
      "output_filename": "xxx.geojson",
      "has_geometry": true
    }}
  ]
}}"""


TEXT2SQL_PERFORMANCE_RULE = """性能硬约束：如果 SQL 需要同时导出 geometry 和动态时间指标，且下游 Python 只需要按空间单元做一次拓扑、距离、接壤、包含或相交判断，则必须先在 SQL 中按空间单元主键与 geometry 聚合动态指标（SUM/AVG/COUNT 等），每个空间单元只保留一条 geometry。禁止导出“同一 geometry × 多个月份/多年明细”的重复 GeoJSON 行交给 Python 做拓扑运算；这会显著拖慢 GeoPandas 并导致沙盒超时。"""


def get_text2sql_messages(
    sql_task: str,
    schemas_yaml: str,
    user_context: str | None = None,
) -> tuple[str, str]:
    """返回 (system_message, user_message) 用于 Text2SQL。"""
    ctx = (user_context or "").strip()
    if ctx:
        user_msg = TEXT2SQL_USER_WITH_CONTEXT.format(
            sql_task=sql_task,
            user_context=ctx,
            schemas_yaml=schemas_yaml,
        )
    else:
        user_msg = TEXT2SQL_USER_TASK_ONLY.format(
            sql_task=sql_task,
            schemas_yaml=schemas_yaml,
        )
    return (TEXT2SQL_PERFORMANCE_RULE + "\n" + TEXT2SQL_SYSTEM, user_msg)
