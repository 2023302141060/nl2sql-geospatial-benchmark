# -*- coding: utf-8 -*-
"""
执行 PostGIS SQL 并将结果导出为 GeoJSON 或 JSON。

重要：不使用 psycopg2 的 cursor.fetchall() 处理含几何列的结果。
- 含几何列：统一使用 geopandas.read_postgis(sql, conn/engine, geom_col) + gdf.to_file(..., driver="GeoJSON")。
- 不含几何列：使用 pandas.read_sql 得到 DataFrame，再保存为 JSON 或返回字典。
"""
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import config

# 全局单例 Engine，避免 ReAct 多轮循环中频繁创建导致连接耗尽
_engine = None
_operational_retry_active = False
_MAX_SQL_TEXT_CHARS = 60000
_ROW_LOCKING_PATTERN = re.compile(r"\b(FOR\s+UPDATE|FOR\s+SHARE|LOCK\s+TABLE)\b", re.IGNORECASE)
_RISKY_FUNCTION_PATTERN = re.compile(
    r"\b(pg_sleep|dblink|copy\s+\(|copy\s+[a-z_][\w]*\s+from|set_config\s*\(|pg_read_file\s*\(|pg_ls_dir\s*\(|lo_import\s*\(|lo_export\s*\()",
    re.IGNORECASE,
)
_CTE_WRITE_PATTERN = re.compile(
    r"^\s*WITH\b[\s\S]*?\b(INSERT|UPDATE|DELETE|MERGE)\b",
    re.IGNORECASE,
)


def get_engine():
    """获取 SQLAlchemy 引擎（单例 + 连接池），供 geopandas / pandas 使用。"""
    global _engine
    if _engine is None:
        st_ms = int(getattr(config, "SQL_STATEMENT_TIMEOUT_MS", 30000))
        _engine = create_engine(
            config.DATABASE_URL,
            connect_args={"options": f"-c statement_timeout={st_ms}"},
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
    return _engine


def reset_engine() -> None:
    """Dispose the current pool so a transient DB disconnect gets a fresh connection."""
    global _engine
    engine = _engine
    _engine = None
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass


def _validate_readonly_sql(sql: str) -> str | None:
    """校验 SQL 只读性，拦截非查询语句与多语句执行。"""
    sql_stripped = sql.strip()
    if not sql_stripped:
        return "安全拦截: SQL 不能为空。"
    if len(sql_stripped) > _MAX_SQL_TEXT_CHARS:
        return f"安全拦截: SQL 过长（>{_MAX_SQL_TEXT_CHARS} 字符），已拒绝执行。"

    normalized = _strip_sql_literals_and_comments(sql_stripped)
    sql_upper = normalized.upper()
    dangerous_pattern = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|GRANT|TRUNCATE|CREATE|REVOKE)\b", re.IGNORECASE)
    dangerous_match = dangerous_pattern.search(sql_upper)
    if dangerous_match:
        return f"安全拦截: 严禁执行包含 '{dangerous_match.group(1).upper()}' 的非只读语句！"

    if _has_multiple_sql_statements(sql_stripped):
        return "安全拦截: 严禁执行多语句 SQL。"

    if _CTE_WRITE_PATTERN.search(normalized):
        return "安全拦截: 检测到带副作用的 CTE 写操作。"

    if _ROW_LOCKING_PATTERN.search(normalized):
        return "安全拦截: 严禁执行带锁的查询（FOR UPDATE / LOCK TABLE / FOR SHARE）。"

    risky_match = _RISKY_FUNCTION_PATTERN.search(normalized)
    if risky_match:
        return f"安全拦截: 严禁执行高风险函数或语句片段 {risky_match.group(1)!r}。"

    allowed_prefixes = ("SELECT", "WITH")
    if not sql_upper.startswith(allowed_prefixes):
        return "安全拦截: 仅允许执行 SELECT / WITH 开头的只读查询。"

    return None


def _strip_sql_literals_and_comments(sql: str) -> str:
    """去除字符串字面量与注释，便于安全关键字检测。"""
    result = []
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    while i < len(sql):
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
                result.append("\n")
            else:
                result.append(" ")
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                result.extend([" ", " "])
                i += 2
            else:
                result.append(" ")
                i += 1
            continue

        if in_single:
            if c == "'" and nxt == "'":
                result.extend([" ", " "])
                i += 2
                continue
            if c == "'":
                in_single = False
            result.append(" ")
            i += 1
            continue

        if in_double:
            if c == '"':
                in_double = False
            result.append(" ")
            i += 1
            continue

        if c == "-" and nxt == "-":
            in_line_comment = True
            result.extend([" ", " "])
            i += 2
            continue

        if c == "/" and nxt == "*":
            in_block_comment = True
            result.extend([" ", " "])
            i += 2
            continue

        if c == "'":
            in_single = True
            result.append(" ")
            i += 1
            continue

        if c == '"':
            in_double = True
            result.append(" ")
            i += 1
            continue

        result.append(c)
        i += 1

    return "".join(result).strip()


def _apply_row_limit_if_missing(sql: str, max_rows: int) -> tuple[str, bool]:
    """在无最终结果 LIMIT 时追加 LIMIT，降低 pd.read_sql 一次性加载导致的 OOM 风险。

    使用剥离字面量/注释后的文本做 LIMIT 关键字词边界匹配，减少列名含 limit 的误伤。
    注意：大 GROUP BY 结果仍可能被截断，属用正确性换内存安全的防爆盾。
    """
    stripped = sql.strip()
    if max_rows <= 0:
        return stripped, False
    normalized = _strip_sql_literals_and_comments(stripped)
    if re.search(r"\bLIMIT\b", normalized, flags=re.IGNORECASE):
        return stripped, False
    core = stripped.rstrip().rstrip(";")
    # 用子查询包裹并换行闭合，避免末尾单行 `-- 注释` 把追加的 LIMIT 吞进注释导致防护失效
    wrapped = f"SELECT * FROM (\n{core}\n) AS _limit_wrapper LIMIT {int(max_rows)}"
    return wrapped, True


def _has_multiple_sql_statements(sql: str) -> bool:
    """检测是否包含多语句（忽略字符串与注释中的分号，允许末尾单个分号）。"""
    normalized = _strip_sql_literals_and_comments(sql)
    semicolon_indexes = [idx for idx, ch in enumerate(normalized) if ch == ";"]
    if not semicolon_indexes:
        return False

    last_non_ws = len(normalized.rstrip()) - 1
    if len(semicolon_indexes) == 1 and semicolon_indexes[0] == last_non_ws:
        return False
    return True


@tool
def execute_sql_and_save_geojson(
    sql: str,
    output_filename: str,
    geom_col: str = "geometry",
    out_dir: str | None = None,
) -> dict:
    """执行一条 PostGIS SQL，并将结果保存到工作区。若结果含几何列则用 GeoPandas 读入并保存为 GeoJSON，否则用 Pandas 保存为 JSON。

    始终使用 SQLAlchemy 引擎 + GeoPandas/Pandas，不使用 psycopg2 的 cursor.fetchall() 处理几何字段：
    - 有几何列时：gpd.read_postgis(sql, engine, geom_col) + gdf.to_file(..., driver="GeoJSON")；
    - 无几何列时：pd.read_sql + df.to_json()。

    Args:
        sql: 完整的 SQL 查询字符串。
        output_filename: 输出文件名（如 xxx.geojson 或 xxx.json）。
        geom_col: 几何列名，默认为 geometry。
        out_dir: 输出目录，默认使用 config.WORKSPACE_DIR。

    Returns:
        包含 success, path, row_count, has_geometry 等字段的字典。
    """
    workspace_dir = Path(config.WORKSPACE_DIR).resolve()
    base_dir = Path(out_dir).resolve() if out_dir else workspace_dir
    try:
        base_dir.relative_to(workspace_dir)
    except ValueError:
        return {
            "success": False,
            "path": "",
            "error": "安全拦截: 输出目录必须位于工作区内。",
        }

    safe_output_name = Path(output_filename).name
    if not safe_output_name or safe_output_name in {".", ".."}:
        return {
            "success": False,
            "path": "",
            "error": "安全拦截: 输出文件名不合法。",
        }

    out_path = (base_dir / safe_output_name).resolve()
    try:
        out_path.relative_to(workspace_dir)
    except ValueError:
        return {
            "success": False,
            "path": "",
            "error": "安全拦截: 输出文件必须位于工作区内。",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    validation_error = _validate_readonly_sql(sql)
    if validation_error:
        return {
            "success": False,
            "path": "",
            "error": validation_error,
        }

    engine = get_engine()

    try:
        sql_exec, applied_limit = _apply_row_limit_if_missing(sql, config.SQL_MAX_RESULT_ROWS)
        core_sql = sql_exec.strip().rstrip(";")
        exec_sql_obj = text(sql_exec.strip())

        idle_ms = int(getattr(config, "SQL_IDLE_IN_TRANSACTION_TIMEOUT_MS", 30000))
        actual_geom_col = None
        result_frame = None
        with engine.connect() as conn:
            conn.execute(text(f"SET LOCAL idle_in_transaction_session_timeout = {idle_ms}"))
            # LIMIT 0 探针：仅取列元数据，避免全表进 Pandas 再逐行解析 WKB
            probe_sql = text(f"SELECT * FROM ({core_sql}) AS _probe LIMIT 0")
            probe_df = pd.read_sql(probe_sql, conn)

            for col in [geom_col, "geom", "wkb_geometry", "shape", "geometry"]:
                if col in probe_df.columns:
                    actual_geom_col = col
                    break

            if actual_geom_col:
                result_frame = gpd.read_postgis(exec_sql_obj, conn, geom_col=actual_geom_col)
                if result_frame.crs is None:
                    result_frame = result_frame.set_crs("EPSG:4326", allow_override=True)
            else:
                result_frame = pd.read_sql(exec_sql_obj, conn)

        # File serialization can be much slower than the query. Close the
        # transaction first so idle_in_transaction_session_timeout cannot
        # terminate the backend while GeoPandas/Pandas writes a large result.
        if actual_geom_col:
            gdf = result_frame
            out_path = out_path.with_suffix(".geojson") if out_path.suffix.lower() != ".geojson" else out_path
            gdf.to_file(out_path, driver="GeoJSON")
            return {
                "success": True,
                "path": str(out_path),
                "row_count": len(gdf),
                "has_geometry": True,
                "applied_row_limit": applied_limit,
                "executed_sql": sql_exec,
            }

        df = result_frame
        if not str(out_path).endswith(".json"):
            out_path = out_path.with_suffix(".json")
        df.to_json(out_path, orient="records", force_ascii=False, indent=2)
        return {
            "success": True,
            "path": str(out_path),
            "row_count": len(df),
            "has_geometry": False,
            "applied_row_limit": applied_limit,
            "executed_sql": sql_exec,
        }
    except Exception as e:
        global _operational_retry_active
        if isinstance(e, OperationalError) and not _operational_retry_active:
            retry_func = getattr(execute_sql_and_save_geojson, "func", None)
            if callable(retry_func):
                _operational_retry_active = True
                reset_engine()
                try:
                    return retry_func(
                        sql=sql,
                        output_filename=output_filename,
                        geom_col=geom_col,
                        out_dir=out_dir,
                    )
                finally:
                    _operational_retry_active = False
        return {
            "success": False,
            "path": str(out_path),
            "applied_row_limit": False,
            "error": str(e),
        }
