# -*- coding: utf-8 -*-
"""Python 分析节点输入文件校验（无 geopandas 依赖，便于单测）。"""
import json
from pathlib import Path


def json_payload_contains_geometry(payload, max_scan_items: int = 50) -> bool:
    """判断 JSON 载荷是否包含可用于空间分析的几何信息。"""
    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
            for feat in payload.get("features", [])[:max_scan_items]:
                if isinstance(feat, dict):
                    geom = feat.get("geometry")
                    if isinstance(geom, dict) and geom.get("type") and geom.get("coordinates") is not None:
                        return True
            return False
        if payload.get("type") == "Feature":
            geom = payload.get("geometry")
            return isinstance(geom, dict) and geom.get("type") and geom.get("coordinates") is not None

        geom_val = payload.get("geometry")
        if geom_val is not None and str(geom_val).strip() not in ("", "null", "None"):
            return True

        scanned = 0
        for value in payload.values():
            if scanned >= max_scan_items:
                break
            scanned += 1
            if json_payload_contains_geometry(value, max_scan_items=max_scan_items):
                return True
        return False

    if isinstance(payload, list):
        for item in payload[:max_scan_items]:
            if json_payload_contains_geometry(item, max_scan_items=max_scan_items):
                return True
        return False

    return False


def validate_python_analysis_input_files(
    workspace_dir: Path,
    path_names: list[str],
    *,
    require_geojson: bool = True,
) -> tuple[bool, str, str | None]:
    """在进入 Python 沙盒前校验输入文件。

    第三项为结构化错误码：``requires_geometry`` | ``unreadable_json`` | ``None``。

    - require_geojson=True（空间分析）：批次中须至少存在一个已存在的 ``.geojson``；
      若仅有纯属性 ``.json`` 且无 ``.geojson``，则拦截。
    - require_geojson=False（纯统计）：允许仅含可读的非 GeoJSON ``.json``（无 geometry 也可放行）。
    """
    workspace_dir = workspace_dir.resolve()
    unreadable_files: list[str] = []
    has_existing_geojson = False

    for rel_path in path_names:
        rel_path_obj = Path(rel_path)
        file_name = rel_path_obj.name
        resolved_path = (workspace_dir / rel_path).resolve()

        if not resolved_path.exists():
            has_parent_parts = any(part not in ("", ".") for part in rel_path_obj.parts[:-1])
            if not rel_path_obj.is_absolute() and not has_parent_parts:
                fallback = (workspace_dir / file_name).resolve()
                resolved_path = fallback if fallback.exists() else resolved_path

        lower = str(resolved_path).lower()
        if lower.endswith(".geojson"):
            if resolved_path.exists():
                has_existing_geojson = True
            continue

        if lower.endswith(".json") and not lower.endswith(".geojson"):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                unreadable_files.append(str(rel_path))
                continue

    if unreadable_files:
        return (
            False,
            (
                "输入数据校验失败：以下 JSON 文件无法读取或解析，已阻止进入 Python 沙盒："
                + "、".join(unreadable_files)
            ),
            "unreadable_json",
        )

    if require_geojson and not has_existing_geojson:
        return (
            False,
            (
                "输入数据校验失败：当前任务为空间分析，批次中必须至少包含一个已存在的 .geojson 文件。"
                "请先通过 text2sql_tool 导出带几何列的数据。"
            ),
            "requires_geometry",
        )

    return True, "", None
