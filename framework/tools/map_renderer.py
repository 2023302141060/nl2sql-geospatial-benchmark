# -*- coding: utf-8 -*-
"""将 GeoJSON 渲染为 Folium 交互地图 HTML。"""
from pathlib import Path

import geopandas as gpd
import folium
from langchain_core.tools import tool

import config


@tool
def render_spatial_map(
    geojson_path: str,
    output_html: str = "map.html",
    out_dir: str | None = None,
) -> str:
    """将指定 GeoJSON 文件渲染为 Folium 交互地图并保存为 HTML。

    Args:
        geojson_path: GeoJSON 文件路径（绝对路径或相对于工作区的文件名）。
        output_html: 输出 HTML 文件名。
        out_dir: 输出目录，默认使用 config.WORKSPACE_DIR。

    Returns:
        生成的 HTML 文件绝对路径。
    """
    workspace_dir = Path(config.WORKSPACE_DIR).resolve()
    path = Path(geojson_path)
    if not path.is_absolute():
        path = (workspace_dir / geojson_path).resolve()
    if not path.exists():
        return f"错误：文件不存在 {path}"

    # 拦截无空间属性的纯 JSON 文件，避免 GeoPandas 报错
    if str(path).lower().endswith(".json") and not str(path).lower().endswith(".geojson"):
        return f"提示：文件 {path.name} 为纯属性数据，不包含空间几何信息，已自动跳过地图渲染。"

    base_dir = Path(out_dir).resolve() if out_dir else workspace_dir
    try:
        base_dir.relative_to(workspace_dir)
    except ValueError:
        return "错误：输出目录必须位于工作区内。"

    safe_output_name = Path(output_html).name
    if not safe_output_name or safe_output_name in {".", ".."}:
        return "错误：输出 HTML 文件名不合法。"

    out_path = (base_dir / safe_output_name).resolve()
    try:
        out_path.relative_to(workspace_dir)
    except ValueError:
        return "错误：输出 HTML 文件必须位于工作区内。"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(path)
    if gdf.empty:
        return f"错误：文件 {path.name} 不包含任何要素，无法渲染地图。"
    if getattr(gdf, "geometry", None) is None:
        return f"错误：文件 {path.name} 不包含几何列，无法渲染地图。"

    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return f"错误：文件 {path.name} 的几何列全部为空，无法渲染地图。"

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    MAX_FEATURES = 3000
    warning_msg = ""
    if len(gdf) > MAX_FEATURES:
        warning_msg = (
            f"（注：数据量高达 {len(gdf)} 条，为防止浏览器卡死，已随机抽样展示其中 {MAX_FEATURES} 条。）"
        )
        gdf = gdf.sample(n=MAX_FEATURES, random_state=42)

    try:
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.005, preserve_topology=True)
    except Exception:
        pass

    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    if len(bounds) != 4 or any(value is None for value in bounds):
        return f"错误：文件 {path.name} 的空间范围无效，无法渲染地图。"

    lat = (bounds[1] + bounds[3]) / 2
    lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[lat, lon], zoom_start=8)
    folium.GeoJson(gdf, name="geojson").add_to(m)
    m.save(str(out_path))
    return str(out_path.resolve()) + (f" {warning_msg}" if warning_msg else "")
