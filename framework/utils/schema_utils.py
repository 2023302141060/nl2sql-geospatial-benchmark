# -*- coding: utf-8 -*-
"""Schema 加载与摘要工具函数，从 agent/tools.py 迁出以降低职责耦合。"""
import copy
import logging
from typing import Any, TypedDict

import yaml

import config


class _SchemaCacheEntry(TypedDict):
    """单表缓存：原始 YAML 文本 + 仅解析一次的字典。"""

    raw: str
    parsed: dict[str, Any]


class _SchemaFileMeta(TypedDict):
    """缓存文件元信息，用于判断 schema 目录内容是否变化。"""

    path: str
    mtime_ns: int


# 全局内存缓存：table_name -> {raw, parsed}，safe_load 仅在首次扫描时执行一次
_SCHEMA_CACHE: dict[str, _SchemaCacheEntry] = {}
_SCHEMA_FILE_METAS: dict[str, _SchemaFileMeta] = {}


def clear_schema_cache() -> None:
    """清空 Schema 进程级缓存。

    供测试、`schema_retriever` 在检测到 YAML 变更并重建向量库时调用，
    避免 RAG 已索引新列而 `load_schemas_by_table_names` 仍读到旧 YAML 的撕裂状态。
    """
    _SCHEMA_CACHE.clear()
    _SCHEMA_FILE_METAS.clear()


def _collect_schema_file_metas() -> dict[str, _SchemaFileMeta]:
    """收集当前 schema 文件元信息，用于检测目录内容变化。"""
    metas: dict[str, _SchemaFileMeta] = {}
    for p in config.SCHEMAS_DIR.glob("*.yaml"):
        try:
            stat = p.stat()
        except OSError as e:
            logging.warning(f"读取 Schema 文件元信息失败 {p}: {e}")
            continue
        metas[str(p.resolve())] = _SchemaFileMeta(path=str(p.resolve()), mtime_ns=stat.st_mtime_ns)
    return metas


def _ensure_schema_cache() -> None:
    """惰性初始化 _SCHEMA_CACHE：首次调用时扫描 SCHEMAS_DIR，后续直接复用内存。"""
    current_metas = _collect_schema_file_metas()
    if _SCHEMA_CACHE and current_metas == _SCHEMA_FILE_METAS:
        return
    clear_schema_cache()
    _SCHEMA_FILE_METAS.update(current_metas)
    for p in config.SCHEMAS_DIR.glob("*.yaml"):
        try:
            raw = p.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
            if data and isinstance(data, dict) and data.get("table_name"):
                _SCHEMA_CACHE[data["table_name"]] = _SchemaCacheEntry(raw=raw, parsed=data)
        except Exception as e:
            logging.error(f"加载 Schema 文件 {p} 失败: {e}")
            continue


def load_all_schemas() -> list[dict]:
    """加载 SCHEMAS_DIR 下所有 YAML 文件，返回解析后的字典列表。

    使用内存缓存中的 parsed，避免每次调用重复 yaml.safe_load；返回深拷贝以免调用方修改污染缓存。
    """
    _ensure_schema_cache()
    return [copy.deepcopy(entry["parsed"]) for entry in _SCHEMA_CACHE.values()]


def get_schema_summaries(schemas: list[dict]) -> str:
    """将 Schema 列表格式化为单行摘要文本，供意图解析 Prompt 使用。"""
    lines = []
    for s in schemas:
        if not s:
            continue
        name = s.get("table_name", "")
        desc = s.get("table_description", "")
        domain = s.get("domain", "")
        lines.append(f"- table_name: {name}, domain: {domain}, description: {desc}")
    return "\n".join(lines) if lines else "（无 Schema）"


def load_schemas_by_table_names(table_names: list[str]) -> str:
    """根据表名列表从内存缓存中获取对应的 YAML Schema 文本（供 text2sql 回退使用）。

    使用进程级内存缓存，避免重复磁盘 I/O，且不依赖 schema_retriever 模块，
    消除 utils → tools 的反向调用链。
    """
    _ensure_schema_cache()
    yamls = []
    for name in table_names:
        entry = _SCHEMA_CACHE.get(name)
        if entry:
            yamls.append(entry["raw"])
    return "\n\n".join(yamls) if yamls else "（无 Schema）"
