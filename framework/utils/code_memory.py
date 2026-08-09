# -*- coding: utf-8 -*-
"""代码模板加载与记忆库存取逻辑，从 agent/tools.py 迁出以降低职责耦合。"""
import difflib
import json
import logging
import re
import time
import uuid
from pathlib import Path

from filelock import FileLock
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from utils.code_utils import strip_markdown_json


class CodeAbstraction(BaseModel):
    abstracted_code: str = Field(description="去除硬编码文件名和特定列名的通用 Python 代码模板")
    semantic_description: str = Field(description="20字以内的精确中文功能描述")


def parse_code_abstraction_llm_text(raw: str) -> CodeAbstraction | None:
    """从 LLM 原始文本中解析 CodeAbstraction（strip_markdown_json + 可选首尾大括号切片）。"""
    s = strip_markdown_json((raw or "").strip())
    try:
        return CodeAbstraction.model_validate(json.loads(s))
    except Exception:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            return CodeAbstraction.model_validate(json.loads(s[i : j + 1]))
        except Exception:
            pass
    return None


def load_code_templates(slots: dict | None) -> list[str]:
    """根据意图槽位从记忆库中检索匹配的代码模板（最多返回 3 个）。"""
    registry_path = config.CODE_TEMPLATES_DIR / "registry.json"
    if not registry_path.exists():
        return []
    try:
        lock_path = registry_path.with_suffix(".json.lock")
        with FileLock(str(lock_path), timeout=5):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
    except Exception:
        return []
    templates = registry.get("templates", [])
    if not templates:
        return []

    pred = (slots or {}).get("spatial_predicate") or ""
    method = (slots or {}).get("analytical_method") or ""
    query_str = f"{pred} {method}".strip()
    base_dir = config.CODE_TEMPLATES_DIR.resolve()

    scored_templates = []
    for t in templates:
        fp = t.get("filepath")
        if not fp:
            continue
        path = config.CODE_TEMPLATES_DIR / fp if not Path(fp).is_absolute() else Path(fp)
        try:
            resolved_path = path.resolve()
        except OSError:
            logging.warning(f"[Memory Cache] 模板路径无法解析，已跳过: {fp}")
            continue
        if resolved_path != base_dir and base_dir not in resolved_path.parents:
            logging.warning(f"[Memory Cache] 模板路径越界，已跳过: {resolved_path}")
            continue
        if not resolved_path.exists():
            continue

        score = 0.0
        if pred and pred in (t.get("spatial_predicate") or ""):
            score += 0.5
        if method and method in (t.get("analytical_method") or ""):
            score += 0.5
        desc = t.get("description") or ""
        if query_str and desc:
            score += difflib.SequenceMatcher(None, query_str, desc).ratio() * 0.5

        scored_templates.append((score, resolved_path))

    scored_templates.sort(key=lambda x: x[0], reverse=True)
    # 过滤得分过低的结果，前三名
    out = []
    for score, path in scored_templates:
        if score <= 0.1:
            continue
        try:
            out.append(path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"[Memory Cache] 模板读取失败，已跳过 {path}: {e}")
    return out[:3]


def save_code_to_memory(code: str, slots: dict | None) -> None:
    """将成功执行的代码抽象化后存入记忆库，供后续 RAG 检索复用。

    文件名使用 时间戳 + 短 UUID 避免并发覆盖；
    registry.json 的读写通过 FileLock 保护，防止并发写入损坏。
    """
    method = (slots or {}).get("analytical_method") or (slots or {}).get("spatial_predicate") or "spatial"
    safe = re.sub(r"[^\w]", "_", method)[:30]
    # 时间戳 + 短 UUID，防止秒级并发产生文件名冲突
    fname = f"{safe}_{int(time.time())}_{uuid.uuid4().hex[:6]}.py"
    semantic_description = "通用空间/统计分析模板"
    content_to_save = code

    abstraction_system = (
        "你是优秀的 GIS / 地理空间 Python 代码抽象专家。"
        "你必须只输出一行合法 JSON，不要 Markdown 代码块、不要前缀说明（如「功能概括：」）。"
        '格式：{"abstracted_code": "...", "semantic_description": "..."}。'
    )
    abstraction_prompt = (
        "任务：对下面这段用于特定任务的 Python 脚本做适度抽象化，便于日后复用为模板。\n\n"
        "要求：\n"
        "1) 保留 `gpd.read_file('xxx.geojson')` 等文件读写与路径形态，不要把具体文件名改成未定义的占位变量（如 INPUT_FILE），避免 NameError。\n"
        "2) 可将硬编码的业务列名、中文字符串过滤条件等替换为合理的通用变量名或注释说明。\n"
        "3) 抽象后的代码仍应是可执行的合法 Python。\n"
        "4) semantic_description：一句中文概括功能，20 字以内。\n\n"
        "仅输出一行 JSON，键为 abstracted_code（字符串）与 semantic_description（字符串）。\n\n"
        "原始代码：\n"
    ) + code

    try:
        llm = config.get_llm()
        resp = llm.invoke([
            SystemMessage(content=abstraction_system),
            HumanMessage(content=abstraction_prompt),
        ])
        raw_content = getattr(resp, "content", "") or ""
        if isinstance(raw_content, list):
            raw_content = "".join(
                str(x.get("text", x)) if isinstance(x, dict) else str(x)
                for x in raw_content
            )
        result = parse_code_abstraction_llm_text(str(raw_content))
        if result and (result.abstracted_code or "").strip():
            content_to_save = result.abstracted_code.strip()
            if (result.semantic_description or "").strip():
                semantic_description = result.semantic_description.strip()
    except Exception as e:
        logging.warning(f"[Memory Cache] 抽象化生成失败，采用原始代码降级存储: {e}")

    try:
        config.CODE_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        path = config.CODE_TEMPLATES_DIR / fname
        path.write_text(content_to_save, encoding="utf-8")

        registry_path = config.CODE_TEMPLATES_DIR / "registry.json"
        lock_path = registry_path.with_suffix(".json.lock")
        with FileLock(str(lock_path), timeout=10):
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"description": "动态技能记忆库", "templates": []}
            if "templates" not in data:
                data["templates"] = []
            data["templates"].append({
                "filepath": fname,
                "spatial_predicate": (slots or {}).get("spatial_predicate"),
                "analytical_method": (slots or {}).get("analytical_method"),
                "description": semantic_description,
            })
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        workspace_script = config.WORKSPACE_DIR / "last_successful_script.py"
        workspace_script.write_text(code, encoding="utf-8")
    except Exception as e:
        logging.warning(f"[Memory Cache] 写入模板或注册表失败: {e}")
