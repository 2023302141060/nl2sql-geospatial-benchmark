# -*- coding: utf-8 -*-
"""代码提取与清洗工具函数，从 agent/tools.py 迁出以降低职责耦合。"""
import json
import re


def _match_balanced_json_fragment(s: str, start: int) -> int | None:
    """从 start（须为 `{` 或 `[`）起做引号/转义感知的括号匹配，返回闭合字符下标；失败返回 None。"""
    if start >= len(s) or s[start] not in "{[":
        return None
    stack: list[str] = []
    if s[start] == "{":
        stack.append("}")
    else:
        stack.append("]")
    in_string = False
    escape = False
    j = start + 1
    while j < len(s):
        ch = s[j]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            j += 1
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return j
        j += 1
    return None


def _first_valid_json_substring(text: str) -> str | None:
    """在文本中从左到右找首个可被 json.loads 解析的完整 `{...}` 或 `[...]` 片段。"""
    s = str(text or "")
    for i, c in enumerate(s):
        if c not in "{[":
            continue
        end = _match_balanced_json_fragment(s, i)
        if end is None:
            continue
        frag = s[i : end + 1]
        try:
            json.loads(frag)
        except Exception:
            continue
        return frag
    return None


def strip_markdown_json(raw: str) -> str:
    """
    鲁棒 JSON 提取：多段 Markdown 代码块时仍取最后一个块的内层文本（与历史行为一致）；
    再在该文本及必要时在全文中用栈匹配剥离首个可解析的 JSON 对象或数组，忽略前后废话。
    """
    s = str(raw or "").strip()
    if not s:
        return s

    matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    focus = matches[-1].strip() if matches else s

    extracted = _first_valid_json_substring(focus)
    if extracted is not None:
        return extracted
    if focus != s:
        extracted = _first_valid_json_substring(s)
        if extracted is not None:
            return extracted

    return focus


def extract_python_code(text: str) -> str:
    """从 LLM 输出文本中提取最后一个 Python 代码块内容。"""
    # 优先寻找 ```python ... ```
    python_matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if python_matches:
        return python_matches[-1].strip()
    # 退而求其次寻找通用的 ``` ... ```
    generic_matches = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_matches:
        return generic_matches[-1].strip()
    return text.strip()
