# -*- coding: utf-8 -*-
"""在隔离子进程中执行 Python 代码（受限执行器，而非真正安全沙盒）。

支持解析 stdout 为 JSON（status/answer_text/data_payload），解析失败时抛出异常供节点触发自愈重试。
"""
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

import config

# ---------------------------------------------------------------------------
# 安全警告 (Security Warning)
# ---------------------------------------------------------------------------
# 原有的 AST 静态黑名单扫描已被废除（因为极易被变量别名或字典 getattr 绕过，提供虚假的安全感）。
# 强烈建议在生产环境中，不要在本机裸机执行，而是将此机制替换为 Docker API 或 nsjail 等底层物理沙盒调用，
# 以实现真正的网络、文件系统、内存（OOM）等级别的隔离防线。



# ---------------------------------------------------------------------------
# 沙盒核心
# ---------------------------------------------------------------------------

class SandboxOutputParseError(ValueError):
    """沙盒脚本未按协议输出合法 JSON 时抛出，供节点捕获并触发重试。"""
    def __init__(self, message: str, raw_stdout: str = ""):
        super().__init__(message)
        self.raw_stdout = raw_stdout


class SandboxSecurityError(ValueError):
    """脚本触发本地执行安全策略时抛出。"""


_FORBIDDEN_IMPORT_MODULES = {
    "socket", "requests", "urllib", "urllib.request", "urllib3", "http", "http.client",
    "ftplib", "telnetlib", "websocket", "websockets", "asyncio", "aiohttp",
    "subprocess", "multiprocessing", "threading", "concurrent", "concurrent.futures",
    "ctypes", "cffi", "marshal", "pickle", "dill", "shelve", "dbm",
    "importlib", "runpy", "pathlib", "shutil",
}

_FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "breakpoint",
}

_FORBIDDEN_ATTR_ROOTS = {
    "os": {"system", "popen", "spawnl", "spawnlp", "spawnv", "spawnvp", "startfile", "remove", "unlink", "rmdir"},
    "sys": {"modules", "path", "path_hooks"},
    "subprocess": {"run", "Popen", "call", "check_call", "check_output"},
}


def _validate_python_code_security(code: str) -> None:
    """对 LLM 代码做最小化静态拦截，降低本地命令执行、网络访问与任意文件读写风险。"""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = str(alias.name or "").strip()
                if mod in _FORBIDDEN_IMPORT_MODULES or any(mod.startswith(f"{x}.") for x in _FORBIDDEN_IMPORT_MODULES):
                    raise SandboxSecurityError(f"安全拦截：禁止导入模块 {mod}")
        elif isinstance(node, ast.ImportFrom):
            mod = str(node.module or "").strip()
            if mod in _FORBIDDEN_IMPORT_MODULES or any(mod.startswith(f"{x}.") for x in _FORBIDDEN_IMPORT_MODULES):
                raise SandboxSecurityError(f"安全拦截：禁止导入模块 {mod}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
                raise SandboxSecurityError(f"安全拦截：禁止调用 {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                root = node.func.value.id
                attr = node.func.attr
                if attr in _FORBIDDEN_ATTR_ROOTS.get(root, set()):
                    raise SandboxSecurityError(f"安全拦截：禁止调用 {root}.{attr}()")


def _check_workspace_file_sizes(max_mb: int | None = None) -> str | None:
    """Windows 沙盒 OOM 保镖：防止超大 GeoJSON 等撑爆宿主内存（无 Unix resource 限制时）。"""
    if max_mb is None:
        max_mb = int(getattr(config, "SANDBOX_OOM_MAX_FILE_MB", 1024) or 1024)
    ws = Path(config.WORKSPACE_DIR)
    max_bytes = max_mb * 1024 * 1024
    guarded = {".json", ".geojson", ".csv"}
    for file_path in ws.glob("*.*"):
        if file_path.suffix.lower() not in guarded:
            continue
        try:
            if file_path.stat().st_size > max_bytes:
                return (
                    f"Sandbox OOM Guard: 文件 {file_path.name} 过大 (>{max_mb}MB)。"
                    "为防止 Windows 宿主内存溢出，沙盒拒绝加载执行。"
                    "请在 Text2SQL 阶段增加 LIMIT 或进行更细粒度的聚合。"
                )
        except OSError:
            continue
    return None


def _extract_json_object_candidates(text: str) -> list[str]:
    """从文本中扫描全部完整 JSON 对象候选串。

    逐字符追踪括号深度与字符串状态，确保多行 JSON 和嵌套结构均可正确提取。
    返回按出现顺序排列的所有完整 JSON 对象；未找到则返回空列表。
    """
    s = str(text or "")
    candidates: list[str] = []
    start_idx: int | None = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if depth == 0 and not in_string and ch != "{":
            continue

        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(s[start_idx:i + 1])
                start_idx = None

    return candidates


def _parse_protocol_json_from_stdout(stdout: str, required_fields: set[str]) -> dict | None:
    """尽量从 stdout 中解析最后一个符合协议的 JSON 对象。"""
    stripped_stdout = str(stdout or "").strip()
    if not stripped_stdout:
        return None

    def _is_valid_protocol(obj) -> bool:
        return isinstance(obj, dict) and required_fields.issubset(obj.keys())

    try:
        temp = json.loads(stripped_stdout)
        if _is_valid_protocol(temp):
            return temp
    except Exception:
        pass

    for line in reversed(stripped_stdout.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            temp = json.loads(line)
            if _is_valid_protocol(temp):
                return temp
        except Exception:
            pass

    candidates = _extract_json_object_candidates(stripped_stdout)
    for candidate in reversed(candidates):
        try:
            temp = json.loads(candidate)
            if _is_valid_protocol(temp):
                return temp
        except Exception:
            pass

    return None


@tool
def execute_python_sandbox(
    code: str,
    timeout: int | None = None,
    script_name: str | None = None,
    strict_json_output: bool = True,
    authorized_input_files: list[str] | None = None,
) -> dict:
    """在独立子进程中执行给定的 Python 代码，捕获 stdout 与 stderr。

    当 strict_json_output=True 时，会尝试从 stdout 中解析最后输出的 JSON 对象（包含 status, answer_text, data_payload）；解析失败则抛出 SandboxOutputParseError，
    供调用方捕获并触发自愈（重试）机制。

    Args:
        code: 完整的 Python 脚本内容。
        timeout: 超时秒数，默认使用 config.SANDBOX_TIMEOUT。
        script_name: 临时脚本文件名（默认为基于 uuid 生成以防并发冲突）。
        strict_json_output: 是否要求 stdout 可解析为 JSON；为 True 且解析失败时抛异常。

    Returns:
        包含 success, stdout, stderr, return_code 的字典；若 strict_json_output 且解析成功，
        会多出 parsed 字段（含 status, answer_text, data_payload）。
    """
    import uuid

    try:
        _validate_python_code_security(code)
    except SandboxSecurityError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }

    timeout = timeout or config.SANDBOX_TIMEOUT
    workspace = config.WORKSPACE_DIR
    workspace.mkdir(parents=True, exist_ok=True)

    oom_warning = _check_workspace_file_sizes()
    if oom_warning:
        out_oom: dict = {
            "success": False,
            "stdout": "",
            "stderr": oom_warning,
            "return_code": -1,
        }
        if strict_json_output:
            out_oom["parsed"] = {
                "status": "error",
                "answer_text": oom_warning,
                "data_payload": {},
            }
        return out_oom

    actual_script_name = script_name or f"sandbox_{uuid.uuid4().hex[:8]}.py"
    run_dir: Path | None = None
    input_basenames: set[str] = set()
    if authorized_input_files:
        run_dir = workspace / ".sandbox_runs" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        for raw in authorized_input_files:
            source = Path(str(raw))
            if not source.is_absolute():
                source = workspace / source.name
            source = source.resolve()
            try:
                source.relative_to(workspace.resolve())
            except ValueError:
                continue
            if not source.is_file():
                continue
            input_basenames.add(source.name)
            shutil.copy2(source, run_dir / source.name)
    execution_dir = run_dir or workspace
    script_path = execution_dir / actual_script_name

    memory_limit_mb = 2048
    safety_header = f"""
import sys
try:
    import resource
    max_mem_bytes = {memory_limit_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
except (ImportError, ValueError, OSError):
    pass
"""
    safe_code = safety_header + "\n" + code
    script_path.write_text(safe_code, encoding="utf-8")

    # 强制子进程 stdout/stderr 使用 UTF-8
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(execution_dir),
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        out = {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": result.returncode,
        }
        if strict_json_output:
            _REQUIRED_FIELDS = {"status", "answer_text", "data_payload"}
            parsed = _parse_protocol_json_from_stdout(stdout, _REQUIRED_FIELDS)

            # 协议字段完整性校验
            if not parsed:
                if result.returncode == 0:
                    raise SandboxOutputParseError(
                        f"无法从 stdout 解析出符合标准协议的 JSON 对象（需包含 {sorted(_REQUIRED_FIELDS)}）",
                        raw_stdout=stdout,
                    )
            else:
                out["parsed"] = parsed
                saved_files = parsed.get("saved_files") if isinstance(parsed, dict) else []
                if run_dir is not None and isinstance(saved_files, list):
                    copied: list[str] = []
                    for raw_saved in saved_files:
                        basename = Path(str(raw_saved)).name
                        if not basename or basename in input_basenames:
                            continue
                        source = run_dir / basename
                        if source.is_file():
                            shutil.copy2(source, workspace / basename)
                            copied.append(basename)
                    out["copied_saved_files"] = copied

        return out
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "return_code": -1,
        }
    except Exception as e:
        if isinstance(e, SandboxOutputParseError):
            raise
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }
    finally:
        # 临时脚本清理：文件被占用或并发删除失败时不影响主流程（故意吞掉 OSError）
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass
        if run_dir is not None:
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
                parent = run_dir.parent
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
