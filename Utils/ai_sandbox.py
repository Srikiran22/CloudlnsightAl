"""Safe AST-Validated Sandboxed Code Execution Engine for Tabular Data Analysis.

Provides execution of Python/Pandas analytical snippets against DataFrames:
1. Rejects dangerous AST nodes (imports, system/os/eval/exec calls, dunder lookups).
2. Restricts runtime scope strictly to pandas, numpy, plotly.express, math, and an isolated clone of the active DataFrame.
3. Captures stdout, generated Plotly figures, and computed scalar/DataFrame results.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from Utils.logsys import get_logger

logger = get_logger("AISandbox")

BANNED_FUNCTIONS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "input",
    "exit",
    "quit",
    "breakpoint",
    "memoryview",
}

BANNED_MODULE_NAMES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "importlib",
    "pickle",
    "shelve",
    "ctypes",
}


class SecurityViolationError(ValueError):
    """Raised when code violates sandboxed execution security policies."""


class SandboxedASTValidator(ast.NodeVisitor):
    """AST visitor enforcing strict sandbox isolation."""

    def visit_Import(self, node: ast.Import) -> None:
        raise SecurityViolationError("Dynamic imports are forbidden in the sandboxed analytical engine.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SecurityViolationError("Dynamic imports are forbidden in the sandboxed analytical engine.")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in BANNED_FUNCTIONS or func_name in BANNED_MODULE_NAMES:
                raise SecurityViolationError(f"Function '{func_name}' is forbidden in the sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise SecurityViolationError(f"Access to private/dunder attribute '{node.attr}' is forbidden.")
        if isinstance(node.value, ast.Name) and node.value.id in BANNED_MODULE_NAMES:
            raise SecurityViolationError(f"Access to module '{node.value.id}' is forbidden.")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise SecurityViolationError("Asynchronous functions are unsupported in sandbox execution.")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        raise SecurityViolationError("Asynchronous loops are unsupported in sandbox execution.")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        raise SecurityViolationError("Asynchronous context managers are unsupported in sandbox execution.")


@dataclass
class SandboxResult:
    """Structured result of sandboxed code execution."""

    success: bool
    code: str
    stdout: str
    result: Any | None
    fig: Any | None
    result_df: pd.DataFrame | None
    error: str | None


def execute_sandboxed_code(
    code_str: str,
    df: pd.DataFrame,
    max_output_rows: int = 1000,
) -> SandboxResult:
    """Validate and execute a Python snippet in a sandboxed, isolated data environment."""
    clean_code = code_str.strip()
    # Strip markdown code fences if present
    if clean_code.startswith("```"):
        lines = clean_code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_code = "\n".join(lines).strip()

    # 1. AST Static Security Audit
    try:
        tree = ast.parse(clean_code)
        validator = SandboxedASTValidator()
        validator.visit(tree)
    except SecurityViolationError as sec_err:
        logger.warning("Sandbox security violation rejected: %s", sec_err)
        return SandboxResult(
            success=False,
            code=clean_code,
            stdout="",
            result=None,
            fig=None,
            result_df=None,
            error=f"Security Policy Violation: {sec_err}",
        )
    except SyntaxError as syn_err:
        return SandboxResult(
            success=False,
            code=clean_code,
            stdout="",
            result=None,
            fig=None,
            result_df=None,
            error=f"Syntax Error in generated code: {syn_err}",
        )

    # 2. Build isolated execution environment
    # Clone DataFrame to guarantee original dataset immutability
    df_clone = df.copy()

    safe_builtins = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "abs": abs,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "isinstance": isinstance,
        "print": print,
    }

    sandbox_globals = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "px": px,
        "math": math,
        "df": df_clone,
    }

    sandbox_locals: dict[str, Any] = {}
    stdout_buffer = io.StringIO()

    # 3. Handle last-statement return value (if expression)
    compiled_code = compile(tree, "<sandbox>", "exec")
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compiled_code, sandbox_globals, sandbox_locals)
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.info("Sandbox execution runtime failure: %s", err_msg)
        return SandboxResult(
            success=False,
            code=clean_code,
            stdout=stdout_buffer.getvalue(),
            result=None,
            fig=None,
            result_df=None,
            error=err_msg,
        )

    # 4. Extract figures and return results
    fig = sandbox_locals.get("fig") or sandbox_globals.get("fig")
    # Verify it is a Plotly figure if present
    if fig is not None and not hasattr(fig, "to_dict"):
        fig = None

    raw_result = sandbox_locals.get("result") or sandbox_globals.get("result")
    result_df = None

    if isinstance(raw_result, pd.DataFrame):
        result_df = raw_result.head(max_output_rows)
    elif isinstance(raw_result, pd.Series):
        result_df = raw_result.head(max_output_rows).to_frame(name="Value")
    elif fig is None and raw_result is None and "df" in sandbox_locals:
        # Check if df was transformed into a new DataFrame
        mutated = sandbox_locals["df"]
        if isinstance(mutated, pd.DataFrame) and not mutated.equals(df_clone):
            result_df = mutated.head(max_output_rows)

    return SandboxResult(
        success=True,
        code=clean_code,
        stdout=stdout_buffer.getvalue(),
        result=raw_result,
        fig=fig,
        result_df=result_df,
        error=None,
    )
