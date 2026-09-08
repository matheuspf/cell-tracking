#!/usr/bin/env python3
"""Normalize notebook exports and select the isolated local Jupyter kernel.

Kaggle notebooks may put future imports in later cells, which nbconvert preserves
at an invalid position in a standalone script. Relocate only top-level future
imports; leave embedded source strings and the archived notebooks unchanged.
"""

import ast
import hashlib
import json
from pathlib import Path
import shutil

from competition_paths import REPO_ROOT


def normalize_future_imports(source: str) -> str:
    try:
        compile(source, "<notebook-export>", "exec")
    except SyntaxError:
        pass
    else:
        return source
    tree = ast.parse(source)
    futures = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    if not futures:
        return source
    names = sorted({a.name for n in futures for a in n.names})
    removed = {i for n in futures for i in range(n.lineno - 1, n.end_lineno)}
    insertion = 0
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion = tree.body[0].end_lineno
    lines = source.splitlines(keepends=True)
    output = []
    for i in range(len(lines) + 1):
        if i == insertion:
            output.append("\nfrom __future__ import " + ", ".join(names) + "\n")
        if i < len(lines) and i not in removed:
            output.append(lines[i])
    return "".join(output)


def main() -> int:
    selection = json.loads((REPO_ROOT / "configs/notebooks.json").read_text())
    root = Path(selection["archive_root"])
    for row in selection["notebooks"]:
        directory = root / row["ref"]
        report_path = directory / "download-report.json"
        report = json.loads(report_path.read_text())
        exported = Path(report["export"]["path"])
        original = exported.read_text()
        fixed = normalize_future_imports(original)
        compile(fixed, str(exported), "exec")
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "get_ipython" for n in ast.walk(ast.parse(fixed))):
            raise RuntimeError(f"Translate IPython calls before terminal use: {exported}")
        runnable_python = Path(report["runnable_python"])
        if runnable_python.exists() and runnable_python.read_text() not in (original, fixed):
            raise RuntimeError(f"Preserving a modified runnable script: {runnable_python}")
        exported.write_text(fixed)
        shutil.copy2(exported, report["runnable_python"])
        runnable = Path(report["runnable_code"])
        notebook = json.loads(runnable.read_text())
        notebook.setdefault("metadata", {})["kernelspec"] = {
            "display_name": "Python (cell-tracking notebooks)",
            "language": "python",
            "name": "cell-tracking-notebooks",
        }
        runnable.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        report["export"].update({
            "syntax_ok": True,
            "syntax_validation": "Compiled with Python 3.12 after relocating top-level future imports",
            "issues": [],
            "sha256": hashlib.sha256(fixed.encode()).hexdigest(),
        })
        report.setdefault("local_setup", {}).update({
            "environment": selection["environment"],
            "kernel": "cell-tracking-notebooks",
            "future_imports_relocated": True,
            "full_pipeline_executed": False,
            "verified_public_score": row["public_score"],
            "published_version": row["version"],
        })
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Ready: {runnable} and {report['runnable_python']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
