"""Inspect/patch an isolated Python export; never import or execute the notebook.

This helper does not certify runtime equivalence or create a Kaggle notebook.
Those measurements are the local Codex tasks in PLAN.md.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

TARGET = "OUTPUT_MOTION_RELINK"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assignment(tree: ast.Module) -> ast.Assign | ast.AnnAssign:
    candidates = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == TARGET:
                    candidates.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == TARGET:
                candidates.append(node)
    writes = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
              and n.id == TARGET and isinstance(n.ctx, (ast.Store, ast.Del))]
    if len(candidates) != 1 or len(writes) != 1 or candidates[0].value is None:
        raise ValueError("Require exactly one module-level assignment and no other writes to " + TARGET)
    return candidates[0]


def inspect_source(source: str) -> dict:
    tree = ast.parse(source)
    node = assignment(tree)
    flags = {}
    for item in tree.body:
        targets = item.targets if isinstance(item, ast.Assign) else []
        if isinstance(item, ast.AnnAssign):
            targets = [item.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("OUTPUT_"):
                flags[target.id] = ast.get_source_segment(source, item.value) if item.value else None
    return {
        "source_sha256": sha256(source.encode("utf-8")),
        "target": TARGET,
        "assignment_line": node.lineno,
        "original_expression": ast.get_source_segment(source, node.value),
        "output_flag_expressions": flags,
        "repair_functions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and any(w in n.name for w in ("motion", "relink", "gap", "division", "smooth", "filter_output"))],
        "runtime_effective_value_verified": False,
        "notebook_executed": False,
    }


def make_candidate(source: str) -> tuple[str, dict]:
    tree = ast.parse(source)
    node = assignment(tree)
    value = node.value
    if isinstance(value, ast.Constant) and value.value is False:
        raise ValueError("Source already disables motion; it is not the intended untouched control")
    # AST columns are UTF-8 byte offsets, not Python character offsets.
    lines = source.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
    raw = source.encode("utf-8")
    candidate = (raw[:start] + b"False" + raw[end:]).decode("utf-8")
    expected = copy.deepcopy(tree)
    assignment(expected).value = ast.Constant(value=False)
    if ast.dump(ast.parse(candidate), include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Unexpected AST change outside the registered assignment")
    compile(candidate, "isolated_no_motion_candidate.py", "exec")
    receipt = inspect_source(source)
    receipt.update({
        "candidate_sha256": sha256(candidate.encode("utf-8")),
        "scientific_changes": [{"target": TARGET, "replacement_expression": "False"}],
        "all_other_source_bytes_preserved": True,
        "runtime_parity_tested": False,
        "leaderboard_validated": False,
    })
    return candidate, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Reviewed original .py export")
    parser.add_argument("--candidate", type=Path, help="NEW isolated .py path; never overwrites")
    parser.add_argument("--receipt", type=Path, help="NEW JSON path; otherwise print to stdout")
    args = parser.parse_args()
    source = args.source.read_bytes().decode("utf-8")
    if args.source.suffix != ".py":
        parser.error("Use the Python export, not an .ipynb; this helper does not parse IPython magics")
    outputs = [p for p in (args.candidate, args.receipt) if p is not None]
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or args.source.resolve() in resolved:
        parser.error("Source, candidate, and receipt must be distinct paths")
    if any(p.exists() or p.is_symlink() for p in outputs):
        parser.error("Refusing to overwrite an existing output")
    try:
        if args.candidate:
            candidate, receipt = make_candidate(source)
        else:
            receipt = inspect_source(source)
    except (SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.candidate:
        with args.candidate.open("xb") as handle:
            handle.write(candidate.encode("utf-8"))
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        with args.receipt.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
