#!/usr/bin/env python3
"""Read-only handover consistency check; no training, downloads or data access."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DOCS = ("CODEX_PROMPT.md", "EVIDENCE.md", "PLAN.md", "IMPLEMENTATION.md", "VALIDATION.md")


def validate(root: Path) -> dict:
    root = root.resolve()
    directory = root / "handover/pipeline-error-training-20260915"
    study = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for name in DOCS:
        if not (directory / name).is_file():
            errors.append(f"Missing handover: {name}")
    for value in study["required_read_paths"]:
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"Missing or unsafe evidence path: {value}")
    for name in ("base_commit", "base_tree", "metric_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", study[name]):
            errors.append(f"Invalid SHA: {name}")
    rows = study["experiments"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate experiment IDs")
    seen: set[str] = set()
    for row in rows:
        missing = set(row["depends_on"]) - seen
        if missing:
            errors.append(f"Forward/unknown dependency for {row['id']}: {sorted(missing)}")
        seen.add(row["id"])
    baselines = study["baselines"]
    if set(baselines) != {"P0", "C4_m6", "C0"}:
        errors.append("Baseline identities changed")
    for name, row in baselines.items():
        if row["clips"] != 199 or row["nodes"] != 4108943:
            errors.append(f"Incomplete baseline population: {name}")
        if sum(row["division_counts"][::2]) != 151:
            errors.append(f"Wrong division truth denominator: {name}")
        if row["edge_counts"][0] + row["edge_counts"][2] != 128883:
            errors.append(f"Wrong edge truth denominator: {name}")
    forbidden = ("knob_sweeps", "target_label_fitting", "global_relinking",
                 "unmatched_proposals_as_negatives", "auto_resume_old_studies",
                 "auto_promote_default", "kaggle_submission")
    for key in forbidden:
        if study["constraints"][key] is not False:
            errors.append(f"Forbidden behavior enabled: {key}")
    limits = study["limits"]
    if sum(study["gpu_hour_reservation"].values()) != limits["measured_gpu_hours"]:
        errors.append("GPU reservation does not match campaign budget")
    if limits["gpu_total_gib"] > 20 or limits["process_tree_rss_gib"] > 50:
        errors.append("Resource safety limits exceed handover envelope")
    if limits["total_cpu_threads"] > study["hardware"]["cpu_cores"]:
        errors.append("CPU thread budget exceeds supplied cores")
    git = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            git = result.stdout.strip()
    return {"ok": not errors, "errors": errors, "experiments": len(ids),
            "read_paths": len(study["required_read_paths"]), "checkout": git,
            "study_status": study["status"], "scope": "Plan consistency only; no experiment or artifact validation"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = validate(args.repo)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"ok": False, "errors": [str(exc)], "scope": "Plan consistency only"}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
