#!/usr/bin/env python3
"""Read-only local inventory. No imports of torch/zarr or reads of image/label chunks."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
from contracts import ContractError, load_study, sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GIB = 1024 ** 3


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def memory_receipt() -> dict[str, float]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in {"MemTotal:", "MemAvailable:"}:
            values[parts[0].rstrip(":") + "_gib"] = int(parts[1]) * 1024 / GIB
    return values


def nearest_existing(path: Path) -> Path:
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def inspect_metadata(data_root: Path, study: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    clips = []
    counts: dict[str, int] = {}
    train = data_root / "train"
    for path in sorted(train.glob("*.zarr")) if train.is_dir() else []:
        clip = path.stem
        embryo = clip.split("_")[0]
        counts[embryo] = counts.get(embryo, 0) + 1
        metadata = path / "0" / "zarr.json"
        record: dict[str, Any] = {"clip_id": clip, "embryo": embryo, "has_paired_geff": path.with_suffix(".geff").exists()}
        try:
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            shape = meta["shape"]
            if (not isinstance(shape, list) or len(shape) != 4
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or shape[1:] != study["data"]["native_zyx"]):
                errors.append(f"Unexpected native TZYX shape: {clip}: {shape}")
            record.update(shape_tzyx=shape, dtype=meta.get("data_type"), metadata_sha256=sha256(metadata))
            if meta.get("data_type") != "uint16":
                errors.append(f"Unexpected dtype: {clip}: {meta.get('data_type')}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Cannot inspect metadata for {clip}: {exc}")
        if not record["has_paired_geff"]:
            errors.append(f"Missing paired GEFF path: {clip}")
        clips.append(record)
    if len(clips) != study["data"]["expected_clips"]:
        errors.append(f"Expected {study['data']['expected_clips']} training clips; found {len(clips)}")
    if counts != study["data"]["expected_embryo_clips"]:
        errors.append(f"Embryo clip inventory mismatch: {counts}")
    return {"root": str(data_root), "clip_count": len(clips), "embryo_clip_counts": counts, "clips": clips, "errors": errors,
            "image_chunks_read": False, "geff_labels_read": False, "spacing_verified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=HERE / "study.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        study = load_study(args.study)
    except (OSError, ValueError, ContractError) as exc:
        print(json.dumps({"status": "invalid_study", "error": str(exc)}, indent=2))
        return 2
    candidates = [REPO / study["data"]["default_alias"], Path(study["data"]["canonical_root"])]
    data_root = args.data_root or next((p for p in candidates if (p / "train").is_dir()), candidates[0])
    output_root = (args.output_root or REPO / study["data"]["output_root"]).resolve()
    inventory = inspect_metadata(data_root.resolve(), study)
    gpu = run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    mem = memory_receipt()
    disk_path = nearest_existing(output_root)
    disk_free = shutil.disk_usage(disk_path).free / GIB
    errors = list(inventory["errors"])
    if output_root.exists() and not output_root.is_dir():
        errors.append("Output root exists but is not a directory")
    warnings = []
    if gpu.get("returncode") != 0:
        errors.append("nvidia-smi unavailable/failed; inspect GPU runtime locally")
    elif "4090" not in gpu.get("stdout", ""):
        warnings.append("GPU listing does not identify the expected RTX 4090; do not assume capacity")
    if mem and mem.get("MemAvailable_gib", 0) < study["resources"]["min_system_available_gib"]:
        errors.append("System available RAM is below the reserved headroom")
    if not mem:
        warnings.append("No Linux /proc/meminfo; measure host RAM/process-tree usage separately")
    if disk_free < study["resources"]["min_durable_free_gib"]:
        errors.append("Output filesystem free space is below the durable reserve")
    if not os.access(disk_path, os.W_OK):
        errors.append("Output ancestor is not writable by this process")
    receipt = {
        "status": "ready_for_D900_D920_audit_not_training_certification" if not errors else "needs_local_resolution",
        "scope": "Read-only metadata/hardware preflight; training, spacing, complete rules, provenance and GPU fit are not verified",
        "python": sys.version, "python_executable": sys.executable, "study_sha256": sha256(args.study),
        "repo_root": str(REPO), "git": run_command(["git", "--no-optional-locks", "-C", str(REPO), "status", "--short", "--branch"]),
        "gpu": gpu, "host_memory": mem, "output_root": str(output_root), "output_filesystem_checked_at": str(disk_path),
        "output_filesystem_free_gib": disk_free, "inventory": inventory, "errors": errors, "warnings": warnings,
        "writes_performed": False,
    }
    print(json.dumps(receipt, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
