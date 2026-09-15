#!/usr/bin/env python3
"""Read-only v7 inventory. No downloads, imports of Ultrack, or baseline runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

GIB = 2**30
RESERVE_GIB = 8
BASE = "fb5521629eb41c8c485b291a5bcf344944c113ae"
INHERITED_BLOBS = {
    "AGENTS.md": "d8a4abb775eca74640a7e322b678ff0efef03d96",
    "tools/segmentation_tracking_v6/ultrack_adapter.py": "31391707541b25cbda5c82fd4bd0ae58d78d2d47",
    "tools/segmentation_tracking_v6/controls.py": "1ad84f0e780f0c97c55095ddbe319991e9729a24",
    "tools/segmentation_tracking_v6/common.py": "66d7771c5da0d59ff2edeb3ce0c6a207473353f3",
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def existing_parent(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        if path.parent == path:
            raise ValueError("No existing ancestor for path")
        path = path.parent
    return path if path.is_dir() else path.parent


def admitted(free_bytes: int, allocation_gib: float) -> bool:
    if not math.isfinite(allocation_gib) or allocation_gib < 0:
        raise ValueError("Projected allocation must be finite and nonnegative")
    return free_bytes >= (RESERVE_GIB + allocation_gib) * GIB


def run(command: list[str], *, timeout: int = 15) -> dict:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def probe_python(executable: str) -> dict:
    code = '''import importlib.util,importlib.metadata,json,sys
out={"python":sys.version.split()[0],"packages":{}}
for name in ("ultrack","mip","zarr","torch","cellpose","detectron2"):
    spec=importlib.util.find_spec(name)
    try: version=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: version=None
    out["packages"][name]={"found":spec is not None,"version":version}
print(json.dumps(out))'''
    result = run([executable, "-I", "-c", code])
    if result["returncode"] != 0:
        return {"executable": executable, "error": result["stderr"]}
    try:
        return {"executable": executable, **json.loads(result["stdout"])}
    except json.JSONDecodeError:
        return {"executable": executable, "error": "Non-JSON environment probe output"}


def inspect(repo: Path, scratch: Path, allocation_gib: float, pythons: list[str]) -> dict:
    repo, scratch = repo.resolve(), scratch.resolve()
    checks, blockers = {}, []
    for name, expected in INHERITED_BLOBS.items():
        path = repo / name
        observed = blob_sha(path.read_bytes()) if path.is_file() else None
        checks[name] = {"expected_blob": expected, "observed_blob": observed,
                        "matches_inspected_parent": observed == expected}
        if observed != expected:
            blockers.append(f"Inspect inherited file drift or missing file: {name}")
    ancestry = run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE, "HEAD"])
    if ancestry["returncode"] != 0:
        blockers.append("Inspected base ancestry not verified; inspect history/worktree before execution")
    status = run(["git", "-C", str(repo), "status", "--porcelain"])
    parent = existing_parent(scratch)
    free = shutil.disk_usage(parent).free
    space_ok = admitted(free, allocation_gib)
    if not space_ok:
        blockers.append("Scratch lacks 8 GiB reserve plus projected allocation; choose an admitted mount")
    if not os.access(parent, os.W_OK):
        blockers.append("Scratch parent is not writable")
    probes = [probe_python(executable) for executable in pythons]
    if not any(p.get("packages", {}).get("ultrack", {}).get("found") for p in probes):
        blockers.append("Ultrack absent from probed runtimes; V700 isolated provisioning required")
    available_ram = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_ram = int(line.split()[1]) * 1024
    ram_scratch = scratch == Path("/dev/shm") or Path("/dev/shm") in scratch.parents
    if ram_scratch and (available_ram is None or not admitted(available_ram, allocation_gib)):
        blockers.append("RAM-backed scratch lacks verified available RAM plus reserve")
    return {
        "status": "inventory_only", "actual_ultrack_execution": False,
        "base_commit": BASE, "base_ancestor_verified": ancestry["returncode"] == 0,
        "worktree_dirty": bool(status["stdout"]) if status["returncode"] == 0 else None,
        "inherited_files": checks, "python_probes": probes,
        "scratch": {"path": str(scratch), "existing_parent": str(parent),
                    "free_gib": free/GIB, "projected_allocation_gib": allocation_gib,
                    "reserve_gib": RESERVE_GIB, "capacity_admitted": space_ok,
                    "writable_parent": os.access(parent, os.W_OK), "ram_backed": ram_scratch},
        "available_ram_gib": None if available_ram is None else available_ram/GIB,
        "blockers": blockers,
        "manual_checks_required": ["actual active jobs and process ownership", "installation/download peak size",
                                   "clip metadata and source/model hashes", "actual CBC solve and mask export",
                                   "per-clip peak memory including database and worker duplication"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scratch", type=Path, default=Path("/kaggle/working/cell-tracking/ultrack-integration-v7"))
    parser.add_argument("--allocation-gib", type=float, default=0.0,
                        help="Additional next-stage allocation; zero inventories capacity only")
    parser.add_argument("--python", dest="pythons", action="append", help="Interpreter to probe; repeatable")
    parser.add_argument("--output", type=Path, help="Optional NEW local-only receipt; never overwritten")
    args = parser.parse_args()
    if not math.isfinite(args.allocation_gib) or args.allocation_gib < 0:
        parser.error("--allocation-gib must be finite and nonnegative")
    report = inspect(args.repo, args.scratch, args.allocation_gib, args.pythons or [sys.executable])
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
