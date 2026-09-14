"""Small stdlib contracts. These validate declared records, not actual training reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """A declared study or evaluation contract was violated."""


def load_study(path: Path) -> dict[str, Any]:
    study = json.loads(path.read_text(encoding="utf-8"))
    validate_study(study)
    return study


def validate_study(study: dict[str, Any]) -> None:
    try:
        v, r = study["validation"], study["resources"]
        directions = {(d["source"], d["target"]) for d in v["directions"]}
        if directions != {("44b6", "6bba"), ("6bba", "44b6")} or len(v["directions"]) != 2:
            raise ContractError("Both unique opposite-embryo directions are required")
        for key in ("target_image_training", "incumbent_teacher", "external_teacher",
                    "target_threshold_tuning", "allow_visible_test_validation",
                    "unmatched_predictions_are_false_positives", "automatic_promotion"):
            if v[key] is not False:
                raise ContractError(f"Forbidden validation option: {key}")
        if v["freeze_all_directions_seeds_predictions_before_scores"] is not True:
            raise ContractError("Freeze both directions and seeds before scores")
        if v["primary"] != "N_ema" or v["anchor"] != "N_base":
            raise ContractError("Fixed primary/anchor changed")
        if study["data"]["native_zyx"] != [64, 256, 256]:
            raise ContractError("Native acquired grid changed")
        if study["model"]["primary_input_resizing_allowed"] is not False:
            raise ContractError("Native inputs may be cropped, not resized")
        recipes = study["recipes"]
        ids = [x["id"] for x in recipes]
        if len(ids) != len(set(ids)):
            raise ContractError("Duplicate recipe ID")
        core = {x["id"]: x for x in recipes if x["tier"] in {"P0", "P1"}}
        if set(core) != {"N_base", "N_ema", "B64_stride", "N_lowpass"}:
            raise ContractError("Core comparisons are missing or changed")
        for recipe in recipes:
            if recipe["initialization"] != "random_source_only":
                raise ContractError("Non-source initialization")
            if not recipe["seeds"] or len(recipe["seeds"]) != len(set(recipe["seeds"])):
                raise ContractError("Missing or repeated seeds")
        for recipe in core.values():
            if recipe["seeds"] != study["training"]["seeds"] or recipe["schedule"] != "locked_final":
                raise ContractError("Core seeds/schedules must match")
        if len(study["training"]["seeds"]) != 2:
            raise ContractError("Two prespecified seeds are required")
        fits = sum(len(x["seeds"]) * len(directions) for x in recipes)
        if fits > r["max_retained_fits"]:
            raise ContractError("Registry exceeds retained-fit cap")
        if sum(r["initial_allocation_gpu_hours"].values()) > r["total_active_gpu_hours"]:
            raise ContractError("Resource allocation exceeds budget")
        if r["one_gpu_job_at_a_time"] is not True or study["hardware"]["gpu_workers"] != 1:
            raise ContractError("Only one GPU job is authorized")
        for key, value in study["permissions"].items():
            if value is not False:
                raise ContractError(f"Unexpected permission: {key}")
    except (KeyError, TypeError) as exc:
        raise ContractError(f"Malformed study: {exc}") from exc


def require_complete_frames(expected: dict[str, int], observed: dict[str, list[int]]) -> None:
    """Observed includes an entry even for frames with zero predicted points."""
    if not expected or set(expected) != set(observed):
        raise ContractError("Clip IDs must match exactly; intersections are invalid")
    for clip, count in expected.items():
        frames = observed[clip]
        if type(count) is not int or count < 1:
            raise ContractError(f"Invalid metadata frame count: {clip}")
        if any(type(t) is not int for t in frames):
            raise ContractError(f"Non-integer frame ID: {clip}")
        if len(frames) != count or set(frames) != set(range(count)):
            raise ContractError(f"Missing, duplicated or out-of-range frames: {clip}")


def require_source_ancestry(artifacts: dict[str, dict[str, Any]], root: str, source: str) -> None:
    """Audit transitive manifest declarations; do not mistake this for a read guard."""
    active: set[str] = set()
    done: set[str] = set()

    def visit(key: str) -> None:
        if key in active:
            raise ContractError(f"Cyclic ancestry: {key}")
        if key in done:
            return
        if key not in artifacts:
            raise ContractError(f"Missing ancestor: {key}")
        item = artifacts[key]
        if item.get("provenance") != "verified_source_only":
            raise ContractError(f"Unknown, external or exposed ancestor: {key}")
        embryos = item.get("fit_embryos")
        if not isinstance(embryos, list) or any(e != source for e in embryos):
            raise ContractError(f"Target or malformed fitting exposure: {key}")
        parents = item.get("parents")
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            raise ContractError(f"Malformed parents: {key}")
        if not embryos and item.get("kind") not in {"random_initialization", "fixed_code"}:
            raise ContractError(f"Missing declared fitting population: {key}")
        if not item.get("evidence"):
            raise ContractError(f"Missing provenance evidence: {key}")
        active.add(key)
        for parent in parents:
            visit(parent)
        active.remove(key)
        done.add(key)

    visit(root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
