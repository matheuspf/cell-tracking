"""Verify shipped checkpoint provenance and overlap with our detector benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/incumbent-provenance-20260914"
INPUTS = Path("/kaggle/input/datasets/pilkwang")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    secondary = INPUTS / "biohub-temporal-unet3d-seed314159-v1"
    primary = INPUTS / "biohub-tracking-support-pack-50ep-v1"
    weights = secondary / "weights/unet_transformer/split_0"
    split = json.loads((weights / "split_manifest.json").read_text())
    snapshot = json.loads((weights / "SNAPSHOT_MANIFEST.json").read_text())
    config = json.loads((weights / "training_config.json").read_text())
    history = list(csv.DictReader((weights / "history.csv").open()))
    hashes = {}
    for name, expected in snapshot["files"].items():
        actual = sha(weights / name)
        assert actual == expected["sha256"], name
        hashes[name] = actual
    manifest = json.loads((secondary / "ARTIFACT_MANIFEST.json").read_text())
    assert manifest["model"]["weight_sha256"] == hashes["edge_predictor_best.pth"]
    assert manifest["model"]["training"] == config
    old = json.loads((REPO / "results/detector-screen-20260914/provenance.json").read_text())["receipts"]["incumbent/provenance.json"]
    assert old["weights"][1]["sha256"] == hashes["edge_predictor_best.pth"]
    primary_manifest = json.loads((primary / "ARTIFACT_MANIFEST.json").read_text())
    primary_sha = sha(primary / primary_manifest["model"]["weight_path"])
    assert primary_sha == primary_manifest["model"]["weight_sha256"] == old["weights"][0]["sha256"]
    primary_weight_dir = (primary / primary_manifest["model"]["weight_path"]).parent
    primary_record_files = sorted(p.name for p in primary_weight_dir.iterdir() if p.is_file())
    panel_path = REPO / "results/detector-screen-20260914/panel.json"
    panel = json.loads(panel_path.read_text())
    assessment = {r["dataset"] for r in panel["frames"] if r["role"] == "assessment"}
    train, monitor = set(split["train"]), set(split["test"])
    public_train = {p.stem for p in (REPO / "data/train").glob("*.zarr")}
    assert train == public_train and len(train) == config["train_datasets"] == 199
    assert len(monitor) == config["validation_datasets"] == 40
    assert monitor <= train and assessment <= train
    best = max(history, key=lambda r: float(r["validation_score"]))
    assert int(best["epoch"]) == snapshot["best_epoch"]
    assert float(best["validation_score"]) == snapshot["best_score"]
    assert len(history) == int(history[-1]["epoch"]) == snapshot["epoch"] == 400
    for r in history:
        assert abs(float(r["validation_score"]) - float(r["validation_acc"]) * float(r["validation_recall"])) < 1e-12
    trainer = secondary / "repo/scripts/train_unet_transformer.py"
    text = trainer.read_text()
    assert 'train_files = [data_dir / name for name in fold_data["train"]]' in text
    assert 'test_files = [data_dir / name for name in fold_data["test"]]' in text
    assert 'score = test_acc * test_recall' in text
    training_gaps = [x for x in ("snapshot_interval", "split_manifest", "training_config", "checkpoint_last", "eval_max_batches") if x not in text]
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "verified against shipped artifacts",
        "conclusion": "The exact incumbent secondary checkpoint declares training exposure to every assessment clip. The local detector benchmark is not held-out evaluation. No evidence of hidden competition-test-label access was found; these statements concern different kinds of exposure.",
        "primary": {"sha256": primary_sha, "artifact_name": primary_manifest["artifact_name"],
                    "weight_directory_files": primary_record_files,
                    "training_split_shipped": "split_manifest.json" in primary_record_files,
                    "training_history_shipped": "history.csv" in primary_record_files,
                    "training_run_provenance": "Incomplete; do not infer a 50-epoch horizon from the dataset slug."},
        "secondary": {"sha256": hashes["edge_predictor_best.pth"], "training_config": config,
                      "train_clips": len(train), "monitor_clips": len(monitor), "train_monitor_overlap": len(train & monitor),
                      "assessment_clips": len(assessment), "assessment_train_overlap": len(assessment & train),
                      "assessment_monitor_overlap": len(assessment & monitor), "assessment_datasets": sorted(assessment),
                      "monitor_datasets": sorted(monitor), "train_equals_all_public_train": train == public_train,
                      "monitor_clips_by_embryo": {embryo: sum(name.startswith(embryo+'_') for name in monitor) for embryo in sorted({name.split('_')[0] for name in monitor})},
                      "history_epochs": len(history), "best_epoch": int(best["epoch"]),
                      "best_monitor_acc_times_recall": float(best["validation_score"]),
                      "monitor_metric_definition": "edge-classification accuracy times detected-node recall",
                      "monitor_score_is_official_competition_metric": False, "snapshot_hashes_verified": hashes},
        "training_source": {"path": str(trainer), "sha256": sha(trainer), "source_available": True,
                            "semantics": "Reads the two split lists literally, trains on train, monitors on test, selects the best acc*recall checkpoint. No overlap exclusion is performed.",
                            "exact_run_reproducibility_gaps": training_gaps,
                            "interpretation": "The supplied trainer is useful source, but it lacks several recorded snapshot-run facilities. This is not a complete exact-run attestation."},
        "source_files": {str(p): sha(p) for p in (primary / "ARTIFACT_MANIFEST.json", secondary / "ARTIFACT_MANIFEST.json", weights / "SNAPSHOT_MANIFEST.json", trainer, panel_path)},
        "public_sources": ["https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1",
                           "https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
                           "https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion?scriptVersionId=347965685",
                           "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793"],
        "scope": "Checksums authenticate correspondence within the released bundle, not an independent attestation of historical training execution. A trained encoder carries exposure into downstream embeddings even if a new head is fitted only on the other embryo. No model or input was modified."
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"secondary_train": len(train), "train_monitor_overlap": len(train & monitor), "assessment_train_overlap": len(assessment & train), "assessment_monitor_overlap": len(assessment & monitor), "best_epoch": int(best["epoch"]), "hashes_verified": len(hashes), "training_source_available": True, "exact_run_gaps": training_gaps}))


if __name__ == "__main__":
    main()
