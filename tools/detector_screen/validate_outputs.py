"""Validate completed detector artifacts and hash the realized predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "work/detector-screen-20260914"
FULL_METHODS = [
    "incumbent", "cellect", "organoid", "nucverse", "spotiflow",
    "spotiflow_loose", "xenopus", "xenopus_centroid", "pacmap",
    "cellpose_cpdino_vitb",
]
PILOT_METHODS = ["xenopus_loose", "anystar", "anystar_centroid", "focus_reference"]


def load_prediction(path: Path):
    with np.load(path, allow_pickle=False) as data:
        centers, scores = data["centers_zyx"], data["scores"]
        assert centers.ndim == 2 and centers.shape[1] == 3, path
        assert scores.shape == (len(centers),), path
        assert np.isfinite(centers).all() and np.isfinite(scores).all(), path
        return centers.copy(), scores.copy()


def validate_method(method, panel):
    rows = [r for r in panel["frames"] if method in FULL_METHODS or r["role"] == "pilot"]
    folder = ROOT / "predictions" / method
    expected = {r["key"] for r in rows}
    actual = {p.stem for p in folder.glob("*.npz")}
    assert actual == expected, (method, "missing", sorted(expected - actual), "extra", sorted(actual - expected))
    hashes, clipped, duplicates, count = {}, 0, 0, 0
    missing_receipt_fields = {}
    counts_by_role = {"pilot": 0, "assessment": 0}
    for row in rows:
        path = folder / (row["key"] + ".npz")
        centers, scores = load_prediction(path)
        rounded = np.rint(centers)
        native = np.clip(rounded, 0, np.asarray(row["shape"]) - 1)
        clipped += int(np.any(rounded != native, axis=1).sum())
        duplicates += len(native) - len(np.unique(native, axis=0))
        count += len(centers)
        counts_by_role[row["role"]] += len(centers)
        receipt_path = path.with_suffix(".json")
        if method not in ("incumbent", "focus_reference"):
            receipt = json.loads(receipt_path.read_text())
            assert receipt["key"] == row["key"], receipt_path
            for field in ("candidates", "count", "centers", "n_centers"):
                if field in receipt:
                    assert receipt[field] == len(centers), (receipt_path, field)
            if method.startswith("spotiflow"):
                threshold = .05 if method == "spotiflow_loose" else .3
                assert receipt["threshold"] == threshold, receipt_path
                assert receipt["input_shape"] == [64, 64, 64], receipt_path
                assert (scores >= threshold).all(), path
            if method.startswith("xenopus"):
                assert receipt["input_shape"] == [105, 256, 256], receipt_path
                if "n_tiles" in receipt:
                    assert receipt["n_tiles"] == [4, 2, 2], receipt_path
                else:
                    missing_receipt_fields["n_tiles"] = missing_receipt_fields.get("n_tiles", 0) + 1
                assert receipt["actual_zoom_zyx"] == [105 / 64, 1., 1.], receipt_path
        hashes[row["key"]] = {
            "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest() if receipt_path.exists() else None,
        }
    # Check that the two Spotiflow thresholds use the identical shared pass.
    if method == "spotiflow":
        for row in rows:
            centers, scores = load_prediction(folder / (row["key"] + ".npz"))
            loose_centers, loose_scores = load_prediction(ROOT / "predictions/spotiflow_loose" / (row["key"] + ".npz"))
            keep = loose_scores >= .3
            assert np.array_equal(centers, loose_centers[keep]), row["key"]
            assert np.array_equal(scores, loose_scores[keep]), row["key"]
    payload = (json.dumps(hashes, indent=2, sort_keys=True) + "\n").encode()
    manifest = ROOT / "evaluation/output-hashes" / (method + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(payload)
    return dict(method=method, frames=len(rows), candidates=count,
                candidates_by_role=counts_by_role,
                clipped_centers_after_rounding=clipped,
                duplicate_centers_after_rounding=duplicates,
                missing_optional_receipt_fields=missing_receipt_fields,
                output_manifest_sha256=hashlib.sha256(payload).hexdigest(),
                output_manifest=str(manifest.relative_to(ROOT)),
                finite_arrays_and_receipts_valid=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=FULL_METHODS + PILOT_METHODS)
    args = parser.parse_args()
    panel = json.loads((ROOT / "panel.json").read_text())
    path = ROOT / "evaluation/output-validation.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    results = {r["method"]: r for r in previous.get("methods", [])}
    for method in args.methods:
        result = validate_method(method, panel)
        results[method] = result
        print(json.dumps(result), flush=True)
    payload = dict(panel_definition_sha256=panel["definition_sha256"],
                   methods=[results[m] for m in sorted(results)],
                   scope="Artifact integrity and frozen output geometry; model source parity and official metric checks have separate receipts.")
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
