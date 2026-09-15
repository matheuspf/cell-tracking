"""Portable report from frozen detector results; no models or predictions change."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import warnings

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/cellpose-refine-v1"
METRIC_REV = "075fc5f5a52d11077f9dc2b074644618f26939e2"
LABELS = {"baseline": "Cellpose baseline", "20260914": "Head · seed 20260914",
          "314159": "Head · seed 314159", "constant": "Source-only constant offset",
          "incumbent": "Incumbent · exposure uncertain"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def group(data, embryo, encoding="integer"):
    return next(r for r in data[encoding] if r["embryo"] == embryo)


def bootstrap():
    """Paired, embryo-stratified clip resampling; not an independent embryo CI."""
    sources = {"baseline": REPO / "results/detector-development-20260914/cellpose_cpdino_vitb-integer-frames.json",
               "20260914": RESULTS / "seed-20260914-integer-frames.json",
               "314159": RESULTS / "seed-314159-integer-frames.json",
               "constant": REPO / "work/cellpose-refine-v1/offset-control/integer-detail.json"}
    rows = {k: read(p) for k, p in sources.items()}
    baseline = {r["key"]: r for r in rows["baseline"]}
    for method in rows.values():
        assert {r["key"] for r in method} == set(baseline)
        assert all(r["gt"] == baseline[r["key"]]["gt"] for r in method)
    clips = sorted({r["dataset"] for r in baseline.values()})
    clip_index = {c: i for i, c in enumerate(clips)}
    values = {}
    for key, method in rows.items():
        value = np.zeros((len(clips), 3), dtype=np.float64)
        for r in method:
            value[clip_index[r["dataset"]]] += [r["gt"], r["matches"]["3.0"], r["matches"]["7.0"]]
        values[key] = value
    rng = np.random.default_rng(14092026)
    repetitions = 10000
    weights = np.zeros((repetitions, len(clips)), dtype=np.int64)
    for embryo in ("44b6", "6bba"):
        ix = np.array([i for i, c in enumerate(clips) if c.startswith(embryo)])
        assert len(ix) == 20
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=repetitions)
    estimates = []
    for embryo in ("pooled", "44b6", "6bba"):
        ix = np.array([i for i, c in enumerate(clips) if embryo == "pooled" or c.startswith(embryo)])
        base = weights[:, ix] @ values["baseline"][ix]
        for method in ("20260914", "314159", "constant"):
            sample = weights[:, ix] @ values[method][ix]
            for j, radius in ((1, 3), (2, 7)):
                delta = 100 * (sample[:, j] - base[:, j]) / base[:, 0]
                estimates.append({"embryo": embryo, "method": method, "radius_um": radius,
                                  "delta_pp": float(100 * (values[method][ix, j].sum() - values["baseline"][ix, j].sum()) / values["baseline"][ix, 0].sum()),
                                  "interval_95_pp": np.quantile(delta, [.025, .975]).tolist()})
    return {"seed": 14092026, "repetitions": repetitions, "estimates": estimates,
            "input_sha256": {str(p.relative_to(REPO)): sha(p) for p in sources.values()},
            "scope": "Exploratory paired clip-resampling percentile intervals, stratified by embryo. These quantify sensitivity to the 40 observed clips. Crops can overlap and only two previously examined embryos exist; these are not biological-generalization confidence intervals. No multiplicity adjustment or model selection."}


def official_examples():
    metric_path = REPO / "work/annotation-selection-v1/official/src/tracking_cellmot/metrics.py"
    remote = json.loads(subprocess.check_output(["gh", "api", "repos/royerlab/kaggle-cell-tracking-competition/commits/main"], text=True))
    assert remote["sha"] == METRIC_REV, "Released scorer changed: review before reporting."
    metadata = json.loads(subprocess.check_output(["gh", "api", f"repos/royerlab/kaggle-cell-tracking-competition/contents/src/tracking_cellmot/metrics.py?ref={METRIC_REV}"], text=True))
    raw = metric_path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert blob == metadata["sha"], "Installed scorer differs from released source."
    sys.path.insert(0, str(metric_path.parents[1]))
    from tracking_cellmot import metrics
    examples = []
    # Synthetic count inputs isolate tradeoffs; these are NOT measured Cellpose graph scores.
    cases = [("Reference", 80, 0, 20, 1000), ("20% more submitted nodes; same links", 80, 0, 20, 1200),
             ("20% more nodes; four recovered correct edges", 84, 0, 16, 1200),
             ("20% more nodes; ten extra evaluable wrong edges", 80, 10, 20, 1200)]
    for name, tp, fp, fn, nodes in cases:
        row = metrics.per_sample_metrics(metrics.EvaluationResult(tp, fp, fn, 5, 0, 5, nodes), 1000, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.summarise([row])
        examples.append({"scenario": name, "nodes": nodes, "estimated_nodes": 1000,
                         "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
                         "edge_jaccard": row["edge_jaccard"], "multiplier": 1 - .1 * row["total_node_ratio"],
                         "adjusted_edge_jaccard": row["adj_edge_jaccard"], "division_jaccard": .5,
                         "combined_score": result["score"]})
    assert np.isclose(examples[0]["combined_score"], .85)
    assert np.isclose(examples[1]["combined_score"], .834)
    assert np.isclose(examples[2]["combined_score"], .8732)
    assert examples[3]["combined_score"] < examples[1]["combined_score"]
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "revision": METRIC_REV,
            "remote_commit_date": remote["commit"]["committer"]["date"], "source_sha256": sha(metric_path),
            "released_source_matches_local": True, "examples": examples,
            "source_url": f"https://github.com/royerlab/kaggle-cell-tracking-competition/blob/{METRIC_REV}/src/tracking_cellmot/metrics.py",
            "scope": "Illustrative inputs executed through official per_sample_metrics and summarise. Fixed division Jaccard 0.5; one sample. These are not model results or simulated image detections."}


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |", *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def make_data():
    m = read(RESULTS / "metrics.json")
    c = read(RESULTS / "offset-control.json")
    old = read(REPO / "results/detector-development-20260914/localization-headroom.json")
    assert m["data_fingerprint"] == c["data_fingerprint"] == old["data_fingerprint"]
    methods = {"baseline": m["baseline"], **m["seeds"], "constant": c["groups"], "incumbent": old["methods"]["incumbent"]}
    for method in methods.values():
        assert group(method, "pooled")["gt"] == 2383
        assert group(method, "pooled")["frames"] == 400
    counts = {group(methods[k], "pooled")["predicted"] for k in ("baseline", "20260914", "314159", "constant")}
    assert counts == {156226}
    return {"labels": LABELS, "methods": methods, "head_gates": m["promotion_checks"], "constant_gates": c["gates"],
            "capacity": read(RESULTS / "integer-capacity.json")["groups"], "bootstrap": bootstrap(),
            "official": official_examples(), "fingerprint": m["data_fingerprint"],
            "created_utc": datetime.now(timezone.utc).isoformat()}


def report_tables(data):
    methods = data["methods"]
    pooled = [[LABELS[k], f'{group(v,"pooled")["predicted"]:,}', *[f'{100*group(v,"pooled")["recall"][f"{r}.0"]:.2f}' for r in range(1, 8)]] for k, v in methods.items()]
    pooled_headers = ["Method", "Candidates", *[f"R{r} (%)" for r in range(1, 8)]]
    embryo_rows = []
    for embryo in ("44b6", "6bba"):
        for k, method in methods.items():
            g = group(method, embryo); f = group(method, embryo, "float")
            pairs = g["close_pairs"]["14"]
            embryo_rows.append([embryo, LABELS[k], f'{g["matches"]["3.0"]}/{g["gt"]} ({100*g["recall"]["3.0"]:.2f}%)',
                                f'{g["matches"]["7.0"]}/{g["gt"]} ({100*g["recall"]["7.0"]:.2f}%)',
                                f'{f["mean_error_censored_at_7_um"]:.3f}', f'{pairs["both_matched"]["3.0"]}/{pairs["pairs"]}',
                                f'{g["edge_endpoints"]["available"]}/{g["edge_endpoints"]["eligible"]}'])
    embryo_headers = ["Target embryo", "Method", "R3 matches / GT", "R7 matches / GT", "Float censored mean (µm) ↓", "Close pairs, both at R3", "GT edges, endpoints available"]
    gate_rows = []
    for embryo in ("44b6", "6bba"):
        b = group(methods["baseline"], embryo); bf = group(methods["baseline"], embryo, "float")
        for method in ("20260914", "314159", "constant"):
            g = group(methods[method], embryo); f = group(methods[method], embryo, "float")
            q3 = np.mean([g["budgets"][str(k)]["recall"]["3.0"] - b["budgets"][str(k)]["recall"]["3.0"] for k in (100, 200, 400)])
            deltas = [g["recall"]["7.0"] - b["recall"]["7.0"], *[g["budgets"][str(k)]["recall"]["7.0"] - b["budgets"][str(k)]["recall"]["7.0"] for k in (100, 200, 400)]]
            ed = f["mean_error_censored_at_7_um"] - bf["mean_error_censored_at_7_um"]
            passed = q3 > 0 and g["recall"]["3.0"] > b["recall"]["3.0"] and ed < 0 and min(deltas) >= -.002
            recorded = next(x for x in data["constant_gates"] if x["target_embryo"] == embryo)["passes_same_engineering_gate"] if method == "constant" else next(x for x in data["head_gates"] if x["target_embryo"] == embryo and str(x["seed"]) == method)["passes_declared_engineering_gate"]
            assert passed == recorded
            gate_rows.append([embryo, LABELS[method], f'{100*(g["recall"]["3.0"]-b["recall"]["3.0"]):+.3f}', f'{100*q3:+.3f}',
                              *[f'{100*x:+.3f}' for x in deltas], f'{ed:+.4f}', "Pass" if passed else "Fail"])
    gate_headers = ["Target", "Method", "Δ full R3 (pp)", "Δ Q3 (pp)", "Δ full R7 (pp)", "Δ R7 K100", "Δ R7 K200", "Δ R7 K400", "Δ float error (µm)", "Guard"]
    budget_rows = []
    for k in ("100", "200", "400", "all"):
        b = group(methods["baseline"], "pooled"); item = b if k == "all" else b["budgets"][k]
        budget_rows.append([k, f'{item["predicted"]:,}', *[f'{100*(group(methods[m],"pooled") if k=="all" else group(methods[m],"pooled")["budgets"][k])["recall"]["7.0"]:.2f}' for m in ("baseline", "20260914", "314159", "constant")]])
    budget_headers = ["Per-frame cap", "Candidates", "Cellpose R7 (%)", "Head 20260914 R7", "Head 314159 R7", "Constant R7"]
    ci_rows = [[r["embryo"], LABELS[r["method"]], f'R{r["radius_um"]}', f'{r["delta_pp"]:+.2f}', f'[{r["interval_95_pp"][0]:+.2f}, {r["interval_95_pp"][1]:+.2f}]'] for r in data["bootstrap"]["estimates"]]
    example_rows = [[r["scenario"], r["nodes"], f'{r["edge_jaccard"]:.4f}', f'{r["multiplier"]:.2f}', f'{r["combined_score"]:.4f}'] for r in data["official"]["examples"]]
    return {"pooled": (pooled_headers, pooled), "embryos": (embryo_headers, embryo_rows), "gates": (gate_headers, gate_rows),
            "budgets": (budget_headers, budget_rows), "intervals": (["Target", "Method", "Metric", "Δ recall (pp)", "95% clip-resampling interval"], ci_rows),
            "examples": (["Illustrative count input, not model results", "Submitted nodes", "Edge Jaccard", "Multiplier", "Combined score"], example_rows)}


def main():
    data = make_data()
    tables = report_tables(data)
    write(RESULTS / "paired-clip-bootstrap.json", data["bootstrap"])
    write(RESULTS / "official-node-count-examples.json", data["official"])
    write(RESULTS / "report-data.json", data)
    template = Path(__file__).with_suffix(".html").read_text()
    for key, (headers, rows) in tables.items():
        template = template.replace(f"@@{key.upper()}@@", table(headers, rows))
    template = template.replace("@@DATA@@", json.dumps(data, allow_nan=False).replace("</", "<\\/"))
    assert "@@" not in template
    (RESULTS / "report.html").write_text(template)
    markdown = Path(__file__).with_suffix(".md").read_text()
    for key, (headers, rows) in tables.items():
        markdown = markdown.replace(f"@@{key.upper()}@@", md_table(headers, rows))
    assert "@@" not in markdown
    (REPO / "docs/cellpose-refinement-results-20260914.md").write_text(markdown)
    print(json.dumps({"report": str(RESULTS / "report.html"), "official_source_verified": data["official"]["revision"],
                      "bootstrap_estimates": len(data["bootstrap"]["estimates"]), "methods": len(data["methods"])}))


if __name__ == "__main__":
    main()
