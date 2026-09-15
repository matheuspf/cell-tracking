"""Publish measured complete-clip results as a small interactive local canvas."""
from __future__ import annotations

import json
from pathlib import Path

from tools.annotation_selection.common import clean
from tools.annotation_selection.metric_adapter import aggregate
from tools.cellpose_ultrack.track import REPO, ROOT

CANVAS = Path("/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/cellpose-ultrack-20260914.canvas.tsx")

TEMPLATE = r'''import { BarChart, Button, Divider, Grid, H1, H2, Link, Row, Stack, Table, Text, useHostTheme, useState } from "cursor/canvas";

type Clip = { dataset: string; embryo: string; score: number; edge: number; adjusted: number; division: number | null; tp: number; fp: number; fn: number; dtp: number; dfp: number; dfn: number; gt: number; matched: number; rawMatched: number; predicted: number; rawPredicted: number; estimate: number; available: number; compute: number; tracking: number; solverGap: number; };
type Summary = { score: number; edge_jaccard: number; adj_edge_jaccard: number; division_jaccard: number | null; counts: { edge_tp: number; edge_fp: number; edge_fn: number; division_tp: number; division_fp: number; division_fn: number; num_pred_nodes: number } };
type Evidence = { clips: Clip[]; summaries: Record<string, Summary>; expected: number; complete: boolean; date: string; stability: { dataset: string; canonical_score: number; repeat_score: number; score_delta: number } | null; };
const data: Evidence = __DATA__;
const fixed = (v: number | null) => v === null ? "Undefined" : v.toFixed(5);
const pct = (v: number) => (100 * v).toFixed(2) + "%";
const n = (v: number) => v.toLocaleString("en-US");

export default function CellposeUltrackResults() {
  const theme = useHostTheme();
  const [group, setGroup] = useState("pooled");
  const rows = data.clips.filter(c => group === "pooled" || c.embryo === group);
  const summary = data.summaries[group];
  const sum = (key: keyof Clip) => rows.reduce((a, c) => a + Number(c[key]), 0);
  const source = "Source: frozen Cellpose cpDINO-ViT-B + ultrack/CBC; official scorer 075fc5f; complete frames 0–99; 14 September 2026.";
  return <Stack gap={22} style={{ padding: 24, maxWidth: 1220, margin: "0 auto", color: theme.text.primary }}>
    <Stack gap={7}>
      <Text size="small" tone="secondary">BIOHUB CELL TRACKING · COMPLETE GRAPH EVALUATION</Text>
      <H1>Cellpose + ultrack: measured tracking score</H1>
      <Text>{data.complete ? "Completed" : "Interim"}: {data.clips.length} of {data.expected} full clips, {data.clips.length * 100} volumes. Frozen Cellpose masks, standard overlap linking and CBC joint selection in 20-frame windows with overlap 5.</Text>
      <Text size="small" tone="secondary">Local exploratory evaluation on two reused public embryos. No fitted parameters or hidden-test score. No matched full-clip FOCUS3D comparison. Inherited pretraining exposure remains unresolved.</Text>
    </Stack>
    <Row gap={8} wrap>{Object.keys(data.summaries).map(g => <Button key={g} variant={g === group ? "primary" : "secondary"} onClick={() => setGroup(g)}>{g === "pooled" ? "All scored clips" : "Embryo " + g}</Button>)}</Row>
    <Grid columns="repeat(auto-fit, minmax(210px, 1fr))" gap={22}>
      <Stack gap={5}><Text size="small" tone="secondary">FULL COMPETITION METRIC</Text><H2 style={{ color: theme.accent.primary }}>{fixed(summary.score)}</H2><Text>Adjusted edge + 0.1 × division Jaccard</Text></Stack>
      <Stack gap={5}><Text size="small" tone="secondary">ADJUSTED EDGE JACCARD</Text><H2>{fixed(summary.adj_edge_jaccard)}</H2><Text>Unadjusted: {fixed(summary.edge_jaccard)}</Text></Stack>
      <Stack gap={5}><Text size="small" tone="secondary">DIVISION JACCARD</Text><H2>{fixed(summary.division_jaccard)}</H2><Text>TP / FP / FN: {summary.counts.division_tp} / {summary.counts.division_fp} / {summary.counts.division_fn}</Text></Stack>
    </Grid>
    <Divider />
    <Stack gap={8}>
      <H2>Complete tracking metric by clip</H2>
      <Text size="small" tone="secondary">X: dataset · Y: competition score (dimensionless; values can exceed 1)</Text>
      <BarChart categories={rows.map(c => c.dataset)} yMin={0} yMax={1.1} height={245} showValues
        series={[{ name: "Full tracking score", tone: "info", data: rows.map(c => c.score) }]} />
      <Text size="small" tone="secondary">{source} Pooled results use official count-weighted aggregation, not the mean of these bars.</Text>
    </Stack>
    <Stack gap={8}>
      <H2>Where the score is lost</H2>
      <Grid columns="repeat(auto-fit, minmax(300px, 1fr))" gap={24}>
        <Stack gap={7}>
          <Text>Raw Cellpose annotated-node recall: <strong>{pct(sum("rawMatched") / sum("gt"))}</strong>. After ultrack selection: <strong>{pct(sum("matched") / sum("gt"))}</strong>.</Text>
          <Text>{n(summary.counts.edge_tp)} correctly linked edges; {n(summary.counts.edge_fn)} missing GT edges; {n(summary.counts.edge_fp)} evaluable false-positive edges.</Text>
          <Text>{n(sum("available"))} GT edges have both endpoints available after selection. Ultrack links {pct(summary.counts.edge_tp / sum("available"))} of those correctly.</Text>
        </Stack>
        <Stack gap={7}>
          <Text>{n(sum("predicted"))} selected nodes versus a GEFF total-node estimate of {n(sum("estimate"))}; {n(sum("rawPredicted"))} raw Cellpose instances.</Text>
          <Text size="small" tone="secondary">Each clip is adjusted separately by max(0, edge Jaccard × [1.1 − 0.1 × predicted / estimated nodes]). Unmatched cells are not automatically known false positives.</Text>
          <Text size="small" tone="secondary">Division evidence is sparse. The full six-clip panel has only five annotated division events.</Text>
        </Stack>
      </Grid>
    </Stack>
    <Stack gap={8}>
      <H2>Per-clip counts and score components</H2>
      <Table headers={["Dataset", "Full score", "Edge J", "Adjusted edge J", "Edge TP / FP / FN", "Division TP / FP / FN", "Selected nodes"]}
        rows={rows.map(c => [c.dataset, fixed(c.score), fixed(c.edge), fixed(c.adjusted), `${c.tp} / ${c.fp} / ${c.fn}`, `${c.dtp} / ${c.dfp} / ${c.dfn}`, n(c.predicted)])}
        columnAlign={["left", "right", "right", "right", "right", "right", "right"]} striped />
      <Text size="small" tone="secondary">{source} Coordinates are rounded native ZYX mask centroids; node assignment uses the physical 7 µm gate.</Text>
    </Stack>
    <Divider />
    <Stack gap={8}>
      <H2>Runtime and verification</H2>
      <Text>Cellpose processing: {(sum("compute") / 60).toFixed(1)} minutes for {rows.length * 100} volumes, {(sum("compute") / (rows.length * 100)).toFixed(2)} seconds per volume. Tracking stages: {(sum("tracking") / 60).toFixed(1)} minutes. GPU queue time is excluded.</Text>
      <Text size="small" tone="secondary">A simple 19,900-volume extrapolation gives {(sum("compute") / (rows.length * 100) * 19900 / 3600).toFixed(1)} hours of local Cellpose processing. This is not measured full-dataset or Kaggle runtime.</Text>
      <Text size="small" tone="secondary">Checks cover complete movies, mask/link/export coordinate agreement, actual parent IDs, no merges, at most two daughters, SQL mask exclusions, CSV round trips, fresh matching gates, and independent official aggregation. Largest recorded CBC relative gap: {Math.max(...rows.map(c => c.solverGap)).toExponential(2)}.</Text>
      {data.stability && <Text size="small" tone="secondary">One independent hierarchy/link/solver repeat on {data.stability.dataset} scored {fixed(data.stability.repeat_score)} versus the original {fixed(data.stability.canonical_score)} (change {data.stability.score_delta.toFixed(5)}). Original graphs remain in the reported aggregate. This is a stability check, not a confidence interval.</Text>}
      <Row gap={16} wrap><Link href="/home/mpf/code/kaggle/cell-tracking/tools/cellpose_ultrack/README.md">Reproduction and method</Link>{data.complete && <Link href="/home/mpf/code/kaggle/cell-tracking/results/cellpose-ultrack-20260914/summary.json">Complete evidence JSON</Link>}<Link href="https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md">Official metric</Link></Row>
    </Stack>
  </Stack>;
}
'''


def main():
    panel = json.loads((ROOT / "panel.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "evaluation").glob("*.json")) if p.name != "summary.json"]
    if not rows:
        raise ValueError("No measured complete-clip scores to render")
    data = dict(expected=len(panel["clips"]), complete=len(rows) == len(panel["clips"]), date="2026-09-14", clips=[], summaries={}, stability=None)
    if (ROOT / "stability.json").exists():
        stability = json.loads((ROOT / "stability.json").read_text())
        data["stability"] = {key: stability[key] for key in
                             ("dataset", "canonical_score", "repeat_score", "score_delta")}
    for group in ["pooled", *sorted({r["embryo"] for r in rows})]:
        subset = [r for r in rows if group == "pooled" or r["embryo"] == group]
        summary = aggregate(subset, [r["dataset"] for r in subset])
        data["summaries"][group] = {key: summary[key] for key in
                                    ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard", "counts")}
    for r in rows:
        name = r["dataset"]
        tracking = json.loads((ROOT / "tracking" / name / "result.json").read_text())
        frames = [json.loads(p.read_text()) for p in (ROOT / "predictions/cellpose_cpdino_vitb").glob(f"{name}-*.json")]
        s = r["single_clip_summary"]
        data["clips"].append(dict(dataset=name, embryo=r["embryo"], score=s["score"], edge=r["edge_jaccard"],
                                  adjusted=r["adj_edge_jaccard"], division=s["division_jaccard"],
                                  tp=r["edge_tp"], fp=r["edge_fp"], fn=r["edge_fn"],
                                  dtp=r["division_tp"], dfp=r["division_fp"], dfn=r["division_fn"],
                                  gt=r["gt_nodes"], matched=r["matched_nodes"], rawMatched=r["raw_cellpose"]["matched_nodes"],
                                  rawPredicted=r["raw_cellpose"]["nodes"], predicted=r["num_pred_nodes"], estimate=r["estimated_total"],
                                  available=r["gt_edges_with_matched_endpoints"],
                                  compute=sum(f["timing_seconds"]["total"] for f in frames),
                                  tracking=sum(tracking["stage_seconds"].values()),
                                  solverGap=max(solver["gap"] for solver in tracking["solver"])))
    # The managed canvas directory already exists; write directly to its exact path.
    CANVAS.write_text(TEMPLATE.replace("__DATA__", json.dumps(clean(data), separators=(",", ":"))))
    print(CANVAS)
    if data["complete"] and (ROOT / "validation.json").exists():
        write_markdown_report()


def write_markdown_report():
    result = json.loads((ROOT / "evaluation/summary.json").read_text())
    validation = json.loads((ROOT / "validation.json").read_text())
    summary = result["summaries"]["pooled"]
    rows = result["rows"]
    total_gt = sum(r["gt_nodes"] for r in rows)
    selected_matches = sum(r["matched_nodes"] for r in rows)
    raw_matches = sum(r["raw_cellpose"]["matched_nodes"] for r in rows)
    available_edges = sum(r["gt_edges_with_matched_endpoints"] for r in rows)
    counts = summary["counts"]
    statuses = [s for clip in validation["solver_statuses"] for s in clip["solves"]]
    status_counts = {s: sum(x["status"] == s for x in statuses) for s in sorted({x["status"] for x in statuses})}
    lines = [
        f"**Cellpose cpDINO-ViT-B + ultrack scored {summary['score']:.5f} on six complete 100-frame clips.**",
        "This uses the full official competition metric, including temporal edges, divisions and the GEFF total-node adjustment.",
        "The panel contains three clips per embryo and was inherited from the existing detector pilot. It is an exploratory local result on reused public embryos; hidden-test performance is unmeasured.",
        "",
        f"- Full score: **{summary['score']:.8f}**.",
        f"- Adjusted edge Jaccard: **{summary['adj_edge_jaccard']:.8f}**.",
        f"- Unadjusted edge Jaccard: **{summary['edge_jaccard']:.8f}**; TP / FP / FN = {counts['edge_tp']:,} / {counts['edge_fp']:,} / {counts['edge_fn']:,}.",
        f"- Division Jaccard: **{summary['division_jaccard']:.8f}**; TP / FP / FN = {counts['division_tp']} / {counts['division_fp']} / {counts['division_fn']}.",
        "",
        "Scores by embryo use the organizer's count-weighted aggregation:",
        "",
    ]
    for embryo in ("44b6", "6bba"):
        s = result["summaries"][embryo]
        lines.append(f"- **{embryo}: {s['score']:.8f}**; edge Jaccard {s['edge_jaccard']:.8f}; division TP / FP / FN {s['division_tp']} / {s['division_fp']} / {s['division_fn']}.")
    lines.extend(["", "Complete-clip scores:", ""])
    for row in rows:
        lines.append(f"- `{row['dataset']}`: **{row['single_clip_summary']['score']:.8f}**; {row['num_pred_nodes']:,} selected nodes; division TP / FP / FN {row['division_tp']} / {row['division_fp']} / {row['division_fn']}.")
    lines.extend([
        "",
        f"Raw Cellpose recovered **{raw_matches:,}/{total_gt:,} annotated nodes ({100 * raw_matches / total_gt:.2f}%)**. After ultrack hypothesis selection, this was **{selected_matches:,}/{total_gt:,} ({100 * selected_matches / total_gt:.2f}%)**. Both rates are pooled by annotated-node count, distinct from the official summary's unweighted mean per-clip recall.",
        f"The selected graphs make both endpoints available for **{available_edges:,} GT edges**; **{counts['edge_tp']:,} ({100 * counts['edge_tp'] / available_edges:.2f}%)** of those are linked correctly. There are **{counts['edge_fp']:,} evaluable false-positive edges**. Sparse unmatched cells and forks are not automatically known false positives.",
        "Only five annotated divisions are present in this panel. Its division result is too small a sample to establish broad division performance.",
        "",
        "The checkpoint is `cpdino-vitb`, the strongest new pretrained detector in the repo's earlier screen, SHA256 `3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f`. Its original eager BF16, batch-8, native-geometry 3D inference recipe was retained. There was no fine-tuning. Inherited pretraining exposure remains unresolved, as documented in the prior detector review.",
        "Ultrack uses Cellpose labels to construct foreground/contours and hierarchical mask hypotheses, stock IoU link weights with a 15 micrometer physical gate, and CBC joint selection. Final runs all use 20-frame windows with overlap 5. Selected observation `id`/`parent_id` yields complete 100-frame graphs; the score is not an average of window scores. Masks, SQL coordinates and serialized linker centroids pass an agreement audit, and integer CSV export passes round-trip checks.",
        "Two execution issues were resolved explicitly. A six-voxel disconnected component crashed hierarchy construction, so its existing minimum-component filter was set to exclude regions below eight voxels. Next, global CBC solved the first clip at 0.9488359062 but found no feasible solution on the dense second clip within 180 seconds. The final windowed recipe was applied uniformly, including rerunning the first clip. Those initial results/failures are retained separately; no mask or association weights were tuned against labels.",
        f"All **{len(statuses)} window solves** completed. Recorded statuses: {status_counts}; largest relative gap **{max(s['gap'] for s in statuses):.3g}**. Per-window solver optimality does not establish optimality for a global 100-frame objective.",
        "",
        f"Cellpose used **{validation['cellpose_processing_seconds'] / 60:.1f} processing minutes**, averaging **{validation['cellpose_mean_processing_seconds']:.2f} seconds per volume**, plus **{validation['cellpose_gpu_queue_seconds'] / 60:.1f} minutes waiting for the shared GPU**. Tracking stages totalled **{validation['tracking_stage_seconds'] / 60:.1f} minutes** and overlapped later segmentation. These processing sums exclude failed diagnostic solves and are not end-to-end wall time.",
        f"A simple 19,900-volume extrapolation gives **{validation['cellpose_projected_19900_volume_processing_hours']:.1f} hours** of local Cellpose processing before tracking. Full-199 and Kaggle notebook runtime were not measured.",
        "",
        f"Validation passed for all six complete graphs, the official edge/division accounting, CSV integrity, and exact agreement with **{validation['benchmark_parity_frames']} previously benchmarked Cellpose masks and center arrays**. Eight graph/metric tests and the additional real-ultrack tiny-component regression passed in their isolated runtimes.",
        "No matched complete-clip FOCUS3D run was performed, so this study does not establish whether Cellpose + ultrack beats FOCUS3D + ultrack on the same cohort.",
        "",
        "[Metric evidence](../results/cellpose-ultrack-20260914/summary.json), [validation and runtime receipts](../results/cellpose-ultrack-20260914/validation.json), [per-window solver receipts](../results/cellpose-ultrack-20260914/tracking-receipts.json), [final configuration](../configs/cellpose-ultrack-windowed-v1.json), and [reproduction commands](../tools/cellpose_ultrack/README.md).",
        "The unmodified [organizer metric](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md) is pinned to `075fc5f5a52d11077f9dc2b074644618f26939e2`, verified against upstream main on 14 September 2026.",
    ])
    stability_path = ROOT / "stability.json"
    if stability_path.exists():
        stability = json.loads(stability_path.read_text())
        lines.append(f"An independent reconstruction of the dense clip `{stability['dataset']}` from identical masks and settings scored **{stability['repeat_score']:.8f}**, versus the original **{stability['canonical_score']:.8f}**: a change of **{stability['score_delta']:.8f}**. The original graph remains in the aggregate. This exposes small run-to-run tracking variation; one repeat does not give a confidence interval. [Stability receipt](../results/cellpose-ultrack-20260914/stability.json).")
    divisions_path = REPO / "results/cellpose-ultrack-20260914/division-diagnostics.json"
    if divisions_path.exists():
        diagnostics = json.loads(divisions_path.read_text())
        events = [event for clip in diagnostics for event in clip["events"]]
        available = sum(e["selected_ultrack"]["all_three_sides_matched"] for e in events)
        missed_available = sum(e["selected_ultrack"]["all_three_sides_matched"] and not e["recovered"] for e in events)
        lines.append(f"Official local division-window matching finds parent-side and both daughter-side evidence after selection for **{available}/{len(events)} annotated divisions**. Of those available events, **{missed_available} are still not recovered** as valid divisions. This diagnostic separates local node availability from division topology; it does not replace the official division score. [Division diagnostics](../results/cellpose-ultrack-20260914/division-diagnostics.json).")
    path = REPO / "docs/cellpose-ultrack-20260914.md"
    path.write_text("\n\n".join(line for line in lines if line) + "\n")
    print(path)


if __name__ == "__main__":
    main()
