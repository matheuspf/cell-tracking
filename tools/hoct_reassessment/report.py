"""Build the compact results document and interactive comparison from saved scores."""
from __future__ import annotations

import json
from pathlib import Path

from .common import CONFIG, ROOT, REPO, RESULTS, read, sha, write

CANVAS = Path("/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/hoct-reassessment-20260914.canvas.tsx")
LABELS = {
    "v3": "v3 reference (training exposed)",
    "previous_H_general_J": "Previous HOCT + baseline structure",
    "legacy_physical": "HOCT physical-unit control + baseline",
    "legacy_voxel": "HOCT voxel-unit correction + baseline",
    "ultrack_no_division": "Cellpose + ultrack, no divisions",
    "cellpose_general_v1_cuda": "Cellpose + HOCT general_v1",
    "cellpose_general_v1_no_division_cuda": "Cellpose + HOCT general_v1, no divisions",
    "cellpose_ctc_v0_cuda": "Cellpose + HOCT ctc_v0",
    "cellpose_ctc_v0_no_division_cuda": "Cellpose + HOCT ctc_v0, no divisions",
    "cellpose_source_residual_cuda": "Cellpose + HOCT, source-only adaptation",
    "cellpose_source_residual_no_division_cuda": "Cellpose + HOCT, source-only adaptation, no divisions",
}


def collect():
    summaries, rows = {}, {}
    for path in sorted(RESULTS.glob("summary-*.json")):
        result = read(path)
        assert result["config_sha256"] == sha(CONFIG)
        for key, value in result["summaries"].items():
            if key in summaries:
                assert summaries[key] == value
            summaries[key] = value
        for row in result["rows"]:
            key = row["arm"], row["dataset"]
            if key in rows:
                assert rows[key]["inputs"] == row["inputs"]
            rows[key] = row
    if set(LABELS) - set(summaries):
        raise ValueError("Comparison is incomplete")
    return summaries, list(rows.values())


def run():
    summaries, rows = collect()
    independent = [key for key in LABELS if key.startswith("cellpose_")]
    best = max(independent, key=lambda key: summaries[key]["pooled"]["score"])
    s = summaries[best]["pooled"]
    ul = summaries["ultrack_no_division"]["pooled"]
    v3 = summaries["v3"]["pooled"]
    delta_units = summaries["legacy_voxel"]["pooled"]["score"] - summaries["legacy_physical"]["pooled"]["score"]
    frozen_key = "cellpose_general_v1_no_division_cuda"
    adapted_key = "cellpose_source_residual_no_division_cuda"
    adapted = summaries[adapted_key]["pooled"]
    transfer_deltas = {embryo: summaries[adapted_key][embryo]["score"] - summaries[frozen_key][embryo]["score"]
                       for embryo in ("44b6", "6bba")}
    edge_gap = ul["edge_jaccard"] - s["edge_jaccard"]
    count_adjustment_gap = (ul["adj_edge_jaccard"] - ul["edge_jaccard"]) - (s["adj_edge_jaccard"] - s["edge_jaccard"])
    diagnostics = read(RESULTS / "diagnostics.json")
    validation = read(RESULTS / "validation.json")
    inheritance = read(RESULTS / "legacy-inheritance.json")
    comparisons = [dict(key=key, label=LABELS[key], summaries=summaries[key],
                        uses_baseline=key in {"v3", "previous_H_general_J", "legacy_physical", "legacy_voxel"}) for key in LABELS]
    title = f"Best standalone HOCT result: {s['score']:.6f} on six complete clips"
    facts = [
        f"Best of the six standalone HOCT configurations: {LABELS[best]}, {s['score']:.8f}. The matched ultrack no-division control scores {ul['score']:.8f}; v3 scores {v3['score']:.8f}.",
        f"Changing only legacy model units changes the full pilot score by {delta_units:+.8f}. Keeping the old calibration and baseline-dependent decoder does not produce a gain on this panel.",
        "The standalone experiment uses actual unchanged Cellpose masks and HOCT's parent-versus-orphan probabilities. It uses neither baseline links nor native detector features. Two frozen checkpoints are followed by one fixed source-only residual-head recipe, trained on the other embryo.",
        "The old adapted HOCT graph (H_probe_J) shares 98.66% of predicted edges with v3; its decoder explicitly protects 1,033,995 baseline edges across 199 clips. Its old score is not an independent tracking baseline.",
        "Only six reused public clips are measured here. The incumbent has known training overlap; exact HOCT and Cellpose pretraining lists remain unresolved. No result establishes performance on unseen embryos."
    ]
    adaptation_finding = (f"Source-only adaptation reaches {adapted['score']:.8f} with divisions disabled, "
                          f"below the frozen checkpoint in both transfer directions: {transfer_deltas['44b6']:+.8f} "
                          f"on target 44b6 and {transfer_deltas['6bba']:+.8f} on target 6bba. Lower source training "
                          "loss did not transfer to a higher competition score. This recipe is not promoted.")
    counts = diagnostics[best]["counts"]
    error_finding = (f"Of {s['counts']['edge_fn']} missed true edges, {counts['candidate_present_not_selected']} "
                     f"had the correct candidate available but unselected, {counts['missing_matched_endpoint']} "
                     f"lacked a matched detection endpoint, and {counts['outside_candidate_bank']} were outside "
                     f"the candidate bank. Of {s['counts']['edge_fp']} evaluable false edges, "
                     f"{counts['touches_unmatched_endpoint']} touch an unmatched prediction. Sparse annotation "
                     "does not establish that every such prediction is a false cell.")
    gap_finding = (f"The {ul['score']-s['score']:.8f} score gap to ultrack consists of {edge_gap:.8f} "
                   f"in pooled edge Jaccard and {count_adjustment_gap:.8f} in the aggregation's node-count "
                   "adjustment; both no-division arms have zero division reward. This is an arithmetic "
                   "decomposition of the scorer, not a causal detector-versus-tracker ablation. HOCT keeps "
                   f"{s['counts']['num_pred_nodes']:,} observations versus ultrack's {ul['counts']['num_pred_nodes']:,}.")
    result = dict(title=title, facts=facts, best_arm=best, best_score=s["score"], delta_ultrack=s["score"]-ul["score"],
                  delta_v3=s["score"]-v3["score"], methods=comparisons, rows=rows, diagnostics=diagnostics,
                  adaptation_finding=adaptation_finding, error_finding=error_finding, gap_finding=gap_finding,
                  transfer_deltas=transfer_deltas, ultrack_gap_edge=edge_gap, ultrack_gap_node_adjustment=count_adjustment_gap,
                  unit_delta=delta_units, validation=validation, inheritance=inheritance,
                  metric_revision=read(CONFIG)["metric_revision"], config_sha256=sha(CONFIG))
    write(RESULTS / "comparison.json", result)
    lines = ["# HOCT reassessment — 14 September 2026", "", "**"+title+".**", "", *[f+"\n" for f in facts],
             "## Full competition metric", "", "Each value uses all nodes and edges in the same six 100-frame clips. These are not 199-clip or leaderboard scores.", ""]
    for key in LABELS:
        row = summaries[key]["pooled"]
        c = row["counts"]
        lines.append(f"- **{LABELS[key]}:** {row['score']:.8f}; edge TP/FP/FN {c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']}; division TP/FP/FN {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {c['num_pred_nodes']:,} nodes.")
    lines += ["", "## Source-only adaptation", "", "The frozen general_v1 backbone supplies 288-dimensional edge embeddings. Mean features across the same overlapping windows feed one strongly regularized convex residual on log parent probabilities; the orphan alternative stays in the normalization. The recipe uses L2=0.1, source-only standardization and at most 300 L-BFGS iterations. No baseline features or edge membership are used.",
              "", "Each direction fits only its own three source clips: 1,072 supported incoming-parent targets for 44b6 and 2,785 for 6bba. Competing incoming parents are negatives; unlabeled targets, absent candidate parents and unrecorded daughters remain unknown. Both models are frozen before adapted target graphs are produced. Each fit has a file-access guard rejecting the opposite embryo. This is one exploratory recipe after the frozen-checkpoint screen, not untouched validation.",
              "", adaptation_finding,
              "", "## Remaining errors", "", error_finding, "", gap_finding,
              "", "With divisions enabled, general_v1 recovers two of five annotated divisions and produces 189 evaluable false divisions. The source-only residual recovers three but produces 333 false divisions. Parent-choice supervision alone has not supplied the evidence needed to distinguish mitosis or a true track birth from a competing continuation.",
              "", "The evidence points to observation selection and calibrated parent-versus-birth decisions as the next HOCT targets. Simply expanding the distance gate is unlikely to repair much here: only two missed true edges fall outside the candidate bank. This round does not rule out HOCT with different detections, supported birth supervision or a different observation-selection objective.",
              "", "## Validation and interpretation", "", "All 22 packaged HOCT source files match the pinned upstream Git blobs. Real-mask features agree with the upstream extractor on 273 regions, with maximum absolute difference 7.49e-6. Eight unit contracts and six actual upstream SCIP comparisons pass. CPU/CUDA FP32 logit agreement was checked on 16 real contextual batches across two models and two clips; this is tolerance agreement, not bit identity. Both checkpoints produce identical CPU/GPU edge graphs on one complete 100-frame clip, with and without divisions.",
              "", "The standalone decoder exactly solves the default HOCT adjacent-frame, fixed-observation objective. This is possible because node cost -10 makes every observation favorable. All raw Cellpose detections therefore remain in the graph; ultrack uses a different observation-selection stage. A shared GPU queue limits HOCT leases to ten elapsed seconds, checked between batches, while other experiments continue.",
              "", "The unit control changes model positions and equivalent diameter by 1/1.625 and inertia by 1/1.625². Its output coordinates, candidates, image/intensity features, windows, calibration and baseline prior stay fixed. This control does not test recalibration, independent detection or full upstream normalization.",
              "", "New graphs were frozen before fresh official evaluation. The scorer matches native integer centers within 7 µm and includes the node-count adjustment and division component. Both embryos are reported separately in the comparison artifact. Unknown annotations are not treated as biological negatives.",
              "", "## Artifacts", "", "- [Interactive comparison]("+str(CANVAS)+")", "- [Comparison and per-embryo counts](../results/hoct-reassessment-20260914/comparison.json)", "- [Error attribution](../results/hoct-reassessment-20260914/diagnostics.json)", "- [Reproduction](../tools/hoct_reassessment/README.md)", "- [Pinned official HOCT source](https://github.com/royerlab/hoct/tree/2ccc5040823bc944ab67790abd1f56eea7cd4f05)", "", "No baseline was replaced and no submission was made.", ""]
    (REPO / "docs/hoct-reassessment-20260914.md").write_text("\n".join(lines))
    canvas = (CANVAS_TEMPLATE
              .replace("__DATA__", json.dumps(dict(title=title, facts=facts, methods=comparisons, best=best,
                                                diagnostics=diagnostics, adaptation=adaptation_finding,
                                                errors=error_finding, gap=gap_finding), separators=(",", ":"))))
    CANVAS.write_text(canvas)
    print(title, "ultrack delta", result["delta_ultrack"], "v3 delta", result["delta_v3"], flush=True)


CANVAS_TEMPLATE = '''import { BarChart, Button, Divider, Grid, H1, H2, Link, Row, Stack, Table, Text, useHostTheme, useState } from "cursor/canvas";
type Counts = { edge_tp: number; edge_fp: number; edge_fn: number; division_tp: number; division_fp: number; division_fn: number; num_pred_nodes: number };
type Score = { score: number; counts: Counts; n: number; n_adj: number; edge_jaccard: number; division_jaccard: number; division_tp: number; division_fp: number; division_fn: number; node_recall: number; adj_edge_jaccard: number };
type Method = { key: string; label: string; summaries: Record<string, Score>; uses_baseline: boolean };
type Diagnostic = { counts: Record<string, number>; clips: {dataset: string; missing_edges: Record<string, number>; false_edges: Record<string, number>}[] };
type Evidence = { title: string; facts: string[]; methods: Method[]; best: string; diagnostics: Record<string, Diagnostic>; adaptation: string; errors: string; gap: string };
const data: Evidence = __DATA__;
const fmt = (n: number) => n.toFixed(6);
const count = (n: number) => n.toLocaleString("en-US");
const triple = (a: number, b: number, c: number) => [a,b,c].map(count).join(" / ");
export default function HoctReassessment() {
  const theme = useHostTheme();
  const [group, setGroup] = useState("pooled");
  const [showControls, setShowControls] = useState(false);
  const methods = data.methods.filter(m => showControls || !m.uses_baseline || m.key === "v3");
  const best = data.methods.find(m => m.key === data.best)!;
  const baseline = data.methods.find(m => m.key === "v3")!;
  const ultrack = data.methods.find(m => m.key === "ultrack_no_division")!;
  const current = best.summaries[group];
  const diagnostic = data.diagnostics[data.best];
  const parts = diagnostic.clips.filter(c => group === "pooled" || c.dataset.startsWith(group));
  const missing = ["missing_matched_endpoint", "outside_candidate_bank", "candidate_present_not_selected"];
  const missingLabels = ["A detector endpoint is not matched", "Correct link outside candidate bank", "Correct candidate not selected"];
  const totals = missing.map(k => parts.reduce((v,c) => v + (c.missing_edges[k] || 0), 0));
  return <Stack gap={24} style={{padding:24, maxWidth:1180, color:theme.text.primary}}>
    <Stack gap={9}>
      <H1>HOCT: corrected features and standalone tracking</H1>
      <Text>{data.title}</Text>
      <Text tone="secondary">Six complete 100-frame public clips, three per embryo. Full competition metric, 14 September 2026. These clips have been reused for development.</Text>
    </Stack>
    <Row gap={8} wrap>{[["pooled","All six clips"],["44b6","Embryo 44b6"],["6bba","Embryo 6bba"]].map(([key,label]) => <Button key={key} variant={group === key ? "primary" : "secondary"} onClick={() => setGroup(key)}>{label}</Button>)}</Row>
    <Grid columns="repeat(auto-fit,minmax(220px,1fr))" gap={24}>
      <Stack gap={6}><Text tone="secondary">Best pooled HOCT configuration, selected cohort</Text><H2>{fmt(current.score)}</H2><Text size="small">{best.label}</Text></Stack>
      <Stack gap={6}><Text tone="secondary">Difference from ultrack no-division control</Text><H2>{(current.score - ultrack.summaries[group].score).toFixed(6)}</H2><Text size="small">Same source Cellpose masks; observation selection differs.</Text></Stack>
      <Stack gap={6}><Text tone="secondary">v3 reference</Text><H2>{fmt(baseline.summaries[group].score)}</H2><Text size="small">Known overlap between training and assessment data.</Text></Stack>
    </Grid>
    <Divider />
    <Stack gap={8}>
      <H2>Full competition tracking score</H2>
      <Button variant="secondary" onClick={() => setShowControls(!showControls)}>{showControls ? "Hide legacy controls" : "Show legacy unit controls"}</Button>
      <Text size="small" tone="secondary">Horizontal axis: score (unitless). Vertical axis: tracking configuration. Higher is better; zero-based scale.</Text>
      <BarChart horizontal categories={methods.map(m=>m.label)} height={Math.max(280,methods.length*40)} yMin={0} yMax={1.1} showValues series={[{name:"Full competition score",tone:"info",data:methods.map(m=>m.summaries[group].score)}]} />
      <Text size="small" tone="secondary">Source: pinned official scorer 075fc5f, frames 0–99. Exact weighted aggregation, not the average of clip scores.</Text>
      <div style={{overflowX:"auto"}}><Table headers={["Method","Score","Edge TP / FP / FN","Division TP / FP / FN","Nodes"]} rows={methods.map(m=>{const s=m.summaries[group],c=s.counts;return [m.label,fmt(s.score),triple(c.edge_tp,c.edge_fp,c.edge_fn),triple(c.division_tp,c.division_fp,c.division_fn),count(c.num_pred_nodes)];})} columnAlign={["left","right","right","right","right"]} striped /></div>
    </Stack>
    <Grid columns="repeat(auto-fit,minmax(340px,1fr))" gap={28}>
      <Stack gap={9}>
        <H2>Where the best pooled HOCT configuration misses edges</H2>
        <Table headers={["Failure stage","Missed edges"]} rows={missingLabels.map((label,i)=>[label,count(totals[i])])} columnAlign={["left","right"]} />
        <Text size="small" tone="secondary">Official endpoint matching and the actual candidate bank determine these categories. This is post hoc diagnosis, not a fitted correction.</Text>
      </Stack>
      <Stack gap={9}>
        <H2>What changed</H2>
        <Text>Real Cellpose mask morphology is expressed in native voxels. HOCT compares candidate parents with its learned no-parent outcome, using upstream contextual windows and normalization.</Text>
        <Text>The standalone inference reads no baseline graph, native features or Biohub labels. Adapted heads use labels from the other embryo only; both fits are frozen before target graph generation. Separate no-division controls expose the cost of false branching.</Text>
        <Text>{data.adaptation}</Text>
      </Stack>
    </Grid>
    <Stack gap={10}>
      <H2>Why the gap remains across all six clips</H2>
      <Text>{data.errors}</Text>
      <Text>{data.gap}</Text>
    </Stack>
    <Divider />
    <Stack gap={10}>
      <H2>What the previous HOCT score measured</H2>
      <Text>{data.facts[3]}</Text>
      <Text>{data.facts[1]}</Text>
      <Text>{data.facts[4]}</Text>
    </Stack>
    <Stack gap={8}>
      <H2>Verification</H2>
      <Text>22 upstream source files match the pinned Git blobs. Actual-mask features match upstream on 273 regions. Eight unit contracts and six SCIP comparisons validate the implementation. CPU/CUDA agreement is checked within tolerance, with identical edge graphs on one complete clip. Every standalone output CSV passes a round-trip check.</Text>
      <Text tone="secondary" size="small">The fixed-observation decoder applies to adjacent-frame edges with the tested node cost. It is not a replacement for HOCT's general solver when observations can be omitted or edges span gaps.</Text>
      <Row gap={16} wrap>
        <Link href="/home/mpf/code/kaggle/cell-tracking/docs/hoct-reassessment-20260914.md">Results and scope</Link>
        <Link href="/home/mpf/code/kaggle/cell-tracking/results/hoct-reassessment-20260914/comparison.json">Exact counts</Link>
        <Link href="/home/mpf/code/kaggle/cell-tracking/tools/hoct_reassessment/README.md">Reproduction</Link>
        <Link href="https://github.com/royerlab/hoct">Official HOCT repository</Link>
      </Row>
    </Stack>
  </Stack>;
}
'''


if __name__ == "__main__":
    run()
