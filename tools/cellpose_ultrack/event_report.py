"""Render the matched baseline audit and completed event-cost controls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from tools.cellpose_ultrack.event_costs import ARMS, OUT, STUDY
from tools.cellpose_ultrack.track import REPO, ROOT

CANVAS = Path("/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/cellpose-ultrack-20260914.canvas.tsx")
LABELS = {"stock": "Stock ultrack", "division-surcharge": "Extra division cost",
          "reference-events": "Reference event costs", "no-division-control": "No divisions (diagnostic)",
          "v3": "Selected v3", "public_harmonic": "Public Harmonic Fusion",
          "pair-control": "Explicit pairs, no image", "pair-image": "Temporal image pairs",
          "persistent-control": "Persistence, no image", "persistent-image": "Image + persistence"}
CAUSES = {
    "missing_from_raw_cellpose_and_selected_graph": "No raw or selected endpoint match",
    "lost_during_ultrack_selection": "Endpoint lost during selection",
    "candidate_link_present_but_not_selected": "Correct candidate not selected",
    "candidate_link_lost_to_top_five_iou_pruning": "Correct link pruned by top-five IoU",
    "candidate_link_lost_to_ten_nearest_hypotheses": "Correct link outside ten nearest hypotheses",
}
TEMPLATE = r'''import { BarChart, Button, Divider, Grid, H1, H2, Link, Row, Stack, Table, Text, useHostTheme, useState } from "cursor/canvas";
type Counts = { edge_tp: number; edge_fp: number; edge_fn: number; division_tp: number; division_fp: number; division_fn: number; num_pred_nodes: number };
type Summary = { score: number; edge_jaccard: number; adj_edge_jaccard: number; division_jaccard: number; counts: Counts };
type Method = { key: string; label: string; summaries: Record<string, Summary>; forks: Record<string, number>; clips: { dataset: string; embryo: string; score: number }[] };
type LatestAudit = { paragraphs: string[]; missed: { label: string; count: number }[]; falseEdges: { label: string; count: number }[]; gap: { label: string; value: number; fraction: number }[]; clips: { dataset: string; fp: number; fn: number; divisionFp: number }[] };
type Evidence = { methods: Method[]; causes: { label: string; counts: Record<string, number> }[]; validation: string; adaptations: string[]; latestAudit: LatestAudit | null };
const data: Evidence = __DATA__;
const fmt = (v: number) => v.toFixed(5);
const n = (v: number) => v.toLocaleString("en-US");
const triplet = (a: number, b: number, c: number) => n(a) + " / " + n(b) + " / " + n(c);
export default function CellposeUltrackInvestigation() {
  const theme = useHostTheme();
  const [group, setGroup] = useState("pooled");
  const [showAll, setShowAll] = useState(false);
  const latest = data.methods.some(m => m.key === "persistent-image") ? ["persistent-control", "persistent-image"] : ["pair-control", "pair-image"];
  const displayed = data.methods.filter(m => showAll || ["stock", "no-division-control", ...latest, "v3"].includes(m.key));
  const stock = data.methods.find(m => m.key === "stock")!;
  const baseline = data.methods.find(m => m.key === "v3")!;
  const cp = stock.summaries[group], base = baseline.summaries[group];
  const gt = cp.counts.edge_tp + cp.counts.edge_fn;
  const fp = .5 * (gt - cp.counts.edge_fn + gt - base.counts.edge_fn) * (1 / (gt + base.counts.edge_fp) - 1 / (gt + cp.counts.edge_fp));
  const fn = (cp.counts.edge_fn - base.counts.edge_fn) * .5 * (1 / (gt + cp.counts.edge_fp) + 1 / (gt + base.counts.edge_fp));
  const count = (base.adj_edge_jaccard - base.edge_jaccard) - (cp.adj_edge_jaccard - cp.edge_jaccard);
  const division = .1 * (base.division_jaccard - cp.division_jaccard);
  const clips = stock.clips.filter(c => group === "pooled" || c.embryo === group);
  const caption = "Source: six complete pilot clips, frames 0–99; official scorer 075fc5f; development evaluation, 14 September 2026. Identical clips for every method.";
  return <Stack gap={24} style={{ padding: 24, maxWidth: 1250, margin: "0 auto", color: theme.text.primary }}>
    <Stack gap={8}>
      <Text size="small" tone="secondary">BIOHUB CELL TRACKING · ERROR ANALYSIS AND CONTROLLED ADAPTATION</Text>
      <H1>Cellpose + ultrack: divisions and the baseline gap</H1>
      <Text>Measured comparison of v3, stock Cellpose/ultrack, event-cost controls and temporal daughter-pair evidence. All Cellpose arms reuse the same 600 mask volumes, segmentation hypotheses and candidate links.</Text>
      <Text size="small" tone="secondary">Only five annotated divisions occur in this pilot. Sparse annotation does not estimate biological division prevalence. Controls were chosen after the stock error audit; this is development evidence.</Text>
    </Stack>
    <Row gap={8} wrap>{["pooled", "44b6", "6bba"].map(g => <Button key={g} variant={g === group ? "primary" : "secondary"} onClick={() => setGroup(g)}>{g === "pooled" ? "All six clips" : "Embryo " + g}</Button>)}</Row>
    <Grid columns="repeat(auto-fit, minmax(220px, 1fr))" gap={24}>
      <Stack gap={5}><Text size="small" tone="secondary">STOCK CELLPOSE + ULTRACK</Text><H2>{fmt(cp.score)}</H2><Text>Full competition score</Text></Stack>
      <Stack gap={5}><Text size="small" tone="secondary">V3, SAME CLIPS</Text><H2 style={{ color: theme.accent.primary }}>{fmt(base.score)}</H2><Text>Gap: {fmt(base.score - cp.score)}</Text></Stack>
      <Stack gap={5}><Text size="small" tone="secondary">RECOVERED / ANNOTATED DIVISIONS</Text><H2>{cp.counts.division_tp} / {cp.counts.division_tp + cp.counts.division_fn}</H2><Text>v3: {base.counts.division_tp} / {base.counts.division_tp + base.counts.division_fn}</Text></Stack>
    </Grid>
    {data.adaptations.length > 0 && <Stack gap={9}>
      <H2>What the division adaptations changed</H2>
      {data.adaptations.map((text, i) => <Text key={i}>{text}</Text>)}
    </Stack>}
    {data.latestAudit && <Stack gap={12}>
      <H2>Where the errors are now</H2>
      <Text size="small" tone="secondary">Latest image + persistence graph, all six complete clips pooled. Fresh official matching; this section always shows the whole pilot.</Text>
      {data.latestAudit.paragraphs.map((text, i) => <Text key={i}>{text}</Text>)}
      <Grid columns="repeat(auto-fit, minmax(340px, 1fr))" gap={24}>
        <Stack gap={8}><H2>391 missed temporal edges</H2>
          <Table headers={["Measured stage", "Missing edges"]} rows={data.latestAudit.missed.map(r => [r.label,n(r.count)])} columnAlign={["left","right"]} striped />
        </Stack>
        <Stack gap={8}><H2>376 false temporal edges</H2>
          <Table headers={["Endpoint matching", "False edges"]} rows={data.latestAudit.falseEdges.map(r => [r.label,n(r.count)])} columnAlign={["left","right"]} striped />
          <Text size="small" tone="secondary">334 false edges have a source with one child; 42 originate at a fork. An unmatched prediction is not automatically a biologically false cell.</Text>
        </Stack>
      </Grid>
      <Grid columns="repeat(auto-fit, minmax(340px, 1fr))" gap={24}>
        <Stack gap={8}><H2>Remaining score gap to v3</H2>
          <Table headers={["Metric component","Score gap","Share"]} rows={data.latestAudit.gap.map(r => [r.label,fmt(r.value),(100*r.fraction).toFixed(1)+"%"])} columnAlign={["left","right","right"]} />
          <Text size="small" tone="secondary">Exact descriptive score accounting; division-related edge errors also enter the edge terms. This is not a causal model ablation.</Text>
        </Stack>
        <Stack gap={8}><H2>Error counts by clip</H2>
          <Table headers={["Dataset","Edge FP","Edge FN","Division FP"]} rows={data.latestAudit.clips.map(r => [r.dataset,n(r.fp),n(r.fn),n(r.divisionFp)])} columnAlign={["left","right","right","right"]} striped />
          <Text size="small" tone="secondary">Counts depend on annotated coverage; clips contain different numbers of annotated cells and edges.</Text>
        </Stack>
      </Grid>
      <Link href="/home/mpf/code/kaggle/cell-tracking/docs/cellpose-ultrack-current-errors-20260914.md">Current error findings and interpretation</Link>
    </Stack>}
    <Divider />
    <Stack gap={8}>
      <H2>Full tracking score: latest comparison and earlier controls</H2>
      <Row gap={8}><Button variant={showAll ? "secondary" : "primary"} onClick={() => setShowAll(false)}>Latest comparison</Button><Button variant={showAll ? "primary" : "secondary"} onClick={() => setShowAll(true)}>Every completed arm</Button></Row>
      <Text size="small" tone="secondary">X: official competition score (dimensionless; may exceed 1) · Y: method</Text>
      <BarChart horizontal categories={displayed.map(m => m.label)} yMin={0} yMax={1.1} height={Math.max(260, displayed.length * 38)} showValues series={[{ name: "Full tracking score", tone: "info", data: displayed.map(m => m.summaries[group].score) }]} />
      <Text size="small" tone="secondary">{caption} Score = weighted adjusted edge Jaccard + 0.1 × micro division Jaccard.</Text>
      <div style={{ overflowX: "auto" }}><Table headers={["Method", "Full score", "Edge TP / FP / FN", "Division TP / FP / FN", "All forks", "Selected nodes"]}
        rows={displayed.map(m => { const s = m.summaries[group], c = s.counts; return [m.label, fmt(s.score), triplet(c.edge_tp,c.edge_fp,c.edge_fn), triplet(c.division_tp,c.division_fp,c.division_fn), n(m.forks[group]), n(c.num_pred_nodes)]; })}
        columnAlign={["left", "right", "right", "right", "right", "right"]} striped /></div>
      <Text size="small" tone="secondary">All forks include unannotated cells; they are not all known errors. Division FP is the official evaluable count. The no-division control cannot recover mitosis.</Text>
    </Stack>
    <Grid columns="repeat(auto-fit, minmax(340px, 1fr))" gap={28}>
      <Stack gap={9}>
        <H2>Why the original objective overbranches</H2>
        <Text>With fixed selected nodes, a second daughter competes with starting a new track. For an interior child, the objective difference is:</Text>
        <Text><strong>IoU(second link)⁴ + division weight − appearance weight</strong></Text>
        <Text>Stock gives both events −0.001, so any positive overlap favors the fork in this local comparison. Copying −0.1 for both leaves the same local preference.</Text>
        <Text>Extra division cost uses −0.011 versus −0.001 for a new track. The extra link must contribute more than 0.01. The no-division control uses −1.01, exceeding the maximum possible extra overlap reward.</Text>
        <Text size="small" tone="secondary">Verified with five actual CBC solves. Global selection, competing links and window boundaries also affect the final graph.</Text>
      </Stack>
      <Stack gap={9}>
        <H2>Accounting for the original score gap</H2>
        <Table headers={["Component", "Score contribution"]} rows={[["Excess false-positive edges", fmt(fp)], ["Missing true edges", fmt(fn)], ["Node-count adjustment effect", fmt(count)], ["Division bonus", fmt(division)], ["Total v3 − stock gap", fmt(base.score - cp.score)]]} columnAlign={["left", "right"]} />
        <Text size="small" tone="secondary">Exact descriptive decomposition. FP/FN contributions average both substitution orders because Jaccard is nonlinear. This is not a causal detector-versus-linker decomposition.</Text>
        <Text>False forks also create false temporal edges. Event-cost changes must be evaluated against the whole graph score.</Text>
      </Stack>
    </Grid>
    <Divider />
    <Grid columns="repeat(auto-fit, minmax(340px, 1fr))" gap={28}>
      <Stack gap={9}>
        <H2>Where stock misses annotated edges</H2>
        <Table headers={["Measured failure stage", "Missing GT edges"]} rows={data.causes.map(c => [c.label, n(c.counts[group] || 0)])} columnAlign={["left", "right"]} striped />
        <Text size="small" tone="secondary">Post hoc diagnosis of {n(cp.counts.edge_fn)} missing edges. None of the missing candidate links between selected matched endpoints exceeds the physical 15 µm gate.</Text>
      </Stack>
      <Stack gap={9}>
        <H2>Why detection recall is not enough</H2>
        <Text>Across all six clips, raw Cellpose matches 96.84% of annotated nodes; selected ultrack matches 96.11%; v3 matches 98.91%.</Text>
        <Text>533 of stock’s 540 evaluable false edges touch an unmatched endpoint. For 262, that endpoint is still within 7 µm of the expected GT cell. Competing hypotheses or duplicated detections can affect identity assignment.</Text>
        <Text size="small" tone="secondary">Matching diagnostics do not establish that every unmatched object is biologically false. Separating fragmentation, localization and genuine nearby cells requires visual review.</Text>
        <Text>v3 retains learned image-based associations and conservative division repair. Stock ultrack here uses mask overlap alone. The new pair model supplies division appearance evidence; continuation links still use overlap.</Text>
      </Stack>
    </Grid>
    <Stack gap={8}>
      <H2>Same-clip score comparison</H2>
      <div style={{ overflowX: "auto" }}><Table headers={["Dataset", ...displayed.map(m => m.label)]} rows={clips.map(c => [c.dataset, ...displayed.map(m => fmt(m.clips.find(x => x.dataset === c.dataset)!.score))])} striped /></div>
      <Text size="small" tone="secondary">{caption} Aggregates use official weights, not the arithmetic mean of these rows.</Text>
    </Stack>
    <Divider />
    <Stack gap={9}>
      <H2>What the forum changes</H2>
      <Text>The thread warns against identifying mitosis from displacement alone or using sparse labels as a biological prior. Participants suggest temporal appearance and daughter-pair evidence; those are hypotheses requiring validation.</Text>
      <Text>Its Figure 6 screenshot refers to the earlier sparse zebrafish experiment. The linked implementation uses a learned foreground/contour model on the dense channel. In the published paper this is Figure 4; Figure 6 is a different neuromast experiment.</Text>
      <Text>The repo’s full 199-clip v3 evaluation recovers only 29 of 151 annotated divisions, with 92 evaluable false divisions. Improving true division recovery remains necessary even for v3.</Text>
      <Text size="small" tone="secondary">The linked external Zarr metadata confirms spacing 1.625 × 0.40625 × 0.40625 µm. Matching spacing alone does not establish independent embryo provenance.</Text>
      <Row gap={16} wrap><Link href="https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573">Forum thread</Link><Link href="https://royerlab.github.io/ultrack/optimizing.html">Ultrack tuning guidance</Link><Link href="https://www.nature.com/articles/s41592-025-02778-0">Published paper</Link></Row>
    </Stack>
    <Stack gap={9}>
      <H2>Validation and limits</H2>
      <Text>{data.validation}</Text>
      <Text size="small" tone="secondary">Candidate database fingerprints, source graphs, geometry metadata and graph/CSV checks are verified after solving. No baseline replacement or hidden-test inference was performed.</Text>
      <Text size="small" tone="secondary">Both embryos have been reused. v3 includes components with documented upstream label exposure; a source-only residual fit does not remove inherited exposure. The measured gap cannot isolate architecture from training-data effects.</Text>
      <Row gap={16} wrap><Link href="/home/mpf/code/kaggle/cell-tracking/docs/cellpose-ultrack-error-analysis-20260914.md">Findings and adaptation</Link><Link href="/home/mpf/code/kaggle/cell-tracking/results/cellpose-ultrack-event-costs-20260914/comparison.json">All cost-control results</Link><Link href="/home/mpf/code/kaggle/cell-tracking/results/cellpose-ultrack-20260914/error-analysis.json">Matched baseline audit</Link><Link href="/home/mpf/code/kaggle/cell-tracking/tools/cellpose_ultrack/README.md">Reproduction</Link></Row>
    </Stack>
  </Stack>;
}
'''


def main():
    analysis = json.loads((REPO / "results" / ROOT.name / "error-analysis.json").read_text())
    comparison = json.loads((OUT / "comparison.json").read_text())
    learned_path = REPO / "results/cellpose-ultrack-learned-divisions-20260914/comparison.json"
    learned = json.loads(learned_path.read_text()) if learned_path.exists() else None
    persistent_path = REPO / "results/cellpose-ultrack-persistent-divisions-20260914/comparison.json"
    persistent = json.loads(persistent_path.read_text()) if persistent_path.exists() else None
    latest_path = REPO / "results/cellpose-ultrack-persistent-divisions-20260914/error-analysis.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else None
    learned_arms = ["pair-control", "pair-image"] if learned else []
    if learned:
        comparison["arms"].update(learned["arms"])
    persistent_arms = ["persistent-control", "persistent-image"] if persistent else []
    if persistent:
        comparison["arms"].update({a:persistent["arms"][b] for a,b in zip(persistent_arms, ("pair-control", "pair-image"), strict=True)})
    methods = []
    for arm in ["stock", *ARMS, *learned_arms, *persistent_arms, "v3", "public_harmonic"]:
        if arm in comparison["arms"]:
            value = comparison["arms"][arm]
            rows, summaries = value["rows"], value["summaries"]
            fork_key, summary_key = "predicted_divisions", "single_clip_summary"
        else:
            rows = [r for r in analysis["rows"] if r["arm"] == arm]
            summaries = analysis["summaries"][arm]
            fork_key, summary_key = "predicted_forks", "summary"
        display_summaries = {g:{k:s[k] for k in ("score","edge_jaccard","adj_edge_jaccard","division_jaccard","counts")} for g,s in summaries.items()}
        methods.append(dict(key=arm, label=LABELS[arm], summaries=display_summaries,
                            forks={g: sum(r[fork_key] for r in rows if g == "pooled" or r["embryo"] == g)
                                   for g in ("pooled", "44b6", "6bba")},
                            clips=[dict(dataset=r["dataset"], embryo=r["embryo"], score=r[summary_key]["score"]) for r in rows]))
    causes = [dict(label=label, counts={g: sum(d["fn_reasons"].get(key, 0) for d in analysis["diagnostics"]
                                              if g == "pooled" or d["dataset"].startswith(g))
                                      for g in ("pooled", "44b6", "6bba")}) for key, label in CAUSES.items()]
    statuses = Counter(s for v in comparison["validation"].values() for s in v["solver_statuses"])
    max_gap = max(v["maximum_solver_relative_gap"] for v in comparison["validation"].values())
    for study in (learned, persistent):
        for v in (study["validation"].values() if study else []):
            statuses.update(v["solver_statuses"])
            max_gap = max(max_gap,v["maximum_solver_relative_gap"])
    validation = (f"All {(len(ARMS) + len(learned_arms) + len(persistent_arms)) * 6} new complete graphs were scored afresh. Solver statuses: "
                  + ", ".join(f"{v} {k}" for k, v in sorted(statuses.items()))
                  + f". Largest relative gap: {max_gap:.3g}. Eighteen real-CBC event, pair and persistence checks passed.")
    adaptations = []
    if learned:
        adaptations.append("All-six-clip result: frozen temporal image evidence recovers one division with 29 FP and scores 0.81942. Its matched zero-image pair control recovers none with 20 FP and scores 0.82656. The extra division recovery costs more in temporal-edge accuracy than it gains in the division bonus.")
        adaptations.append("Nine of the 29 evaluable false divisions have a daughter with no immediate continuation; 20 already persist. The recovered true division has both daughters continue once, but one ends before a second continuation. A stricter two-frame persistence filter would remove that recovery in the frozen graph.")
        adaptations.append("The image model is the existing v4 C4 optical tower, directly fitted on the opposite embryo. Its inherited synthetic calibration and upstream label exposure remain. It is not a newly calibrated biological division probability.")
    if persistent:
        ps = persistent["arms"]["pair-image"]["summaries"]["pooled"]
        pc = persistent["arms"]["pair-control"]["summaries"]["pooled"]
        c = ps["counts"]
        adaptations.append(f"Follow-up with one-frame daughter persistence: image arm {ps['score']:.5f}, matched no-image control {pc['score']:.5f}; difference {ps['score']-pc['score']:+.5f}. Image-arm division TP/FP/FN: {c['division_tp']}/{c['division_fp']}/{c['division_fn']}. Both arms constrain the actual selected continuation edges, including processing-window seams. This follow-up was motivated by the observed pilot errors.")
        adaptations.append("The final optical variant improves on stock, but does not beat the simpler control or increase aggregate division recall: it still recovers one of five annotated events. Compared with its matched persistence control it loses 16 correct temporal edges and adds 21 false edges. Continuation identity and calibration at the new Cellpose proposals remain unresolved.")
    latest_view = current_audit_view(latest) if latest else None
    CANVAS.write_text(TEMPLATE.replace("__DATA__", json.dumps(dict(methods=methods, causes=causes, validation=validation, adaptations=adaptations, latestAudit=latest_view), separators=(",", ":"))))
    write_report(analysis, comparison, validation, learned, persistent)
    if latest:
        write_current_report(latest, latest_view)
    print(CANVAS)


def write_report(analysis, comparison, validation, learned=None, persistent=None):
    stock = comparison["arms"]["stock"]["summaries"]["pooled"]
    base = analysis["summaries"]["v3"]["pooled"]
    lines = [
        f"**On the same six complete clips, stock Cellpose + ultrack scores {stock['score']:.5f}, versus {base['score']:.5f} for selected v3.** The gap is {base['score'] - stock['score']:.5f}. The existing all-199 v3 score of 0.93480 is a different cohort. Public Harmonic Fusion scores 0.95765 on these same six clips.",
        "This uses frozen Cellpose cpDINO-ViT-B and fresh official matching of full 100-frame graphs: 600 volumes, 4,141 annotated nodes, 4,026 annotated edges and five annotated divisions.",
        "**Most of the original gap is temporal-edge accuracy.** Exact descriptive accounting assigns 0.09073 to excess edge FP, 0.05751 to missing true edges, 0.01163 to the change in node-count adjustment, and 0.01205 to the division bonus. The FP/FN terms average both substitution orders because Jaccard is nonlinear. They do not causally separate detector and linker effects.",
        "Stock edge TP/FP/FN is 3,674/540/352 versus v3’s 3,923/91/103. The 352 missed edges comprise:\n\n- 165 without an endpoint match in both raw Cellpose and selected graphs.\n- 31 whose available raw endpoint matches are lost during selection.\n- 115 with selected endpoint matches and a correct candidate edge that the solver does not choose.\n- 35 pruned at the top-five IoU step, plus six excluded by the ten-nearest-hypothesis cap.\n\nNone of those 41 missing selected-endpoint candidates exceeds 15 µm. Widening the distance gate alone addresses none of them.",
        "Raw annotated-node recall is 96.84%; after ultrack, 96.11%; v3, 98.91%. Of 540 evaluable false edges, 533 touch an unmatched endpoint; 262 of those endpoints remain within 7 µm of the expected GT node. Competing hypotheses or duplicates can affect matching, but a biological classification requires visual review. Missing-edge rates show no evident window-boundary spike: 8.43% at core boundaries versus 8.76% elsewhere. This is not a windowing ablation.",
        "**Division recall needs improvement in both methods.** On this pilot, both recover one of five divisions, with 219 versus three evaluable false divisions. Stock’s 3,324 total predicted forks are not all known errors. Across all 199 clips, v3 recovers 29/151 annotated divisions with 92 FP; division Jaccard is 0.11934.",
        "The refreshed [forum thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573) corrects its base-rate interpretation: sparse annotated track segments do not establish biological division prevalence. Displacement alone does not reliably separate divisions from continuations. Participants suggest temporal appearance and daughter-pair evidence; the small appearance probe is preliminary. Unknown branches must not automatically become negative training labels.",
        "**Our stock objective favors extra branches.** For fixed selected nodes, a fork versus one continuation plus an interior new track differs by IoU(second link)^4 + division_weight − appear_weight. Both event weights were −0.001, so any positive extra overlap improves that local fork comparison. Five actual CBC tests confirm the sign and the effect of unequal event costs. Raising both penalties equally preserves this preference. [Solver source](/home/mpf/code/kaggle/ultrack/ultrack/core/solve/solver/mip_solver.py:150), [upstream tuning guidance](https://royerlab.github.io/ultrack/optimizing.html).",
        "**Three controls were declared before new scoring.** The masks, hierarchy, candidate links, IoU values, 20-frame windows with overlap 5, eight CBC threads and 180-second per-window limit are unchanged:\n\n- Extra division cost: division −0.011; appearance/disappearance −0.001. Its 0.01 surcharge needs IoU above approximately 0.316 in the local fixed-node comparison. This is an objective-scale control, not a fitted biological prior.\n- Reference event costs: all three −0.1, copying only these weights from the pinned sparse-zebrafish configuration. This is not a paper replication.\n- No-division diagnostic: division −1.01; appearance/disappearance −0.001. With IoU^4 bounded by one, replacing the extra branch with a new track strictly improves the objective. Zero exported forks are required; this is not a division model.",
        "Measured full-panel results:",
    ]
    for arm in ["stock", *ARMS]:
        value = comparison["arms"][arm]
        s, c = value["summaries"]["pooled"], value["summaries"]["pooled"]["counts"]
        lines.append(f"- **{LABELS[arm]}: {s['score']:.8f}** (change {value['score_change_from_stock']:+.8f}); edge TP/FP/FN {c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']}; division TP/FP/FN {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {value['predicted_divisions']:,} total forks; {c['num_pred_nodes']:,} selected nodes.")
    if learned:
        lines.append("**Temporal image evidence is now integrated into ultrack’s joint optimization.** An explicit binary variable identifies the actual parent and two chosen daughter edges; a fork must select exactly one compatible pair. The existing v4 C4 optical model is evaluated at the current region centers, using each observation’s preceding/current/following images. Its symmetric pair-versus-no-fork logit is divided by the original source model’s temperature, clipped to ±8 and scaled by 0.01. The comparison sets this term to zero with identical pair constraints and candidates. No geometric mitosis gate, top-k pair cap or target-label fitting was used. Pairs are discarded only when replacing their weaker edge by a new track provably improves both objectives.")
        for arm in ("pair-control","pair-image"):
            s = learned["arms"][arm]["summaries"]["pooled"]
            c = s["counts"]
            lines.append(f"- **{LABELS[arm]}: {s['score']:.8f}**; edge TP/FP/FN {c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']}; division TP/FP/FN {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {learned['arms'][arm]['predicted_divisions']:,} total forks.")
        lines.append(f"Image-term score change versus the exact pair control: **{learned['paired_score_changes']['pooled']:+.8f}**, with embryo changes 44b6 **{learned['paired_score_changes']['44b6']:+.8f}**, 6bba **{learned['paired_score_changes']['6bba']:+.8f}**. [Learned-division evidence and checks](../results/cellpose-ultrack-learned-divisions-20260914/comparison.json). Direct optical fits are routed from the opposite embryo; synthetic initialization/replay and original generator temperature remain inherited. This tests an adapter, not new source calibration or an independent biological posterior.")
        lines.append("**The frozen optical extension fails its matched control.** It gains one division TP but adds nine division FP, loses 19 edge TP and adds 30 edge FP. The recovered event is 44b6_a21120c2 at frame 52; stock instead recovered the frame-zero event in 6bba_fe670320. The optical arm therefore demonstrates changed event recognition, not increased aggregate division recall. Its logits do not reliably rank errors below the recovered event: six evaluable false divisions have a larger image logit than that true division.")
        lines.append("The [post-prediction branch audit](../results/cellpose-ultrack-learned-divisions-20260914/branch-audit.json) finds nine of 29 evaluable false divisions with no continuation on one daughter, versus none of the single recovered true division. Twenty false divisions already continue for at least one frame. Requiring two continuation steps would remove 12 of the current false divisions and the recovered true division; these are graph diagnostics, not rescored counterfactual predictions.")
    if persistent:
        lines.append("**A separately declared follow-up adds one-frame daughter persistence.** Every selected daughter must have a selected outgoing edge when a following movie frame exists. Clip-final daughters are exempt. Committed outgoing edges constrain right window boundaries, and daughter continuation is enforced for inherited forks at left boundaries. The final exported graph must satisfy the rule everywhere. This is motivated by the inspected pilot errors; it is not an unseen test or a duration sweep. The unchanged image evidence and pair bank are reused in both arms.")
        for arm in ("pair-control","pair-image"):
            s = persistent["arms"][arm]["summaries"]["pooled"]
            c = s["counts"]
            lines.append(f"- **{LABELS[arm.replace('pair-', 'persistent-')]}: {s['score']:.8f}**; edge TP/FP/FN {c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']}; division TP/FP/FN {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {persistent['arms'][arm]['predicted_divisions']:,} total forks.")
        lines.append(f"Image-term score change with persistence held fixed: **{persistent['paired_score_changes']['pooled']:+.8f}**; 44b6 **{persistent['paired_score_changes']['44b6']:+.8f}**, 6bba **{persistent['paired_score_changes']['6bba']:+.8f}**. [Persistence plan](../results/cellpose-ultrack-persistent-divisions-20260914/plan.json), [results and checks](../results/cellpose-ultrack-persistent-divisions-20260914/comparison.json). The no-division control remains a diagnostic; suppressing all forks does not solve mitosis recognition.")
        si = persistent["arms"]["pair-image"]["summaries"]["pooled"]
        sc = persistent["arms"]["pair-control"]["summaries"]["pooled"]
        previous = learned["arms"]["pair-image"]["summaries"]["pooled"]
        lines.append(f"Persistence improves the optical arm by **{si['score']-previous['score']:+.8f}**, removes seven evaluable division FP, retains one TP, recovers five temporal edges and removes 15 edge FP. However, the optical term with persistence still loses {sc['counts']['edge_tp']-si['counts']['edge_tp']} correct temporal edges and adds {si['counts']['edge_fp']-sc['counts']['edge_fp']} false temporal edges versus its matched no-image control. These gains therefore do not justify replacing the simpler control. All 994 optical-arm forks obey the final-graph rule; 55 occur at the last transition, where future daughter persistence cannot be observed. All forks include unannotated cells and are not all known false divisions.")
    lines.extend([
        "The forum screenshot uses earlier figure numbering: its sparse-zebrafish Figure 6 is Figure 4 in the [published paper](https://www.nature.com/articles/s41592-025-02778-0). The [linked implementation](https://github.com/royerlab/ultrack_supplementary/tree/14d24aa3cada922d2ff7ac5b8adf39df85658b44/configuration/sparse_zebrafish) uses a learned foreground/contour U-Net on the dense channel, supplying different hypotheses from our hard Cellpose masks. The published Figure 6 is a separate neuromast experiment using fine-tuned Cellpose, evaluated with a different metric. [External image metadata](https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/.zattrs) confirms Kaggle’s ZYX spacing of 1.625/0.40625/0.40625 µm. Same spacing does not establish independent embryo provenance.",
        "**The remaining modeling gap is continuation identity and calibration at Cellpose proposals.** The temporal pair reward is implemented and tested; continuation links still use overlap. A further model should evaluate learned continuation scores at the actual Cellpose anchors and train division-versus-continuation-plus-birth competition on observed source-embryo events. Valid daughter proposals excluded by the top-IoU link shortlist also need coverage. Fit event costs on the same score scale; signed association logits require an identity transform, not a fourth power. See the repo’s [existing adapter design](focus-detector-ultrack.md).",
        "For fitting, use observed continuations and divisions as supervised evidence while keeping unknown branches and track truncations unknown. Group crops and every learned component by embryo, and audit external-volume overlap before using it for transfer validation. These reused six clips cannot independently validate a division model. v3 has documented upstream label exposure, so this comparison cannot isolate architecture from training-data effects.",
        validation,
        "The first cache-reuse attempt omitted ultrack’s geometry metadata file and failed before optimization. Copying and verifying the original metadata fixed the runner without changing costs or candidate banks. Failure logs remain beside each run. No mask inference was rerun, and no selected baseline was replaced.",
        f"[Declared plan](../results/{STUDY}/plan.json), [all cost controls and validation](../results/{STUDY}/comparison.json), [matched error audit](../results/{ROOT.name}/error-analysis.json), [all-199 v3 receipt](../results/strong-tracker-v3/fresh_selected_score_summary.json), [reproduction](../tools/cellpose_ultrack/README.md).",
    ])
    path = REPO / "docs/cellpose-ultrack-error-analysis-20260914.md"
    if persistent:
        lines.insert(0, "**Latest result: temporal daughter-pair scoring plus one-frame daughter persistence scores 0.82416, versus 0.79986 for stock, but 0.82814 for its matched no-image control.** Evaluable false divisions fall from 219 to 22 versus stock, while true recovery remains one of five. The structural follow-up improves the optical-only variant by 0.00474 but does not establish a better division model than the simpler control. The six clips are a reused development panel, not independent validation.")
        lines.insert(1, "The [current error audit](cellpose-ultrack-current-errors-20260914.md) separates the latest graph’s missed endpoints, rejected links and false temporal edges, with a fresh score comparison against the matched control and v3.")
    path.write_text("\n\n".join(lines) + "\n")
    print(path)


def current_audit_view(audit):
    t = audit["totals"]["current"]
    g = audit["gap_decompositions"]["v3"]
    s = audit["summaries"]["current"]["pooled"]
    edge_share = (g["excess_edge_false_positives"]+g["missing_true_edges"])/g["total_gap"]
    paragraphs = [
        f"Ordinary temporal tracking dominates the remaining gap: {100*edge_share:.1f}% of the {g['total_gap']:.5f} score difference to v3 comes from false and missed edges. The division-bonus difference contributes {100*g['division_bonus']/g['total_gap']:.1f}%. Current edge TP/FP/FN: {s['counts']['edge_tp']}/{s['counts']['edge_fp']}/{s['counts']['edge_fn']}.",
        "Of the 391 missed edges, 197 have both endpoints matched: 155 correct candidate links are rejected and 42 are pruned before optimization. Another 194 lack a matched endpoint: 165 already lack a raw Cellpose endpoint match, and 29 lose available raw matches during ultrack selection. No missing candidate between matched endpoints exceeds the 15 µm distance gate.",
        "374 of 376 false edges touch an unmatched prediction. In 186 cases that point is still within 7 µm of the expected annotated cell, which was assigned to another prediction. This supports investigating competing detections and which observation the link chooses; the matching result alone cannot distinguish fragmentation, nearby real cells or duplicate centers.",
        "Endpoint failures warrant a localization check. Among 160 unmatched annotated nodes after selection, 101 have the nearest predicted center between 7 and 10 µm away, while 59 have none within 10 µm. Even before ultrack, 80 of the 131 raw Cellpose misses fall in that 7–10 µm band. Proximity alone does not distinguish an off-center detection from a different nearby cell.",
        "The 6bba_af149c94 and 6bba_fe670320 clips contain 442 of 767 edge FP+FN counts (57.6%). Division errors cluster elsewhere: 13 of 22 false divisions are in 44b6_a21120c2. Counts reflect different annotation coverage across clips.",
        "The image model’s incremental regression has a different pattern from the overall error burden: versus the matched control it adds 22 false edges from forks and removes one false edge from single-child links. Thus the remaining total error is mostly ordinary tracking, while the image term’s extra FP cost is concentrated in branching.",
    ]
    return dict(paragraphs=paragraphs,
                missed=[dict(label=label,count=t["fn_reasons"].get(key,0)) for key,label in CAUSES.items()],
                falseEdges=[dict(label=label,count=t["fp_categories"].get(key,0)) for key,label in (
                    ("matched_source_unmatched_target","Matched source → unmatched target"),
                    ("unmatched_source_matched_target","Unmatched source → matched target"),
                    ("both_matched_wrong_identity","Both matched, wrong identity"))],
                gap=[dict(label=label,value=g[key],fraction=g[key]/g["total_gap"]) for key,label in (
                    ("missing_true_edges","Missing true edges"),("excess_edge_false_positives","Excess false edges"),
                    ("node_count_adjustment","Node-count adjustment"),("division_bonus","Division bonus"))],
                clips=[dict(dataset=r["dataset"],fp=r["edge_fp"],fn=r["edge_fn"],divisionFp=r["division_fp"]) for r in audit["rows"] if r["arm"]=="current"])


def write_current_report(audit, view):
    t = audit["totals"]["current"]
    lines = ["**Most remaining error is ordinary frame-to-frame tracking.** This audit concerns the latest temporal-image + one-frame-persistence arm, score **0.82415820**, on the same six full 100-frame clips. Its matched control scores 0.82813536; selected v3 scores 0.97178168.",
             *view["paragraphs"],
             "The 155 rejected correct candidates face conflicting selected links: 80 have the source linked elsewhere and the target assigned another parent; 36 have the source linked elsewhere while the target starts a new track; 39 have the source end while the target has another parent. Thus every one of these misses competes with an alternative selected connection. This does not prove that all conflicts are local decisions; ultrack optimizes a joint graph.",
             "**The remaining 0.14762348 score gap to v3 breaks down as follows:**"]
    lines.extend(f"- {r['label']}: {r['value']:.8f} ({100*r['fraction']:.1f}% of the gap)." for r in view["gap"])
    lines.extend([
        "This is exact descriptive metric accounting, with FP/FN contributions averaging both substitution orders. It is not a causal detector/linker decomposition. Division errors can also create false temporal edges; the division-bonus term does not capture every consequence of a bad fork.",
        "Current node matching recovers 3,981/4,141 annotated nodes (96.14%), versus v3’s 4,096 (98.91%). Current and matched-control graphs both have 3,832 annotated edges with matched endpoints; v3 has 3,972. Relative to v3’s 103 edge FN, the current 391 comprise 140 additional misses with unavailable matched endpoints and 148 additional misses despite available endpoints.",
        "**What the image model changes:** both persistence arms have the same aggregate matched-node and available-edge counts. The image arm increases rejected correct candidates from 139 to 155, and adds 21 net edge FP. The 44b6_a21120c2 clip alone adds 24 FP and six FN versus the control, while recovering the frame-52 division. Its 13 division FP account for most of the image arm’s 22. This points to the added image reward favoring wrong associations in that clip. Solver tolerances can also change individual edges; this comparison does not establish the biological identity of unmatched detections.",
        "For that incremental comparison, false edges from forks increase from 20 to 42, while false edges from single-child sources decrease from 335 to 334. The net extra 21 FP therefore consists of 22 extra fork-associated FP minus one single-child FP. Overall error burden and the optical term’s specific regression must not be conflated. The image arm loses 31 previously correct edges and gains 15 different correct edges, producing the net loss of 16.",
        "All four missed annotated divisions have raw Cellpose matches on their parent and both daughter sides under the official division-window matcher. Three retain matches on all three sides after selection; the frame-26 event in 6bba_e16ffc58 loses one daughter-side match. Side coverage is diagnostic and does not guarantee an available compatible daughter pair or a recoverable connected lineage.",
        f"Core-boundary missed-edge rates are {t['boundary_missed']}/{t['boundary_edges']} ({100*t['boundary_missed']/t['boundary_edges']:.2f}%) versus {t['interior_missed']}/{t['interior_edges']} ({100*t['interior_missed']/t['interior_edges']:.2f}%) in the interior. This descriptive comparison gives little indication of a large boundary-specific concentration, but is not a windowing ablation.",
        "**Priority suggested by the evidence:** improve localization at Cellpose centers and score continuation identity at those actual proposals; test candidate-link coverage before widening a distance threshold. A larger global division penalty addresses only a small part of the remaining errors. Train and calibrate any new model by embryo, and retain this reused panel as development evidence.",
        "Fresh official scores for all 24 frozen current/control/stock/v3 graphs agree with their saved receipts. Graph hashes, the full candidate banks, actual annotation scales and the pinned unmodified scorer were checked. No model, prediction, link bank or tracking parameter changed.",
        "[Machine-readable audit](../results/cellpose-ultrack-persistent-divisions-20260914/error-analysis.json), [all experiments](cellpose-ultrack-error-analysis-20260914.md), [division-window evidence](../results/cellpose-ultrack-persistent-divisions-20260914-pair-image/division-diagnostics.json), [official metric description](../reference/overview/evaluation.md).",
    ])
    destination = REPO / "docs/cellpose-ultrack-current-errors-20260914.md"
    destination.write_text("\n\n".join(lines)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
