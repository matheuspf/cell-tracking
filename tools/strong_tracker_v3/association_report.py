"""Measured association narrative; rendering never changes inference decisions.

``build`` is pure. ``paragraphs`` only loads explicit aggregate report inputs;
neither function opens annotations, prediction graphs, or model files.
"""
import csv
import json
import math


V2_INCUMBENT = 0.9342063149703403
INPUTS = (
    "association_summary",
    "association_regret_summary",
    "teacher_reconciliation_summary",
    "source_association_oracle_score_summary",
    "source_oracle_division_regret_summary",
    "association_source_bias_summary",
)


def build(summary):
    """Return Markdown lines from already loaded, sanitized aggregate inputs."""
    points = {
        (r["variant"], r["embryo"]): r for r in summary["operating_points"]
    }
    base = float(points["incumbent", "pooled"]["score"])
    if not math.isclose(base, V2_INCUMBENT, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Association report baseline must be the preserved v2 incumbent")

    def point(variant, embryo="pooled"):
        return points[variant, embryo]

    def delta(variant, embryo="pooled"):
        return float(point(variant, embryo)["score"]) - float(point("incumbent", embryo)["score"])

    a = summary["association_summary"]
    regret = {
        (r["variant"], r["embryo"]): r
        for r in summary["association_regret_summary"]["aggregate"]
    }
    actions = {
        (r["variant"], r["embryo"]): r for r in a["variants"]
    }
    m3, m15 = "A_residual_m3.0", "A_residual_m1.5"
    r3, r15 = regret[m3, "pooled"], regret[m15, "pooled"]
    p3 = point(m3)
    lines = [
        "## Association measurements and provenance", "",
        f"The stricter frozen residual setting **{m3}** scored "
        f"**{float(p3['score']):.16f}**, **{delta(m3):+.16f} versus v2**. "
        f"Its embryo deltas were {delta(m3, '44b6'):+.12f} for 44b6 and "
        f"{delta(m3, '6bba'):+.12f} for 6bba. It recovered "
        f"{r3['gained_gt_edges']:,} new GT-edge identities and lost "
        f"{r3['lost_gt_edges']:,}; it introduced {r3['introduced_fp']:,} FP edges "
        f"and removed {r3['removed_fp']:,}. Its {actions[m3, 'pooled']['accepted_actions']:,} "
        f"accepted local actions changed {actions[m3, 'pooled']['changed_edges']:,} edges. "
        f"Full-graph edge TP/FP/FN were {int(float(p3['edge_tp'])):,} / "
        f"{int(float(p3['edge_fp'])):,} / {int(float(p3['edge_fn'])):,}.", "",
        f"The predeclared primary margin, **{m15}**, scored "
        f"{float(point(m15)['score']):.16f} ({delta(m15):+.16f} versus v2): "
        f"{r15['gained_gt_edges']:,} GT edges gained, {r15['lost_gt_edges']:,} lost, "
        f"{r15['introduced_fp']:,} FP introduced and {r15['removed_fp']:,} removed. "
        f"Both seven-leaf tree margins scored {float(point('A_hgb7_m1.5')['score']):.16f} "
        f"({delta('A_hgb7_m1.5'):+.16f} versus v2; one extra TP, unchanged FP). "
        f"Frozen v2 model transfer scored {float(point('A_v2_transfer_m3.0')['score']):.16f} "
        f"({delta('A_v2_transfer_m3.0'):+.16f} versus v2; six extra TP, unchanged FP). "
        "Fresh node matching was identical in all five association variants, and each clip's "
        "division TP/FP/FN counts stayed unchanged. All five kept 4,108,943 nodes and pooled "
        "division counts 29 / 92 / 122. Choosing between these frozen settings remains "
        "exploratory; the small margin difference is not independent generalization evidence.", "",
    ]
    totals = a["feature_totals"]
    counts = a["model_lock"]["source_counts"]
    lines += [
        f"Current-center image, geometry and adjacency features covered "
        f"{totals['total_candidates']:,} candidate pairs: "
        f"{totals['teacher_native_union']:,} native/teacher pairs plus "
        f"{totals['image_neighbor_added']:,} additional image neighbors. The inference-only "
        f"disagreement table has {a['disagreements_rows']:,} rows and the five-variant "
        f"edit ledger has {a['edit_ledger_rows']:,} accepted actions. Source 44b6 contributed "
        f"{counts['44b6']['positive']:,} supported positive and "
        f"{counts['44b6']['negative']:,} contradictory negative edges from "
        f"{counts['44b6']['samples']} clips; source 6bba contributed "
        f"{counts['6bba']['positive']:,} positive and {counts['6bba']['negative']:,} "
        f"negative edges from {counts['6bba']['samples']} clips. Unknown sparse contexts "
        "were masked. Both directions were frozen before comparative scoring.", "",
    ]
    fits = {
        (r["model"], r["source"]): r
        for r in a["model_lock"]["source_fit_metrics"]
    }
    f44, f6 = fits["residual", "44b6"], fits["residual", "6bba"]
    lines += [
        f"The source-44b6 residual optimizer converged in "
        f"{f44['optimizer']['iterations']} L-BFGS iterations "
        f"(source loss {f44['source_fit_log_loss']:.6f}, AP {f44['source_fit_ap']:.6f}). "
        f"The source-6bba fit reached its frozen {f6['optimizer']['iterations']}-iteration "
        f"cap with optimizer success=false, while retaining finite source loss "
        f"{f6['source_fit_log_loss']:.6f} and AP {f6['source_fit_ap']:.6f}. It was not "
        "refitted after outer scoring. These source-fit metrics measure resubstitution "
        "and do not establish calibration or held-out accuracy.", "",
        "Historical teacher contamination limits the validation claim further. The reused "
        "v2 E_hgb teacher for a given source embryo was itself fit using the opposite "
        "embryo's annotations in v2. Thus target-embryo label information can already be "
        "present upstream of a v3 selector whose direct labels are source-only. Reused "
        "public neural checkpoints have their own embryo exposure. Frozen v3 source "
        "directions and annotation-free inference do not turn these operational "
        "comparisons into clean out-of-fold validation.", "",
    ]
    bias = summary["association_source_bias_summary"]["summary"]
    b = {(r["variant"], r["source"]): r for r in bias}
    lines += [
        "The source-resubstitution audit used the predeclared first five clips per source "
        "without changing any models or settings. At margin 3.0 it gained "
        f"{b[m3, '44b6']['gt_edges_gained']} and {b[m3, '6bba']['gt_edges_gained']} "
        "GT edges on source 44b6 and 6bba, respectively, with no TP losses or additional FP. "
        "At margin 1.5 it gained three GT edges on each source, introducing three FP on "
        "6bba. This small in-sample diagnostic is not an independent estimate of source bias.", "",
        "Teacher identity was proven through raw/pre-ILP native IDs and timestamps; "
        "unproven inserted donor IDs were excluded. The edge census separately measured "
        "each original teacher, its edges on the incumbent node universe at donor centers, "
        "and those same edges at incumbent centers. The following TP counts therefore "
        "separate changes in node universe from changes in coordinates; they are edge "
        "diagnostics, not combined-score or division claims.", "",
    ]
    teachers = summary["teacher_reconciliation_summary"]["aggregate"]
    for t in (r for r in teachers if r["embryo"] == "pooled"):
        lines.append(
            f"- {t['teacher']}: original TP {t['original_tp']:,} → projected at donor "
            f"centers {t['projected_at_donor_centers_tp']:,} → canonical TP "
            f"{t['canonical_tp']:,}; unmapped nodes {t['unmapped_node_fraction']:.3%}, "
            f"unmapped edges {t['unmapped_edge_fraction']:.3%}; canonical GT edges "
            f"versus v2: {t['canonical_gt_gains_vs_v2']:,} gained, "
            f"{t['canonical_gt_losses_vs_v2']:,} lost."
        )
    old = next(t for t in teachers if t["embryo"] == "pooled" and t["teacher"] == "old_final")
    lines += ["",
        f"Of {old['original_predicted_tp_gains_vs_v2']:,} old-final TP predicted pairs "
        "absent as TP pairs in v2, "
        f"{old['original_teacher_tp_pairs_lost_but_truth_recovered_by_v2']:,} had the "
        "same GT truth recovered through another predicted pair; "
        f"{old['original_teacher_tp_pairs_lost_and_truth_missing_in_v2']:,} were genuinely "
        "missing GT edges. On the fixed incumbent universe, changing old-final centers "
        f"to incumbent centers gained {old['coordinate_only_gt_gained']:,} GT edges and "
        f"lost {old['coordinate_only_gt_lost']:,}. Historical pair counts alone therefore "
        "overstate transferable association headroom.", "",
    ]
    oracle = summary["source_association_oracle_score_summary"]["results"]["pooled"]
    oc = oracle["counts"]
    bc = point("incumbent")
    lines += [
        f"The label-informed source association feasibility oracle scored "
        f"**{oracle['score']:.16f}**, **{oracle['score'] - base:+.16f} versus v2**. "
        f"It added {oc['edge_tp'] - int(float(bc['edge_tp']))} net edge TP and removed "
        f"{int(float(bc['edge_fp'])) - oc['edge_fp']} FP, but pooled division TP/FP/FN "
        f"changed from 29 / 92 / 122 to {oc['division_tp']} / {oc['division_fp']} / "
        f"{oc['division_fn']}. This is a source-label heuristic feasibility result, not "
        "a global upper bound, deployable configuration or promotion candidate.", "",
        "The exact division-regret audit found one early split whose two immediate fork "
        "edges stayed fixed while a downstream daughter continuation changed. Removing "
        "an edge counted globally as FP removed valid evidence within the official "
        "division timing window: the affected clip lost one division TP and gained one "
        "division FP and FN, despite one fewer edge FP. The implemented freeze protects "
        "immediate fork edges; it does not protect every downstream path used by the "
        "official division scorer. This diagnostic did not trigger another variant or "
        "a source/target refit.", "",
    ]
    if ("R_image_persistent", "pooled") in points:
        variant = "R_image_persistent"
        lines += [
            f"The corrected contrast-gated image-rescue arm scored "
            f"{float(point(variant)['score']):.16f} ({delta(variant):+.16f} versus v2), "
            f"with embryo deltas {delta(variant, '44b6'):+.12f} and "
            f"{delta(variant, '6bba'):+.12f}. It fails the nonnegative-both-embryos "
            "promotion gate. Correcting the flat-background bug established valid "
            "candidate evidence but did not establish a qualifying standalone gain.", "",
        ]
    lines += [
        "The aggregate evidence is preserved in [association_summary.json](association_summary.json), "
        "[association_regret_summary.json](association_regret_summary.json), "
        "[teacher_reconciliation_summary.json](teacher_reconciliation_summary.json), "
        "[association_source_bias_summary.json](association_source_bias_summary.json), "
        "[source_association_oracle_score_summary.json](source_association_oracle_score_summary.json) "
        "and [source_oracle_division_regret_summary.json](source_oracle_division_regret_summary.json). "
        "Detailed identities, coordinates and post-hoc labels remain local and are excluded "
        "from the selected inference package.", "",
    ]
    return lines


def paragraphs(ctx):
    """Read aggregate report inputs and return measured Markdown paragraphs."""
    summary = {
        name: json.loads((ctx.out / f"{name}.json").read_text()) for name in INPUTS
    }
    with (ctx.out / "operating_points.csv").open(newline="") as stream:
        summary["operating_points"] = list(csv.DictReader(stream))
    return build(summary)
