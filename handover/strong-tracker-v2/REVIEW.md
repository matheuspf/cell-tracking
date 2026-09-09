# What v1 actually established, and where a larger gain could come from

Evidence date: 2026-09-08. Source commit:
`f14c9eb368a80ce4cc7aaebae016fb182b1b6978`.

## 1. The result is not a small improvement over the strong tracker

The provenance-clean classical tracker improved from 0.674116 to 0.676907.
Harmonic Fusion's diagnostic score fell from 0.911774 to 0.873322 when the same
primary selector was transferred to its candidates. None of the 72 tested
learned/confidence settings improved the strong baseline's pooled score; the
best hindsight point was 0.911222. This rejects that particular transfer strategy,
not every possible learned native selector.

The clean primary gain decomposes into +0.007121 count effect and -0.004331 graph
effect. Its baseline cannot detect divisions at all. Optimizing it further would
not answer how to improve the strong pipeline.

Sources: [result summary](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/results/annotation-selection-v1/README.md),
[full report](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/results/annotation-selection-v1/report.md).

## 2. Selection was underexplored in three specific ways

**Wrong training population.** `train.source_data` loads `baseline/clean`, while
`public_analysis.run` loads the already trained source models and applies them to
`baseline/public`. There is no native Harmonic Fusion selector fit in that lane.
This preserves a frozen transfer experiment but leaves a major domain mismatch:
DoG peaks and repaired neural tracks differ in appearance, localization, confidence,
fragmentation and track lengths. The public confidence control also assigns zero
to 383,339 nodes (9.27%) with no nearby pre-ILP detection. Missing confidence is not
necessarily low cell probability: some final nodes are generated during repair.

**Few positive examples in the image probe.** The image fit sampled 30,000 source
candidates uniformly, six epochs, a three-layer 2D CNN over 3x32x32 triplanar panels,
no temporal image sequence. With the measured candidate prevalences, the expected
positive sample counts are about 198 and 1,392, not 30,000. These are expectations,
not recovered exact sample-label counts. Positive observations also repeat along
lineages and overlapping crops. The 120,000-candidate tabular subsample similarly
contains about 792 and 5,566 positives in expectation. The actual code/report
show a bounded feasibility probe, not a substantial test of learned visual tracking.

**The objective was a surrogate.** The learned output is binary sparse-node match
membership. Whole-tracklet ranking then uses a fixed 90th percentile of those
scores. This is not the expected loss of evaluated edges/divisions per removed
node, does not account for deletion-triggered rematching, and imposes a fixed
retention budget in every clip even when no low-risk removal exists.

Sources: [training](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/tools/annotation_selection/train.py),
[public transfer](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/tools/annotation_selection/public_analysis.py),
[image model](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/tools/annotation_selection/image_model.py),
[filter](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/tools/annotation_selection/filter_graph.py).

## 3. The stronger, measurable opportunity is division recovery

The strong graph matches 130,959 of 133,318 annotated nodes (98.2305%) but recovers
only 23 of 151 annotated divisions (15.23% recall), with 97 division false positives.
Its division Jaccard is 23/(151+97)=0.092742. This points to a division-specific
bottleneck, but does NOT establish that all missing divisions can be repaired
with current detections: average node recall need not equal daughter recall near
mitosis. Local candidate-coverage and postprocessing audits must decide that.

Hold the adjusted edge term at its observed 0.9024998207. These are calculated
scenarios, not measured improvements:

| Division TP | Division FP | Division Jaccard | Combined local score | Delta |
|---:|---:|---:|---:|---:|
| 23 | 97 | 0.092742 | 0.911774 | baseline |
| 23 | 0 | 0.152318 | 0.917732 | +0.005958 |
| 60 | 60 | 0.284360 | 0.930936 | +0.019162 |
| 75 | 50 | 0.373134 | 0.939813 | +0.028039 |
| 80 | 40 | 0.418848 | 0.944385 | +0.032611 |

Cleaning false divisions alone has limited headroom at unchanged TP. Recovering
missed true divisions is essential for a +0.02 to +0.03-sized change through this
term. Every real graph edit still needs complete rescoring, because edges and
matching may change too.

Source: [public counts](https://github.com/matheuspf/cell-tracking/blob/f14c9eb368a80ce4cc7aaebae016fb182b1b6978/results/annotation-selection-v1/public_summary.json).

## 4. Association repair has much more room than center recall suggests

The same baseline has 122,201 edge TP, 6,885 FP and 6,682 FN. If x wrong links
could each be replaced by one missing correct link, raw edge Jaccard becomes

`(122201+x)/(128883+6885-x)`.

At x=1,000, raw edge Jaccard rises from 0.900072 to 0.914171; at x=1,500 it becomes
0.921299. These are conditional raw-edge calculations, NOT exact adjusted-score
or leaderboard predictions. The locations of corrected links affect sample weights,
count-adjusted aggregation and divisions. Nevertheless, they identify a concrete
error budget worth addressing with cached candidate alternatives and local reranking.

## 5. What the real strong baseline requires from annotation filtering

Let J=0.90007218196 and A=0.90249982067 denote the measured raw and adjusted pooled
edge terms. Under unchanged per-sample FP counts, uniform fractional TP/node
retention in every sample, positive count multipliers, and unchanged divisions:

`A_filtered = u * [A + (1-r)*(1.1*J-A)]`.

This derives from the actual weighted aggregator, rather than substituting the
pooled predicted/estimated node ratio. Here `1.1*J-A=0.08757957948`.

Retaining 90% can gain at most +0.008758 through count reduction alone, even with
zero graph loss under those assumptions. Retaining 80% caps that gain at +0.017516.
To gain +0.024 at 70% retention requires 99.755% uniform TP survival; at 50%
retention it requires 97.909%. Independent-node deletion would translate the
latter to approximately 98.949% annotated-node recall, but a real tracklet selector
must be assessed with actual edge survival/rematching, not the square rule.

A native selector merits a bounded test. It should not be the sole bet for a
large gain, and retaining 90% is not an adequate count-only path to +0.024.

## 6. FOCUS-3D should answer a demonstrated missing-candidate problem

FOCUS-3D supplies per-frame 3D instance segmentation, not a complete temporal
tracker. The branch contains a technical assessment but no measured FOCUS run.
The primary project confirms segmentation and fine-tuning functionality:
[official project](https://github.com/yu-lab-vt/FOCUS-3D).

Prefer a targeted teacher/candidate experiment on ambiguous mitotic neighborhoods
if the audit finds merged daughters or missing centers. Keep the full-image model
replacement conditional on measured center/edge/division gain and end-to-end
runtime. High overall center recall does not rule out localization or mitosis-
specific benefits; it does make an unqualified detector replacement a weak first bet.
Do not upload competition microscopy to a hosted service. Reuse the branch's
FOCUS3D.md for access, provenance, licensing and runtime checks; obtain gated model
access only through already authorized credentials/terms.

## 7. Validation interpretation must improve, not halt development

Two embryos, confirmed overlapping clips, and contaminated public checkpoints
limit generalization claims. Both embryos' outcomes have already been inspected.
Use the existing baseline for operational diagnosis; source-only train/test label
separation remains useful but cannot make these reused outcomes untouched.

Improve overlap geometry when possible: registered crop/time translations can
support global coordinates and purged space-time blocks. A connected overlap
component is not automatically an indivisible training unit if overlapping
supports can be explicitly removed around a block boundary. This is a proposal,
not a claim that v1's 73 registrations already solve global reconstruction.
Do not use lack of a detected overlap as proof of independence. If grouping
remains unresolved, keep the limited fixed-configuration cross-embryo comparison
and report it honestly. Do not spend the entire experiment rebuilding validation
instead of repairing the strong tracker.
