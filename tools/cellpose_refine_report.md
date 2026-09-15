# Cellpose refinement — evaluation against both baselines

**The constant correction improves localization more than either learned head.
The final tracking-score change has not been measured for these refinements.**
“Failed promotion” referred to our conservative detector-stage guard, not an
observed decrease in the Kaggle metric. Treat the constant correction as a
candidate for the full tracking comparison, not as a method proven worse.

Open the standalone report in a browser at <http://127.0.0.1:8768/report.html>.
The [HTML file](../results/cellpose-refine-v1/report.html) also works offline;
Cursor may show its source, so use a web browser to open it. There is no TSX or
Cursor runtime requirement. To restart the local server from this repository:

```sh
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python -m http.server 8768 --bind 127.0.0.1 --directory results/cellpose-refine-v1
```

If the repository is on a remote host, forward port 8768 through Cursor/SSH and
open that forwarded address in your browser. No public hosting is required.

## Evaluation protocol

The fixed assessment contains **400 frames, 40 clips and 2,383 sparse annotated
node observations**: 597 from 44b6 and 1,786 from 6bba. Cellpose and all three
refinement variants emit the same 156,226 candidates with unchanged scores and
ordering. Each target uses a fit from the opposite embryo. Each seed row pools
the two training directions; it is not a seed ensemble.

The incumbent uses 116,533 candidates. Its training provenance is uncertain,
with evidence of evaluation exposure. Report its strong observed result, but do
not treat it as a clean generalization ceiling. Inherited exposure in the
released Cellpose checkpoint also remains unresolved. Both embryos were examined
before this study; none of these results establish unseen-embryo performance.

Recall is matched annotated nodes / annotated nodes, using one-to-one optimal
distance-weighted matching separately at each physical radius. It is not
prediction precision: unannotated cells cannot be classified as false positives
from these sparse labels. Integer output is the submission representation;
float results are localization diagnostics.

## Pooled integer recall at every evaluated radius

@@POOLED@@

The learned heads gain **77 and 104 matches at 3 µm** (+3.23 / +4.36 percentage
points) and gain **0 / lose 3 matches at 7 µm**. The constant control gains
**137 at 3 µm** (+5.75 points) and **3 at 7 µm** (+0.13 points). Its 1 µm and
2 µm recalls also improve. Seed 20260914 instead loses 1.13 points at 2 µm.

Pooled float censored mean error is **2.2380 µm** for Cellpose, **2.2133 / 2.1624**
for the heads and **2.0143** for the constant control. Each unmatched GT receives
a fixed 7 µm error; a matched GT receives its Euclidean error. Lower is better.
Integer annotation precision limits biological interpretation of subvoxel gains.

## Per-embryo results and dense-neighbor diagnostics

@@EMBRYOS@@

Close pairs are annotated pairs separated by at most 14 µm, with both cells
matched within 3 µm. Pair observations can share cells and repeat over time.
Sampled edge-endpoint availability is not the number of correctly predicted
links and is not a graph score. Pooled availability is 1,099/1,159 for Cellpose,
1,099 / 1,097 for the heads, and 1,100 for the constant control.

On identical GT sets matched within 7 µm before and after the head, mean absolute
X/Y errors increase in both embryo directions and both seeds. Z improves on
6bba. The [paired axis-error analysis](../results/cellpose-refine-v1/paired-axis-errors.json)
controls for changing matched populations.

## Exact engineering criteria and failures

These are **our detector-stage criteria**, not official competition criteria.
Every embryo/seed must satisfy all of the following:

- Full-count integer R3 improves.
- Q3, the mean R3 at K=100/200/400 candidates per frame, improves.
- Float censored mean error decreases.
- Integer R7 drops by no more than **0.20 percentage points**, separately at
  full count and at each K=100/200/400.

Every recall delta below is in percentage points relative to Cellpose; error
deltas are in µm. Negative error deltas are improvements.

@@GATES@@

The failed conditions are concrete:

- **44b6, head seed 20260914:** float error **1.7524 → 1.8343 µm**;
  K=100 R7 **274/597 → 272/597** (45.90 → 45.56%, −0.335 points);
  K=400 R7 **544/597 → 541/597** (91.12 → 90.62%, −0.503 points).
- **44b6, head seed 314159:** float error **1.7524 → 1.7733 µm**;
  K=400 R7 **544/597 → 542/597** (91.12 → 90.79%, −0.335 points).
- **44b6, constant correction:** K=100 R7 **274/597 → 268/597**
  (45.90 → 44.89%, −1.005 points). Its other conditions pass. Full-count R7
  loses only one match, 586 → 585, within the 0.20-point allowance.
- **6bba:** both heads and the constant control pass the coded criteria.
  Separately, head close-pair recovery falls **82/108 → 77/108 / 78/108**.
  That subgroup regression was an additional diagnostic, not a coded gate.

These gates were deliberately strict. On 44b6, two lost matches already exceed
the 0.20-point allowance. A K=100 failure may have little relevance to a tracker
that selects substantially more candidates. It warrants examination; it does
not establish a worse final score or justify discarding the constant control.

## Sensitivity to the observed clips

@@INTERVALS@@

These are exploratory paired percentile intervals from 10,000 resamples of
clips within each embryo, seed 14092026. All methods use the same resamples.
They quantify sensitivity to these 40 clips. Crops may overlap and only two
previously examined embryos exist; these are not biological-generalization
confidence intervals, and no multiplicity adjustment is applied. The constant
control was designed after seeing the head study, so its comparison is exploratory.

## Proposal count is a variable to optimize

**Holding the bank fixed isolated the effect of moving centers. It should not
be a permanent design constraint.** The existing count sweep is already useful:

@@BUDGETS@@

K is a per-frame maximum using original confidence ranks; frames with fewer
than K candidates retain all of them. K=400 → all for unchanged Cellpose adds
**52,280 candidates (+50.30%)** and recovers **112 extra annotated nodes at 7 µm**:
2,173 → 2,285, **91.19 → 95.89%**. This measures retaining more of the existing
bank. A bank larger than the current “all” bank has not been tested.

With the current proposals and a maximum 3 µm correction, maximum-cardinality
matching over feasible integer destinations gives optimistic caps of
**90.14% at 3 µm** and **97.52% at 7 µm**. Relative to the constant control, those
leave 301 and 36 recoverable annotated matches under an oracle. At least
**59 observations** remain unreachable at 7 µm under these restrictions.
These bounds ignore biological proposal identity and the official matcher's
distance preference. Adding proposals or enlarging the correction range can
raise the bounds; they do not limit a future detector.

## Effect on the official final metric

The released scorer was refreshed from GitHub for this report and matches the
installed revision `075fc5f5a52d11077f9dc2b074644618f26939e2`.
For each clip, with edge Jaccard J and submitted/estimated node ratio q:

```text
J = edge_TP / (edge_TP + edge_FP + edge_FN)
M = 1.1 − 0.1 × q
adjusted_edge_Jaccard = max(0, J × M)
score = weighted mean(adjusted_edge_Jaccard) + 0.1 × micro division_Jaccard
```

Clip weights are edge TP+FP+FN. The estimate is for all true node observations,
including unannotated cells, not the sparse GT count. Predicted unmatched cells
are not individually counted as false positives; edge FP accounting depends
on available annotated lineage evidence. Every submitted node still affects
the count multiplier, including isolated nodes. Smaller localization errors
have no separate score term but influence matching and linking.
[Official implementation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/metrics.py),
[official explanation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md).

An internal candidate bank is not the submitted graph. If a tracker can reject
extra candidates and the emitted graph stays the same, expanding the bank has
no direct score penalty. Emitting more nodes lowers M; adding useful nodes may
recover enough correct edges and divisions to offset that. Bad extra candidates
can also change one-to-one matches or create wrong links and divisions.

For one clip with unchanged division performance, positive multipliers and
q changing from q0 to q1, a score increase requires:

```text
J_new > J_old × (1.1 − 0.1 × q0) / (1.1 − 0.1 × q1)
```

At q0=1.0, q1=1.2 and J_old=0.80, the multiplier changes 1.00 → 0.98.
J_new must exceed **0.81633**: a **1.633-point** edge-Jaccard gain offsets
20% more submitted nodes. Multi-clip score changes must use the actual weights
and division counts, not this one-clip simplification.

The following illustrative count inputs were executed through the official
`per_sample_metrics` and `summarise` functions. **They are not model scores.**
All use a node estimate of 1,000 and division Jaccard 0.5.

@@EXAMPLES@@

**We do not yet have an end-to-end score comparison for these refinements or
larger proposal banks.** Sampled node recall and endpoint availability cannot
determine edge false positives, division quality, or full-clip count penalties.
In particular, recall × count multiplier would not be the competition score.

## Training and validation

The 398,755-parameter head consumes frozen CPDINO features from three views,
a native `7×25×25` intensity patch and subvoxel query displacement. It predicts
a physical offset bounded at 3 µm, including integer export. The encoder stays
frozen. Source GEFF observations at t=9/29/49/69/89 in all source clips supply
1,002 / 5,770 GT observations from 44b6 / 6bba. Center and jitter queries total
4,008 / 23,080, with 595 / 1,673 supported natural queries. Sets overlap and
are not independent cells. Unknown real proposals receive no supervised loss.
AdamW runs a fixed 3,000 updates per direction/seed with initial LR 0.0003.

The constant control fits three physical offsets by weighted Smooth-L1 using
source targets: `(0.508, 0.051, -0.034)` µm from 44b6 for 6bba, and
`(0.951, 0.175, 0.159)` µm from 6bba for 44b6. It uses equal natural/jitter
sampling mass before original query weights; this population objective is not
an exact replay of stochastic minibatch denominators. It was introduced after
the v1 results; no target labels fit either vector, but the research iteration
was informed by target results.

All four head checkpoints and 824 prediction files were hashed before scoring.
Both constant vectors and all their predictions were frozen before control
scoring. Eleven distinct CPU contract/geometry checks passed, alongside GPU
forward/backward and frozen-feature checks. All 412 cached images match their
canonical raw volumes. Two renamed raw-volume replays reproduce fitted features
and predictions exactly. Python audit hooks block annotation/cache reads;
these are not kernel sandboxes or complete Kaggle-runtime certification.

The 1,407 feature jobs produced 1.82 GiB of caches. Processing excluding GPU
queue time summed to 35.3 minutes; queue time summed to 71.7 minutes. Each
head's optimization/checkpoint phase took about 9.3 seconds excluding queue
time and source-cache loading, with 0.30 / 0.57 GiB peak reserved GPU memory.
Subsequent cached-head studies are inexpensive.

## Next experiment, selected by the actual score

Compare baseline centers, constant correction and both head variants under a
common frozen tracker on complete clips. Cross this with the current proposal
bank and two broader settings selected on source data; permit the same tracker
to reject candidates. A denser bank does not require unfreezing the encoder.
Candidate routes include a lower object threshold, split proposals for merged
masks, or a complementary detector trained from verified sources.

The primary endpoint is the official combined score. Report edge and division
TP/FP/FN, submitted/estimated node ratio, 1–7 µm recall, close-pair recovery and
runtime for each embryo. Freeze all recipes before opposite-embryo scoring.
Use smaller-distance recall to diagnose gains and failures, not to substitute
for final-score measurement. An untouched embryo remains necessary for stronger
evidence about generalization and inherited checkpoint exposure.

## Reproduction and artifacts

- [Complete training implementation and commands](../tools/cellpose_refine/README.md)
- [Frozen head metrics](../results/cellpose-refine-v1/metrics.json)
- [Constant-control metrics](../results/cellpose-refine-v1/offset-control.json)
- [Integer-feasible capacity](../results/cellpose-refine-v1/integer-capacity.json)
- [Artifact integrity validation](../results/cellpose-refine-v1/validation.json)
- [Fitted raw-volume replay](../results/cellpose-refine-v1/trained-raw-replay.json)
- [Paired clip-resampling receipt](../results/cellpose-refine-v1/paired-clip-bootstrap.json)
- [Official-function count examples and source verification](../results/cellpose-refine-v1/official-node-count-examples.json)

Regenerate this report, its portable HTML and derived analysis with
`PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /kaggle/envs/cell-tracking-notebooks/bin/python tools/cellpose_refine_report.py`.
No training, predictions or frozen v1 metrics are changed by the report builder.
