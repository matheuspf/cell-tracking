# Evidence review and change of direction

Reviewed parent: `cec119434f7001ea79beeb137a8f1063e5b0deeb` on
`handover/multidata-training-v4`, September 10, 2026. Sources are paths at that
revision unless stated otherwise. The authoring session did not access microscopy
or execute GPU experiments.

## 1. V4 really trained, but did not produce an adopted gain

`results/multidata-training-v4/final_report.md` and `NEXT_AGENT.md` report 804,000
retained optimizer updates across 69 saved fits, 4.257 summed timed-loop hours,
21 complete variants and 4,179 official per-sample scores. The full real-only C1
and its repeat abstained. Primary C4 gained 0.000146776125 pooled but regressed
on 6bba; its second seed lost 0.000260245840 pooled. The largest descriptive row,
C4_m6 = 0.935178370257, also regressed on 6bba. C0 remains 0.934802374260586.
Do not rebrand the descriptive maximum as an incumbent.

The correct lesson is not that no amount of external data can help. It is that
these data, representations, calibration and integration did not transfer enough.
The completed compute, corrections and matched controls should not be dismissed
as a missing-training problem.

## 2. The entire hypothesis class was still strongly constrained

`tools/multidata_training_v4/models.py` has a small triplanar frame encoder,
masked temporal pooling, geometry MLPs, six candidate daughters and fifteen pair
scores. This is not a replacement temporal association backbone. The detector
is a center-query/offset model. It refines incumbent points rather than detecting
a new full-volume cell population. The real-only and synthetic detector arms
also rebuild continuations, and that rebuild itself causes much of the loss.

`final_report.md` candidate attrition reports 149 matched exact GT parents,
113 with both daughters matched, 104 exact-next-frame pairs inside the pool,
36 after the gate, one after the margin and zero newly accepted exact-ID edits.
Those are strict-ID diagnostics, not the official timing-tolerant division TP
counts. They show why feeding larger datasets into the same gated edit mechanism
need not produce useful graph changes. Loosening a threshold by itself previously
produced too many false forks; a new representation/decoder is the proposal here.

C7 learned its rendered Zoo validation domain but did not improve Biohub. More
rendered examples are therefore not the primary next experiment. No downloaded
Zoo source has paired real microscopy in this workspace. Dense synthetic labels
are not dense real biological ground truth.

## 3. Two major alternatives have not been evaluated here

**Pretrained general-purpose 3D linking:** HOCT uses an edge-centric transformer;
Trackastra CTC is a separately trained association transformer supporting 2D/3D.
They use segmented-region descriptors and temporal context. Their public models
and implemented mask-to-graph routes exist; their quality on our embryos is
unknown. This is a new representation and full linking engine, not a larger v4
MLP on the same fifteen pairs. See SOURCES.md and model_registry.json.

**New image-supported candidates with complete redecoding:** process native
images with the available DeepCenter and U-Net evidence, retain lower-confidence
but plausible alternatives, build instance supports, and decide one-to-one
continuations/two-child divisions/births/deaths jointly. Existing final nodes,
old fork paths and a 10% edit budget must not be the only legal universe.

The v3 raw-decoder trace showed that division_weight=1.2, edge reward <=1 and
free appearance strictly rule out raw forks. V4 tested a corrected objective
in fixtures but deployed local event edits; its original-objective full-data
control was a structural no-op. It did not globally rerun a replacement native
lineage solver on fresh dense evidence. An isolated raw-graph redecoding arm is
therefore substantive new work, not a replay of a completed global experiment.

## 4. Target accounting, not a promise

Incumbent edge TP/FP/FN = 123135/4965/5748; division TP/FP/FN = 29/92/122;
4,108,943 predicted nodes. Adjusted edge contribution is 0.9228682178819851.
The gap to 0.95 is 0.015197625739414.

Holding adjusted edges unchanged, 66 division TP with 92 FP would score
0.9500287117091456 (85 division FN); 60 TP with 60 FP would score
0.9513042368393311. These are calculated requirements, not achievable forecasts.
Real graph edits change edge counts, matching and division paths together.
`target_budget.py` regenerates this arithmetic and clearly labels it.

The goal is to learn better complete lineages. No annotation-count estimate,
scoring labels, hidden test identity, or retrospective oracle may enter inference.
Sparse-label and reused-embryo limits survive every change of architecture.
