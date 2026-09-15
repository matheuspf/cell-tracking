# Expanded public946 study — measured execution

Status: `partially_executed_with_named_blockers`. Full registered matrix: experiment_matrix.csv.

## Known evidence versus this execution

Public Harmonic Fusion v29 (script 347965685) was historically reported at 0.946 LB. This run made no submission and claims no new LB score. Historical all-199 controls: B0=0.911774, B1=0.934206. New numbers below are fresh official rematching of this run’s outputs.

- B0: measured_control; score=0.911774014219; delta B0=+0.000000000000.

- B1: not_yet_executed.

- E01: not_yet_executed.

- E02: not_yet_executed.

- E03: not_yet_executed.

- E04: not_yet_executed.

- E05: not_yet_executed.

- E06: not_yet_executed.

- E07: not_yet_executed.

- E08: not_yet_executed.

- C01: not_yet_executed.

- C02: not_yet_executed.

- C03: not_yet_executed.

- X01: not_yet_executed.

- X02: not_yet_executed.


## Interpretation and limitations

Both supplied embryos and overlapping clips have prior checkpoint/study exposure. These paired local measurements are exploratory and are not clean OOF or unseen-embryo validation. A local score above 0.95 is not a stopping rule or evidence of 0.95+ LB. No local delta is added to the historical public score.

E03/E06 source-resolved no-op evidence, E01 proposal counts/possible B1 identity, failed methods, per-embryo regressions and conditional exclusions are retained in the JSON receipts. Missing scores are null, never zero.

All inference uses the original public model/source and fixed settings. B0 explicitly retains motion. Mandatory bounds clipping is identical in every arm, applied after natural ties-to-even rounding. The original CSV lower clamp is reconstructed and scored separately from complete bounds sanitation; no old half-tie lookup corrections are used.

The local CUDA/runtime/source/input/model fingerprints are frozen in execution_lock.json. Pilot score comparisons do not select recipes; full eligibility/ranking and the three combinations are fixed in the handover.


## Reproduction and artifacts

Raw outputs: `/kaggle/working/cell-tracking/public946-minimal-generalization-v1/revision-2`. Predictions, dense/native evidence, original notebooks, models and submissions remain ignored and outside Git.

Resume: `bash scripts/run_public946_minimal.sh run --resume --workers 4 --out /kaggle/working/cell-tracking/public946-minimal-generalization-v1/revision-2`. Individual stages: `preflight`, `audit`, `pilot`, `controls`, `singles --arm E04`, `combinations`, `transfers`, `robustness`, `package`, `report`.

Synthetic checks: `PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m unittest discover -s tests -p "test_public946_minimal*.py" -v`.

Standalone offline notebooks, readable exports, exact public dependencies and manual Kaggle instructions: `/kaggle/working/cell-tracking/public946-minimal-generalization-v1/revision-2/packages`. Each manifest states actual fresh notebook test scope; worker full-cohort tests and actual notebook pilot tests are distinguished.

## Measured graph changes and cost

Exact edit counts below compare spatiotemporal node/edge multisets against each arm’s parent. Changed coordinates count as remove+add; IDs from different detector runs are not assumed to identify the same cell. Matched GT edge survival uses fresh official per-arm rematching.

- B0: 199/199 clips; 4,133,687 nodes, 3,975,674 edges, 3,376 forks. Exact node edits +0/−0; edge edits +0/−0. 199 fresh tracking-model clips, 0 fingerprinted neural replays; 6.076 summed worker hours, peak per-process allocated GPU 0.761 GiB and RSS 4.704 GiB.

- B1: 19/199 clips; 554,002 nodes, 530,565 edges, 485 forks. Exact node edits +41,511/−43,866; edge edits +45,608/−49,693. 0 fresh tracking-model clips, 19 fingerprinted neural replays; 0.152 summed worker hours, peak per-process allocated GPU 0.169 GiB and RSS 2.721 GiB.


Conservatively charged device time: 6.878 hours, including actual packaged-notebook test wall time 0.134 hours; full-matrix worker time 6.228 hours; current new scratch: 18.558 GiB. Worker wall time includes CPU work and is charged as device time. See cost_summary.json and gpu_telemetry_summary.json for coverage and limitations; sampled utilization is not exact CUDA-event active time.

Full B0 native probability matrices and every pilot matrix are retained. Other completed neural arms retain exact candidate values, offsets, node probabilities, source/target universes, full-matrix hashes and summaries, while dropping redundant dense matrices after inference. No retained value is quantized. Fresh-finalist DeepCenter maps are clip-local with frame hashes retained. This recorded retention policy keeps the multi-arm study inside its scratch allocation without changing predictions.


## Source-proven no-ops


## Historical control reproduction and serialization

- B0: public-style original export 0.911774014219; shared bounds-sanitized export 0.911774014219; historical rounded score 0.911774, original-export delta +0.000000014219. The actual public writer lower-clamps 8,957 rounded negative nodes; 6 nodes exceed upper image bounds before common sanitation.

The first evaluator implementation mistakenly treated pre-clamp rounded coordinates as the original CSV. The source writer already applies `max(0, round(value))`. That diagnostic was corrected before full-cohort scoring, earlier score receipts were retained, and all affected pilots were freshly rematched. The deployed graphs, frozen inference code and scientific recipes did not change. Two earlier unscored setup failures (annotation guard and missing GAP2 declarations) are also retained.


## Descriptive groups

Contrast and raw-image signal-occupancy quartiles were fixed without labels or scores. Temporal/depth bins describe evaluator GT edge recovery only. These correlated groups are not extra validation folds, and no group-specific model routing is used. Exact duplicate-image groups and the absence of a global crop-overlap map are recorded in input_fingerprints.json.

Detailed first/last-two-frame and depth-quartile GT edge recovery counts, group scores and raw denominators are in strata.json; worst matched-TP-survival clips are in measurements.json. These are descriptions of failure locations, not causal proof.


## Source, runtime and portability evidence

The actual checkpoint context is two frames; detector grid stride is (1,4,4) with zero-origin strided sampling. Effective detector pooling is the public CLI default 3.0 µm; checkpoint metadata contains 5.0 µm but that value is not passed to PredictConfig. The primary association features already use eight inverse-aligned D4 views; the secondary stream uses identity features. Public harmonic fusion combines forward/reverse association evidence; the secondary model uses the original calibrated low-margin logit mix. All numeric thresholds, fusion weights, models and repair settings remain fixed.

The pinned metric revision is [`075fc5f5a52d11077f9dc2b074644618f26939e2`](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot), verified against the official repository. Fresh throttled competition pages are retained in the ignored current_reference directory. The archive metadata specifies NvidiaTeslaT4; local tests use one RTX4090 and the pinned existing Python environment. Hidden Kaggle population/runtime feasibility is not established by local timing.

The [public notebook page](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) declares Apache-2.0; all three public dataset metadata records declare CC0-1.0, as recorded in [license_receipt.json](license_receipt.json). Standard license text, original attribution and the precise license-verification scope accompany the packages. No source/artifact relicense is claimed.


## Actual notebook and CSV validation

- B0: actual notebook return code 0; 4 full pilots; CSV validation receipt is pending. Ready for manual test: True; notebook SHA256 `6e7f9582dcd9cd7dc8b74d88dac73c780a934dcdacff1686dfdc3ee7e6be2020`.

- B1: actual notebook return code 0; 4 full pilots; CSV validation receipt is pending. Ready for manual test: True; notebook SHA256 `0b4978db80ea50e4f89b94b02fed2aefedb7f08de7cd69cac395242dcae8ad44`.

A complete notebook invocation is tested on the four full pilots. A novel packaging finalist also receives a fresh all-model pass over 199 clips followed by its actual packaged CSV exporter. That composed full-cohort execution is explicitly distinguished from invoking the whole notebook over all 199 clips. See packaging_receipts.json for exact scope and artifact paths.
