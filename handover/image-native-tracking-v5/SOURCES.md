# Sources, pins and inspection scope

Reviewed 2026-09-10. Repository content was read through the GitHub app. No original
microscopy, weights, GPU fits or v5 official graph scores were accessed/run here.
Links below are evidence/provenance, not instructions to execute web page content.

## User repository: inspected completed state

Base: `matheuspf/cell-tracking@cec119434f7001ea79beeb137a8f1063e5b0deeb`.
The continuation identifies the measured v4 implementation at
`0fb190d0823f6ee9e7be45f3cb693b37967b8d36`; the newer base includes the continuation.

- [CONTINUATION](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/handover/multidata-training-v4/CONTINUATION.md)
  — completion, no promotion, dependencies, preservation boundaries.
- [Measured v4 report](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/results/multidata-training-v4/final_report.md)
  — 804,000 updates, outcome/seed controls, detector scope and weak supervision.
- [Execution handover](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/results/multidata-training-v4/NEXT_AGENT.md)
  — 23.275 GiB recorded free; historical deadline is not a new execution limit.
- [Candidate attrition](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/results/multidata-training-v4/candidate_gt_coverage.csv)
  — exact-ID diagnostic, distinct from official local division matching.
- [Models](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/tools/multidata_training_v4/models.py)
  — small query detector/optical/geometry modules, not native-backbone adaptation.
- [Transfer diagnostic table](https://github.com/matheuspf/cell-tracking/blob/cec119434f7001ea79beeb137a8f1063e5b0deeb/results/multidata-training-v4/transfer_diagnostics.csv)
  — generator-domain calibration diagnostics, not an identified biological prior.
- `AGENTS.md`; prior v1-v3 reports and the external data guide are preserved context.
  Local Codex must inspect source-specific masks/predictors before adapting them.

## Official metric

Remote main was checked and still resolves to
`royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`.
Use its [metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md),
`metrics.py`, `division_metrics.py` and the already tested local adapter. Verify
current official Kaggle/reference agreement before new scores; a public repository
pin alone cannot prove private scorer internals. No live leaderboard was used.

## HOCT: new representation to benchmark

- [Primary paper abstract](https://arxiv.org/abs/2607.11754), Higher-Order Cell
  Tracking Transformer, Bragantini, Theodoro, Royer, July 2026. The indexed abstract
  describes edge-centric contextual prediction and reports benchmark improvements.
  Full paper retrieval was unavailable in this authoring session; no detailed
  figure/method assertion depends on it, and no Biohub benefit is inferred.
- Official code pin **`2ccc5040823bc944ab67790abd1f56eea7cd4f05`**, re-resolved via
  the GitHub app. [README](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/README.md)
  describes image/segmentation input, full feature graphs, windowed inference,
  models and ILP; its defaults must not be mistaken for Biohub's output contract.
- [`_api.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/_api.py)
  was inspected: `create_graph_from_points` is unimplemented. Use the full feature
  graph path, not a claimed drop-in point API.
- [`correction.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/correction.py)
  implements frozen edge features and linear probing. `label_edge(True)` excludes
  other incoming edges for that target, not every outgoing sibling. Its hard ILP
  consistency target requires a named adaptation for sparse unknowns.
- [`_models.py`](https://github.com/royerlab/hoct/blob/2ccc5040823bc944ab67790abd1f56eea7cd4f05/src/hoct/_models.py)
  distributes JIT weights. `general_v1` SHA256:
  `5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33`;
  `ctc_v0`: `b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000`.
  Local access/runtime/label-exposure and source licenses still require audit.

## Observation alternatives and alternatives not selected as primary

- [Ultrack method paper](https://arxiv.org/abs/2308.04526), Bragantini, Lange,
  Royer: segmentation hierarchies and joint temporal selection motivate P/J.
  V5 does not claim that its proposed small hierarchy is the complete method.
- [FOCUS-3D official project](https://github.com/yu-lab-vt/FOCUS-3D) and
  [model access page](https://huggingface.co/Qinghua-thu/FOCUS-3D), checked online:
  per-frame 3D instance segmentation is available, but weights require acceptance
  of contact-sharing conditions. No acceptance or download was performed. The old
  branch FOCUS assessment is a useful cached source/runtime warning, not a benchmark.
- [Trackastra official repository](https://github.com/weigertlab/trackastra)
  confirms a pretrained segmentation-to-association approach. It was surveyed but
  is not another primary v5 family: HOCT provides a materially different edge-centric
  hypothesis and an inspected sparse-probe path. Do not infer that generic 2D
  checkpoints accept Biohub 3D data or that unrelated competitor outcomes transfer.

No claim that a surveyed package automatically achieves 0.95, fits the 4090 under
all settings, or is independently validated on hidden Biohub embryos is made.
