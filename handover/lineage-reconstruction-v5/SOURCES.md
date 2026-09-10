# Source ledger — September 10, 2026

## Inspected project evidence

Parent repo/branch: `matheuspf/cell-tracking`, `handover/multidata-training-v4`,
commit `cec119434f7001ea79beeb137a8f1063e5b0deeb`.

- `results/multidata-training-v4/final_report.md` — no adopted gain, 804k retained
  updates, scope of D/G/I models, full score table, attrition and fresh runtime.
- `results/multidata-training-v4/NEXT_AGENT.md` and
  `handover/multidata-training-v4/CONTINUATION.md` — completed stages, source use,
  dependency/preservation boundaries and expired seven-hour window.
- `tools/multidata_training_v4/models.py` — actual compact detector/geometry/image
  architecture; confirms query refinement and bounded pair mechanism.
- `docs/multidata-training-v4.md`, root `AGENTS.md` — maintained runtime and paths.
- Prior branch `results/strong-tracker-v3/raw_decoder_trace.json` — raw fork
  dominance proof. Its scope is the original objective, not the v5 decoder.
- `docs/external-data-guide/` and v4 dataset-use results — paired synthetic data,
  weak Zoo trajectories, unpaired image/label limitations. Do not turn missing
  real modalities into zeros that purport to be equivalent feature distributions.

This authoring session inspected committed text/code, not raw microscopy or weights.

## Primary external sources checked

**HOCT** (edge-centric pretrained cell tracking):
- Repository: https://github.com/royerlab/hoct
- Source ref: `2ccc5040823bc944ab67790abd1f56eea7cd4f05`.
- `src/hoct/_models.py`: `general_v1`, `ctc_v0`, `general_v0` registry, checkpoint
  SHA verification and cache behavior. See model_registry.json for selected pins.
- `src/hoct/_api.py`: implemented `predict(labels=..., images=..., graph=...)`,
  physical scale, window_size, max_delta_t and graph creation. Critically,
  `create_graph_from_points` contains `pass` and is not usable.
- `src/hoct/correction.py`: `fit_from_labels`/ProbedModel sparse edge probe,
  supported incoming mutual exclusion, optional hard-ILP consistency on unknown
  edges. Disable that hard pseudo-target in the primary adaptation.
- Paper: https://arxiv.org/abs/2607.11754 — methodological motivation only.
  Its benchmark gains are not an estimate of Biohub performance.

**Trackastra** (independent pretrained association transformer):
- Repository: https://github.com/weigertlab/trackastra
- Source ref: `aa57a95160002e0fc70b915ab74178b39c99fd6a`.
- README: image+instance-mask API; division-enabled greedy and ILP modes; training
  support. CTC-shaped data alone does not make sparse Biohub labels complete.
- `trackastra/model/pretrained.json`: `ctc` explicitly supports dimensionality
  [2,3] and uses the v0.3.0 model release. `general_2d` and the SAM2-enhanced model
  are 2D. Do not invent a `general_3d` checkpoint.
- Registry blob observed: `799734e6da5b6bb08df8c537d9904815a65aee20`.

**FOCUS-3D** (optional independent masks, not a temporal tracker):
- https://github.com/yu-lab-vt/FOCUS-3D
- https://huggingface.co/Qinghua-thu/FOCUS-3D
- Current model card lists nuclei/general/membrane checkpoints and Apache-2.0
  weight terms. It still requires acceptance of contact-sharing access conditions.
  Authoring did not accept conditions, download weights or run the model.
- Prior repo assessment `handover/annotation-selection-v1/FOCUS3D.md` documents
  coordinate/resampling/predictor-caching caveats. Revalidate actual installed API.

Sources establish availability and API contracts, not an expected score. Record
actual model-specific terms, training-set provenance, versions and hashes locally.
Revalidate current Kaggle rules and runtime through official saved/source pages
before packaging; code license and weight/data license are separate matters.
