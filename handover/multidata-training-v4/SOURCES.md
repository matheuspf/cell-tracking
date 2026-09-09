# Sources, exact review and verification scope

Reviewed repository: `matheuspf/cell-tracking`, branch `handover/strong-tracker-v3`,
commit `292ecc2569f3a16676de92d80e382be02024d904` on 2026-09-09.

Primary local evidence (paths relative to repository root):
- `results/strong-tracker-v3/final_report.md`: measured score, failed events,
  annotation counts, proposal explosion, teacher leakage and sampling limitations.
- `results/strong-tracker-v3/raw_decoder_trace.json`: actual solver objective and
  fork-dominance result. This limitation is tested, not a new v4 measurement.
- `docs/external-data-guide/{README,DATASETS,ACCESS,BRANCH_CONTEXT}.md` and
  `dataset_inventory.json`: original paths, hashes, grids, label quality, source
  terms, generator behavior, species counts and restoration instructions.
- `tools/biohub_external_data/data_adapter.py`: native/pooled coordinate handling,
  intensity representation, final-frame masking and source clone semantics.
- `tools/strong_tracker_v3/event_model.py`: nine-frame parent-centered model,
  equal-class objective and uncalibrated scores, motivating compatible pretraining.
- Root AGENTS.md and the competition-data skill: preserve inputs and environments.

Official model source inspected via GitHub:
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/train_unet_transformer.py
The temporal U-Net/node-transformer training path and sparse transition masks
are reference interfaces; the patched local predictor/checkpoints must still be
inspected before adapting them. Scorer pin remains the recorded v3 revision until
W400 verifies source drift and the current competition contract.

External source verification attempted in this authoring session:
- https://virtual-embryo-zoo.sf.czbiohub.org/dataset/danior
  fetched successfully; attributes zebrafish tracking to original-author Ultrack
  and lists no external image link. This does not establish physical calibration.
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330
  returned no readable discussion body in web retrieval; use the already archived
  author/organizer statements locally. No fresh permission ruling is claimed.
- https://ssbd.riken.jp/database/project/5-Keller-FishEmbryo/
  timed out. RIKEN statements in this plan are from the committed preparation
  audit; its source-use eligibility remains unresolved.

The plan does not depend on new web downloads or a new external-data search.
No raw microscopy, external source arrays, trained weights or full graph caches
were read in the authoring environment. Included tests exercise synthetic reference
fixtures only. Local Codex verifies data and implements/executes all training.
