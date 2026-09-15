# Incumbent provenance audit

**The dominant ensemble component's published training manifest includes every
clip in our detector assessment. The 99.50% local recall is training-exposed,
not a held-out generalization result. No evidence of hidden competition-test
label access was found.** Those are separate claims.

The exact secondary weight is SHA256
`9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`.
It contributes 80% of the incumbent's detection-logit ensemble; the primary
contributes 20%. The downstream association ensemble has different weights and
gating, so the 80% figure must not be applied to its edge predictor.

## Evidence tied to the actual checkpoint

The released [seed-314159 dataset](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1)
contains `SNAPSHOT_MANIFEST.json`, `split_manifest.json`, `training_config.json`
and `history.csv` alongside the weights. All six snapshot file checksums pass,
including the exact weight hash in our incumbent inference receipt.

- The training list contains **199/199 current public training clips**: both
  embryos, with no entries outside that public train inventory.
- The monitoring list, called `test` in the split file, contains **40 clips**.
  **All 40 also occur in training**, and all belong to embryo `44b6`.
  This is not a held-out validation split.
- **All 40 assessment clips** in our detector benchmark occur in the training
  list; **nine also occur in the monitoring list**.
- The history records **400 epochs**. The selected checkpoint is from
  **epoch 381**, with monitoring `accuracy × recall = 0.9779747766`.
  This multiplies edge-classification accuracy by detected-node recall, **not the official
  edge/division competition score**.
- The recorded configuration uses a two-frame TemporalUNet3D, feature width 32,
  layers 32/64/128, stride `(1,4,4)`, batch size 8, learning rate 0.0001,
  brightness/flip augmentation and detector negative weight 0.01.

The automated [audit receipt](../results/incumbent-provenance-20260914/audit.json)
lists the clip intersections, verified hashes and source paths. It can be
recreated with `PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python tools/audit_incumbent_provenance.py`.

## Correction: training source is available, though the exact run is incomplete

The earlier blanket description “we do not have training code” was too strong.
Both released packs contain a `repo/scripts/train_unet_transformer.py` and the
model definitions. The supplied trainer reads `fold_data["train"]` and
`fold_data["test"]` literally, builds both loaders, trains on the first list,
and saves the checkpoint with best monitoring accuracy × recall. It does not
exclude intersections between those lists.

However, this source is not a complete reproduction of the recorded snapshot
run: it lacks the named snapshot interval, split/config writing,
`checkpoint_last` and `eval_max_batches` facilities present in the run metadata.
The separate `source_scripts/train_full_frame_center_detector.py` is an
auxiliary center-prior trainer; it must not be mistaken for the TemporalUNet
training run.

Checksums establish which shipped files accompany the model. They do not
independently attest every historical training action. The defensible result is
**verified overlap in checkpoint-linked published training records**, sufficient
to reject this benchmark as held-out evaluation.

## The primary checkpoint is less documented

The primary weight is SHA256
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`,
matching the [public support pack](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1).
Its manifest calls it a **400-epoch snapshot**, despite the dataset slug
containing `50ep`. It ships architecture configuration and training source,
but no equivalent primary training split or history. Do not assign it a
specific training population or horizon solely from the slug.

The secondary's documented exposure is enough to establish contamination of
the ensemble's local assessment, regardless of the primary's unresolved split.

## What the competition discussions establish

The relevant threads were inspected alongside the actual code and artifacts:

- In [Only 2 groups of embryo_id?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793),
  organizer Thibgolds states that the scored train and test embryo IDs do not
  overlap. Visible example test clips are not independent validation for a
  checkpoint trained on the public training embryos.
- [Cannot find a training notebook](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738210)
  and [Why train beyond 100 epochs?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739915)
  contain questions, with no substantive replies in the refreshed snapshots.
  They supply no evidence that the weights used hidden test labels.
- [Above the 0.94 line](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738276)
  discusses segmentation, tracking and proxy scores. A participant's assertion
  that leaderboard labels are dense is challenged in the same thread; it is
  not an organizer assurance and should not determine our metric interpretation.
- Public inference notebooks, including the locally mirrored
  [Harmonic Fusion version](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion?scriptVersionId=347965685),
  load the same released weights. Their public availability and scores do not
  make a reused public-training benchmark independent.

Training on all public training clips is compatible with preparing a final
model. The problem here is interpreting a score on those same clips as
generalization. No hidden test-label use or competition misconduct was
established by this audit; no author accusation or contact was made.

## Implications for embedding reuse

The incumbent can technically provide embeddings for Cellpose detections.
Its `encode()` produces 32-channel image features per frame; `_index_features()`
samples them at supplied centers, and `predict_edges()` accepts those sampled
features plus coordinates and positional embeddings. Detection and association
can therefore use different proposal generators.

Features must be sampled at the new Cellpose points, with the existing stride,
normalization and coordinate contract. Copying embeddings by nearest incumbent
node ID would entangle different objects. Concatenating new Cellpose features
would also require a trained projection/fusion head; the pretrained transformer
expects its existing feature width.

Using this encoder carries its historical training exposure into the new
pipeline. A source-only new head does not remove that exposure. Such a probe
can measure practical changes on a fixed development cohort, while a clean
generalization test needs a verified independent embryo or encoders trained
with the target embryo excluded from the start.
