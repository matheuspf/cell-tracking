# V6 execution continuation

Executed on this branch on 2026-09-10 with the existing RTX 4090. This is a
measured execution record. The learned-segmentation/Ultrack study is **blocked**,
not completed using classical masks. No model/data/dependency downloads,
installation, license acceptance, ZIP delivery, new handover branch, merge or
Kaggle submission occurred.

Read [the final report](../../results/segmentation-tracking-v6/final_report.md),
[aggregate scores](../../results/segmentation-tracking-v6/ablation_scores.csv)
and [offline dashboard](../../results/segmentation-tracking-v6/dashboard.html).

## Measured selection

C0 is preserved at **0.934802374260586**. The independently runnable point control
P0 completed fresh official scoring on all 199 clips at **0.9348649864131336**,
delta **+0.000062612152548** (rounded). Per embryo: 44b6 **0.9317272564133575**;
6bba **0.9352841911367036**. Both improve slightly. The 0.95 target was not met.

P0's complete source fits and 199 graph decisions repeated exactly. Both full
unfamiliar-filename fresh clips passed candidate/feature, graph, CSV and early
offline-guard checks. The small point-control gain is retained in the v6
`selection.json`; the original C0 graph/model/selection locks remain unchanged.
This is not a bbox/mask gain and not successful learned-tool integration.

P0 retains the full incumbent native evidence: primary, secondary, augmentation
and teacher votes, the 38-field preselector feature schema and candidate bank.
It fits one L2 logistic residual per source embryo, with native logit offset,
L2=1 and max 250 iterations. It uses the unchanged v3 association decoder with
margin 3 and a 2% edit cap; current forks remain frozen. Source44 converged at
180 iterations. Source6 hit its 250-iteration budget without optimizer convergence;
its finite fixed-budget coefficients and predictions reproduced exactly.
There was no target-dependent extension of that budget.

The sparse-label rule refutes competing incoming parents of a known daughter;
it never makes an unrecorded second daughter negative. Both source models were
frozen before new target scoring. Repeated embryos and inherited checkpoint
exposure remain exploratory validation.

## Exact local artifacts and commands

Repository: `/root/code/kaggle/cell-tracking` (physical alias `/root/cell-tracking`).
Output: `/kaggle/working/cell-tracking/segmentation-tracking-v6`.
Runtime: `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`.

Durable small artifacts are `preflight.json`, `inputs.json`, `configuration.json`,
`models/P0_44b6.json`, `models/P0_6bba.json`, `model_lock.json`, `evaluation/C0/`,
`evaluation/P0/`, `control_receipt.json`, `repeatability.json`, `fresh_receipt.json`,
`fresh_point_receipt.json`, `image_checks.json`, `selection.json`, `status.json`,
`cli_check.json`, `blocked_arm_check.json` and `dashboard_check.json`.
Model coefficients and detailed clip receipts stay outside Git.

The small model JSON SHA256 values are:

- source44: `2ab4d37bc184aa8bed8ddaa5d45ae364b59e794a0d9029143460d7e120978199`
- source6: `938b9f1d4510cc992fbbc3ded648fd63cce8456da2cfa3dbd3f14fad6a80b18d`

Transient graphs, fresh exports, logs and raw optical overlays are in
`/dev/shm/cell-tracking-segmentation-v6`. They are RAM-backed and **do not survive
a reboot**. The existing immutable native package is
`/kaggle/working/cell-tracking/image-native-tracking-v5/inference_package_validation`;
manifest SHA256 `c0b6dae1a34defcd8d896bfc83eae42c961fe3c21effba04b054f7dd6efa217c`.
No new package or archive was created.

Executed study commands, from the repository:

```bash
scripts/run_segmentation_tracking_v6.sh preflight
scripts/run_segmentation_tracking_v6.sh controls
scripts/run_segmentation_tracking_v6.sh fresh
scripts/run_segmentation_tracking_v6.sh fresh-point
scripts/run_segmentation_tracking_v6.sh repeat
scripts/run_segmentation_tracking_v6.sh image-checks
scripts/run_segmentation_tracking_v6.sh report
```

`scripts/run_segmentation_tracking_v6.sh run` executes those stages sequentially.
It sets Python isolation and bounded numerical threads without activation steps.
All computations and outputs are v6-local; old studies remain read-only.

The delivered image-only command is `scripts/run_segmentation_tracking_v6.sh infer
--images IMAGE_DIRECTORY --output NEW_OUTPUT --source-model 44b6|6bba --arm P0`.
It processes one entire clip, selects the source model explicitly, and emits the
P0 `submission.csv` plus unchanged `C0.csv`. Both full-clip CLI executions are
recorded in `cli_check.json`. `--disable-v6` selects C0; the default arm is C0.
The learned arms fail explicitly before any baseline execution because there
is no validated selected learned-mask/hierarchy recipe. These commands are for
local inference; they do not submit anything.

The exact test and browser commands were:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m unittest discover -s tests -p test_segmentation_tracking_v6.py -v
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /root/.conda/envs/cell-tracking/bin/python \
  -m unittest discover -s handover/segmentation-tracking-v6 -p test_region_contracts.py -v
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools \
  /root/.conda/envs/cell-tracking/bin/python \
  -m segmentation_tracking_v6.dashboard_check
```

The direct repeatability and CLI validator modules are
`segmentation_tracking_v6.repeatability` and `segmentation_tracking_v6.cli_check`.
Use the same isolated runtime and thread environment as the wrapper.

## Dependency and execution blockers

- `/root/FOCUS-3D` has three 4.2 GiB checkpoints and a weights README, but no
  Python source or callable headless `infer_volume` backend. All three supplied
  SHA256 sums matched. `model_final_nuclei.pth` is the relevant compartment;
  its SHA256 is `b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a`.
  The annotation runtime also lacks `detectron2`. FOCUS API/preprocessing and
  complete dependency compatibility could not be validated without its source.
- Cellpose is absent in all inspected local runtimes/caches. The attempted
  explicit checkpoint `/root/.cellpose/models/cpsam` is absent. No fallback
  checkpoint was obtained and no package was installed.
- Ultrack source/package is absent. Installed `pyscipopt 6.2.1` does not provide
  Ultrack's hierarchy, links or lineage solver. No solver license was acquired.
- The persistent disk was already at **6.838 GiB free**, below the required
  8 GiB floor. No earlier artifacts were reclaimed. Only small code/model JSONs
  and receipts were persisted; RAM scratch admitted the independent controls.
  Persistent segmentation masks/databases and an all-frame pilot remain blocked.

The FOCUS/Cellpose and Ultrack adapter boundaries, retained occupancy, ownership,
feature and export contracts are implemented. **Their real model/Ultrack APIs
are unvalidated**. The actual mask-division graph integration, hierarchy reader
and U1 candidate-link writer are also unfinished. The original NumPy helper
contracts remain unchanged; a byte-identical copy is the v6 contracts module.
Passing 51 helper tests must not be called successful learned segmentation.

S610 read 32 real frames from four image-selected source clips and created raw
orthogonal image/C0 overlays. It produced **zero learned masks**. No physical
recipe was selected from the two registered 6/9 µm candidates. S620 B0/M0 and
S630 U0/U1 did not run. Mask containment/recall, real segmentation seam/merge
checks, mask storage/time and complete image-to-mask-to-graph parity remain
unmeasured. The CSV includes blank-score blocked rows; it never fills them with C0.

The initial fresh P0 attempt used a heatmap-sharing helper requiring an
`input_hashes` manifest absent from the inherited package. The failed log remains
in `fresh_point.log` and the original RAM scratch. Only the v6 caller was changed
to use independent regeneration, then both clips passed with exact inputs.

## Preservation

No v5 processes were running at preflight, although the stale supervisor JSON
said running. The actual latest v5 snapshot had 12/18 scored configurations and
six/eight native fits; its selected incumbent was C0. No v5 job was stopped,
restarted or edited. Its in-progress results were not described as a final
negative study.

The uncommitted `EXPERIMENTS.md` addition pointing to `/root/FOCUS-3D` is left
unchanged and excluded from the v6 commit. Unrelated user files are excluded.
`preservation.json` records prior code/result/critical artifact hashes;
`preservation_check.json` verifies them at delivery. Only an explicit v6
code/config/report allowlist is committed to this same branch.
