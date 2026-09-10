# Continue from the completed v4 experiments

W400–W490 are complete. The measured implementation and sanitized results were
committed in `0fb190d0823f6ee9e7be45f3cb693b37967b8d36` on
`handover/multidata-training-v4`. Start here when reviewing this branch to propose
the next study. The original [CODEX_PROMPT.md](CODEX_PROMPT.md) is the completed
execution assignment; it is not an instruction to restart v4.

The selected model remains **v3 `A_residual_m3.0` / C0**, with pooled local score
**0.934802374260586**. No external-data candidate passed the complete promotion
rule. Positive pooled exploratory rows must not be presented as adopted gains.

## Read order and evidence map

1. Read [AGENTS.md](../../AGENTS.md), the
   [measured report](../../results/multidata-training-v4/final_report.md), and the
   [execution handover](../../results/multidata-training-v4/NEXT_AGENT.md).
   The [offline dashboard](../../results/multidata-training-v4/dashboard.html)
   presents all variants, embryo comparisons, training curves and diagnostics.
2. Review [ablation scores](../../results/multidata-training-v4/ablation_scores.csv)
   and [all official clip scores](../../results/multidata-training-v4/score_rows.csv).
   These cover 21 complete variants × 199 clips = 4,179 fresh official scores.
   [Arm histories](../../results/multidata-training-v4/arm_training_histories.csv),
   [training summary](../../results/multidata-training-v4/training_summary.csv),
   [learning curves](../../results/multidata-training-v4/learning_curves.csv), and
   [secondary verification](../../results/multidata-training-v4/secondary_verification.json)
   document actual updates, matched controls and replication.
3. Inspect [candidate attrition](../../results/multidata-training-v4/candidate_gt_coverage.csv),
   [detector attribution](../../results/multidata-training-v4/detector_attribution.csv),
   [transfer diagnostics](../../results/multidata-training-v4/transfer_diagnostics.csv),
   and [rendered holdout diagnostics](../../results/multidata-training-v4/rendered_holdout_diagnostics.csv).
   Distinguish these post-outcome diagnostics from prespecified decision rules.
4. Read [dataset use](../../results/multidata-training-v4/dataset_use.csv),
   [source coverage](../../results/multidata-training-v4/source_coverage.csv),
   [sparse-label correction](../../results/multidata-training-v4/sparse_edge_mask_correction.json),
   [failed-fit accounting](../../results/multidata-training-v4/failed_fit_accounting.json),
   and the [external-data guide](../../docs/external-data-guide/README.md).
   The guide records an earlier preparation snapshot; v4 receipts record the
   subsequent training selection and measurements.
5. Read the [implementation and commands](../../docs/multidata-training-v4.md)
   and inspect [maintained code](../../tools/multidata_training_v4/).
   The original [experiment specification](EXPERIMENTS.md) and
   [implementation specification](IMPLEMENTATION.md) explain the intended
   controls; [execution corrections](../../results/multidata-training-v4/implementation_corrections.json)
   and the measured report explain the delivered scope.
6. Review the preserved [v3 report](../../results/strong-tracker-v3/final_report.md)
   before proposing changes to the incumbent. Its established association
   evidence and division protections matter to the v4 failures.

The CSVs are tracked even though generic CSV output is ignored by repository
rules. Use `git ls-files results/multidata-training-v4` to enumerate the complete
committed result set. [artifact_manifest.json](../../results/multidata-training-v4/artifact_manifest.json)
pins 115 sanitized artifacts and 45 implementation files. Preserve those bytes;
this continuation document is separate from that measured snapshot.

## Findings a new proposal must account for

Actual external pretraining, matched real-only controls, both sparse Biohub
adaptation directions, second-seed controls and conditional Zoo rendering ran:
804,000 retained optimizer updates across 69 fits. The 4.257 summed training-loop
hours exclude cache preparation and failed/superseded fits; they are not hours
of continuously saturated GPU utilization.

- Primary C4 gained **+0.000146776125** pooled but lost **0.000116941379** on
  6bba. C4's second seed lost **0.000260245840** pooled. Full C1 controls preserved
  the incumbent in both seeds. External benefit was not replicated.
- C4 with descriptive margin 6 reached **0.9351783702566538**, the highest pooled
  score, but still regressed on 6bba. C5 also failed that embryo. Selecting a
  pooled maximum or another lucky seed would not satisfy the v4 adoption rule.
- C7 learned the rendered Zoo distribution: rendered-test correct-pair recall
  rose to approximately 0.65–0.66, with increased wrong-pair predictions. Its
  Biohub pooled delta was **−0.000007875223**. Renderer accuracy did not establish
  real-image transfer.
- Of 151 observed GT forks, C4 had 104 with both daughters in its six-candidate,
  exact-next-frame proposal set, 36 after its gate, one after its margin, and
  zero newly accepted exact-ID edits after protection/conflicts. This strict
  diagnostic uses different matching from official temporal-tolerance division
  TP; it must not replace the official counts.
- Rebuilt C4 associations beat rebuilt C1 by **+0.014698468708**, yet remained
  **0.012543955843 below v3**. Detector variants also regressed. The implemented
  detector refines incumbent centers; v4 did not test a replacement dense
  detector or new-peak recall. Keep coordinate and association effects separate.

A useful next plan should identify which measured bottleneck it addresses,
state a falsifiable hypothesis, and specify controls, validation, compute budget
and a stop/adoption rule. Open questions include proposal timing and coverage,
calibration under domain shift, weak sparse-event supervision, seed instability,
and preserving incumbent continuation evidence. These are research questions,
not established improvements or instructions to run another threshold sweep.

The v4 selection required a positive pooled delta, no embryo regression beyond
numerical tolerance, and a qualifying same-arm second seed. Shared generator
validation temperatures mean “real-only” refers to neural weight training.
The two repeatedly used Biohub embryos, 44b6-calibrated simulator, inherited
public/teacher exposure and same-acquisition Zoo time blocks make these
operational exploratory results. A new plan must address those limitations
explicitly rather than describe the existing splits as independent biology.
Sparse unknowns stay unknown; quiet-chain negatives retain a weak-supervision
assumption. Do not restore the unsupported edge negatives corrected before
the reported comparisons.

## Local dependencies and restoration boundary

A fresh clone is sufficient to review the code, measured evidence and propose
a new plan. It does not restore microscopy, weights, detailed graphs or caches.
On the execution machine, preserve these locations before replacing the container:

- `/kaggle/working/cell-tracking/multidata-training-v4/`: models, caches,
  predictions, detailed evaluation, logs, failed fits, locks and inference package.
- `/kaggle/working/cell-tracking/annotation-selection-v1/`,
  `/kaggle/working/cell-tracking/strong-tracker-v2/`, and
  `/kaggle/working/cell-tracking/strong-tracker-v3/`: sealed upstream artifacts.
- `/kaggle/input/competitions/biohub-cell-tracking-during-development/`: original
  competition inputs. Repo `data` is an alias.
- Repo `work/biohub-forum-archive/` and `work/biohub-data-guide/`: external
  originals, prepared labels and source receipts. See the committed
  [access instructions](../../docs/external-data-guide/ACCESS.md) for restoration.
- Repo `work/annotation-selection-v1/official/`: pinned local metric checkout,
  revision `075fc5f5a52d11077f9dc2b074644618f26939e2`.
- `/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt`:
  original DeepCenter checkpoint required by the base inference pipeline.

The tested CUDA interpreter is
`/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python` (PyTorch
2.8.0+cu128); use GPU 0, RTX 4090. Full measured versions are in
[runtime_versions.json](../../results/multidata-training-v4/runtime_versions.json).
The separate `/root/.conda/envs/cell-tracking/bin/python` supplies the browser
validation runtime. Set `PYTHONNOUSERSITE=1`; do not install into system Python.
The wrapper already configures CUDA visibility, deterministic CUDA workspace,
thread counts and the local metric import path.

The transfer archive is
`/kaggle/working/cell-tracking/multidata-training-v4/selected_inference_package.zip`
(86,081,216 bytes), SHA-256
`55889f0628251b799b8d1c09796e5da9b7a8be69fec8ca41fbcf9cbecb68b762`.
It bundles new heads but references existing base dependencies: the explicit v1
root supplies `public_harmonic_full/harmonic_isolated.py`, pinned `tracking_repo`
and primary/secondary weights; v2 supplies frozen `models/{source}/E_hgb.joblib`.
The archive alone is insufficient without those dependencies and DeepCenter.
Exact hashes are in its `base/manifest.json` and the committed
[package manifest](../../results/multidata-training-v4/inference_package_manifest.json).

Use the package's `run.sh` with explicit `--python`, `--images`, `--output`,
`--v1`, `--v2` and `--source-model 44b6|6bba` arguments. The image argument is a
directory containing Zarr clips; output must be a new directory. Source-model
names the training source: the scored transfer used 6bba for target 44b6 and
44b6 for target 6bba. The default selected variant is C0;
`--disable-new-heads` explicitly preserves v3 and `--variant C4` exercises the
external candidate. Prediction needs no external training datasets or labels.

Delivery passed 53 contract tests, 13 fresh-image executions, exact C4 head-path
parity on all 199 clips, package hashes/ZIP CRC and offline dashboard checks.
The process-start Python read/socket guards are dependency-use checks; Linux
namespace isolation was unavailable. Local 4090 validation does not certify
Kaggle runtime, and no Kaggle submission was made.

## Preserve the completed study

All required and scheduled v4 work has completion receipts. The recorded
seven-hour window was 2026-09-09 23:09:09 through 2026-09-10 06:09:09 UTC; its
queues are historical execution machinery. Review existing results before
starting a new experiment, and use a new output namespace for scientific changes.
`V4_OUTPUT` relocation expects the preserved v1–v3 roots as sibling directories.
Do not reset old queue deadlines or overwrite checkpoints, prediction locks,
reports, sealed annotation outputs or unrelated working-tree edits.

For a read-only code check from the repository root:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m unittest discover -s tests -p 'test_multidata_v4.py'
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m unittest discover -s handover/multidata-training-v4 -p 'test_contracts.py'
```

Git contains the sanitized evidence needed for review, not a backup of the
container. Keep credentials, raw data, model weights, submissions and detailed
annotation identities outside Git. Existing remote-restoration and external-guide
presentation files in the working tree belong to separate work and are not
prerequisites for this handover.
