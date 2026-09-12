# Local Codex / GPT 6 Pro execution

Implement and execute M600-M690 from `handover/instance-tracking-v6/EXPERIMENTS.md`.
Do not return another plan. This user requests a change to segmentation-driven
tracking using one or more learned detectors/segmenters and Ultrack.

Read root AGENTS.md, relevant competition skills, this entire handover, v5
CONTINUATION.md and its current result tables, and the external-data guide. Read
local official microscopy/marker and metric references. Verify actual local state:
the author inspected v5 commit 0a3105a8f9d1b0954707675b42b9868832df4ac1, which was
an in-progress checkpoint, not a completed study. Snapshot any newer measured
receipts without mutating them. Do not restart v5 training, queues or expired run
windows, and do not claim unfinished v5 outcomes. Do not kill unrelated jobs.
If old processes use the working tree, use a separate v6 worktree and coordinate
GPU use; do not change files underneath running processes.

Preserve C0 = 0.934802374260586 unless an independently documented newer incumbent
has actually been validated locally; record such baseline drift before comparisons.
The goal is >=0.95 on the full official 199-clip local evaluation. Smaller repeatable
improvements are retained as below-target; do not promote a regression.

Run learned segmentation early, before extensive custom tracking code. Attempt
FOCUS-3D with authorized local weights/access and independently run Cellpose-SAM
in 3D. FOCUS currently publishes general, nuclei and membrane checkpoints on a
gated Hugging Face repository. Do not submit personal information, accept new
access terms on the user's behalf, upload microscopy, or use the hosted demo.
If FOCUS access is absent, record `blocked_access` and execute the open alternative
and Ultrack without waiting. The FOCUS arm must remain ready to consume an explicit
checkpoint path. No fabricated weights or automatic tiny random-model substitute.

Keep actual masks/boxes through temporal scoring. A centroid plus synthetic sphere,
a seed-only watershed relabeled as a learned mask, or a mask immediately discarded
after extracting its center is NOT completion of the requested direction. V5's
watershed/HOCT results are controls, not evidence that FOCUS/Cellpose masks have
already been tested. Store native-grid transforms and morphology in both explicit
voxel and physical units. Reproduce upstream feature conventions before using
pretrained HOCT; v5's physical-versus-voxel mismatch must not recur.

Implement source-independent model adapters, mask-to-C0 association, full-mask
link/union features, standard Ultrack integration and its graph-to-CSV export.
Compare point/box/mask feature ablations with identical candidate sets and solver;
then compare fixed masks with Ultrack's competing segmentation hierarchy. Run a
standalone mask-based tracker as well as a C0-preserving hybrid. Protect against
mask flicker, duplicate representations, false splits and temporal crop boundaries.

Do not spend the iteration on another auxiliary fork classifier or a larger
external-data sweep. Optional training means genuine segmentation adaptation or
speed distillation from validated masks, not treating sparse centers as dense masks.
Run the primary segmentation/tracking comparisons even when optional labels,
FOCUS access, StarDist, or fine-tuning are unavailable.

All learned supervision/calibration uses only the declared source embryo. Both
embryos are reused and upstream exposure exists: label results operational
exploratory, never untouched OOF. No target GT in model inputs, candidate generation,
flow, segmentation, hypothesis protection or Ultrack optimizer constraints.
Use strict full-sample official scoring and independent fresh image-to-mask-to-graph
inference. Do not count missing labels as background or treat segmentation stability
as ground-truth IoU. Prior adoption values and all scientific deviations remain
visible. No automatic Kaggle submission or PR merge.

The existing RTX 4090 is the execution device. Use separate optional package
runtimes; never upgrade the user's shared CUDA stack or mutate raw inputs. Check
actual disk/RAM/free GPU first, use per-clip compressed masks/cropped support caches,
run a single GPU-heavy process by default, and checkpoint atomically. Bound initial
segmentation+training GPU use at 48 measured device-hours, at most 24 complete
variants, with >=8 GiB disk reserve and <=20 GiB GPU/24 GiB host-RSS soft limits
lowered to actual availability. Do not confuse caps with a promised runtime.

Write measured reports, a local offline mask/box/point comparison viewer, exact
score rows, runtime accounting, prediction/mask manifests, a selected inference
package and CONTINUATION.md. Commit and push authored code plus sanitized summaries
to `handover/instance-tracking-v6`; raw images, masks, weights, detailed labels and
credentials stay outside Git. Start with the supplied tests/preflight, then do
actual integration and data experiments. Finish with an honest result even when
the new approach does not beat C0; failed optional arms do not block valid outputs.
