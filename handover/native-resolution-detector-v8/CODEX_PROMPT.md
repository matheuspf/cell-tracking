# Execute native-resolution detector v8

Read root AGENTS.md, the competition-data skill, this handover's PLAN.md and
study.json. Implement and execute D800-D890 locally using the existing Biohub data
and RTX 4090. This is the active detector task; older root segmentation handovers
and unfinished v5/v7 plans are context, not concurrent execution instructions.

Train a genuinely new whole-image center detector at native 64x256x256 ZYX.
Do not satisfy this by refining only the old detector's points, interpolating a
64-cubed image back up, or unfreezing an association backbone with its detector
head still frozen. Native full-volume training is the first memory pilot; native
XY tiles with halos are allowed if needed, but inference must cover every raw
voxel field. Record which training mode actually ran.

Audit the exact original loader, peak extraction, target quantization and detector
checkpoint provenance. Prior reports make exposure claims; preserve them as evidence
but resolve the specific loaded artifacts rather than declaring every model clean
or contaminated by association. Unknown stays unknown. The clean primary starts
from random weights, with no uncertain incumbent, FOCUS or historical teacher
in its weights, pseudo-labels, normalization, input features or model selection.

Use both embryo directions with complete dependency-aware isolation. Fit and tune
on the source only; freeze BOTH directions before outer scoring. Overlap evidence
means random frame/clip folds are not acceptable validation. If honest source inner
groups cannot be certified, use the preregistered fallback and label every outer
sweep exploratory. Repeated research on these embryos cannot become an untouched
biological test set. Do not let an old full-tracker score select a clean detector.

Implement masked sparse heatmap/offset training, a source-only EMA consistency
regime, a deliberate missing-as-negative diagnostic control, and the optional
external-object/reference-point regime when valid local assets exist. A cell not
annotated is not background. Keep dense pseudo object centers separate from the
reference points supplied by humans; no duplicate supervision of one object at
two different positions. Do not estimate a PU class prior from annotation fraction.

Use PLAN.md's ordered, bounded experiments and source-only parameter iterations.
Run real images, actual gradients and complete detector comparisons. The default
native model must run even if third-party software is unavailable. Do not repeat
v4's tiny patch-offset study or spend this iteration installing a segmentation
framework. No new datasets, paid services, license acceptance, image upload, or
automatic checkpoint/package downloads. Local assets and installed libraries are
sufficient for mandatory regimes; report optional blockers accurately.

Build maintained code under tools/native_detector_v8/, one wrapper under
scripts/run_native_detector_v8.sh, and tests in tests/test_native_detector_v8.py.
Use ignored work/native-resolution-detector-v8 and the matching /kaggle/working
output namespace. Read previous artifacts only, preserve user modifications and
active jobs, stream data, reserve disk, and write resumable per-fit receipts.
Do not move the whole data collection or allocate all-frame dense teacher caches.

Validate detection independently at several physical tolerances and fixed candidate
budgets; unknown extra detections are not automatically FPs. Also measure integer
export and fixed-linker end-to-end effects. Recompute features/links at changed
centers rather than attaching stale node-ID scores. Preserve C0/P0 fallback and
clearly distinguish label-isolated detector results from tracker uncertainty.

Finish with trained checkpoints and detector-only inference, measured report and
small offline dashboard, source/provenance manifests, all-199-clip detection tables,
selected operational graph comparisons where runnable, and CONTINUATION.md in this
same handover directory. Keep one authoritative PLAN.md; revisions are logged
there and in study.json, not new parallel handovers. Commit/push only explicit
sanitized code/config/results to this branch. Do not merge or submit to Kaggle.
