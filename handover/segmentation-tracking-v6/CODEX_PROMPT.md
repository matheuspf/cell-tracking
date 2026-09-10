# Execute segmentation-first tracking v6

Implement and execute S600-S680 from this handover on the existing machine with
one RTX 4090. The goal is a materially better image-region tracker, targeting
>=0.95 official local score, not another small point-based fork-head sweep.

Read root AGENTS.md, relevant competition skills, this handover, and
handover/image-native-tracking-v5/CONTINUATION.md first. Then inspect the current
local v5 status: the remote parent reviewed here is still in progress. Do not
invent outcomes for pending N1/N2, replica, or fresh-inference experiments. Do not
restart v4/v5, reset old deadlines, terminate unrelated jobs, overwrite locks,
change the active v5 source, or wait for all old experiments merely to begin v6.
Use a separate worktree if old jobs depend on the current checkout; prepare v6
without GPU contention and acquire resources only when safely available.

Freeze C0=0.934802374260586 as the required historical comparison. If a later
fully validated v5 winner is actually present, also lock it before v6 outcomes
and report both deltas. Recompute this baseline with the official 199-clip scorer.
Preserve the original full primary/secondary/eight-view ensemble; a single-primary
control must not be mislabeled C0.

Primary work: independent instance segmenters -> persistent mask/box objects ->
mask/appearance/motion associations -> Ultrack hierarchy/lineage selection ->
Kaggle point/edge export. Do not claim completion by extracting a few watershed
statistics, throwing away masks, and rerunning the previous point pipeline.

Attempt FOCUS-3D through its documented headless infer_volume route with an
already authorized local checkpoint/account. Do not accept contact-sharing terms,
license contracts, paid services, or upload microscopy to a hosted demo. If weights
are inaccessible, record the exact blocker and continue with Cellpose-SAM and
StarDist3D where available. At least one genuinely learned segmentation backend
must run. Classic foreground/watershed is a control, not a substitute for that
requirement. Inspect model-specific licenses and current competition rules.

Use the real input lattice and anisotropy. Preserve raw masks, grid transforms,
physical and model-specific voxel features separately. The v5 HOCT unit audit is
mandatory context: physical descriptors are not interchangeable with a checkpoint's
expected voxel-valued inputs. Prove adapter parity before using its predictions.

Keep sparse unknowns unknown. Point annotations are not segmentation masks. Do
not create dense background labels from unannotated regions, call rendered Zoo
patches real microscopy, or reuse synthetic clone IDs as persistent cell IDs.
Do not learn thresholds from the target embryo, filenames, GT counts, matching
artifacts, or post-outcome optical review. All local outcomes remain exploratory
because the two embryos and upstream checkpoints were repeatedly exposed.

Implement tested stage modules under tools/segmentation_tracking_v6 and a clean
wrapper. Use /kaggle/working/cell-tracking/segmentation-tracking-v6 and ignored
work/segmentation-tracking-v6 only. Keep raw images, masks, checkpoints, full GT
matches and prepared external arrays out of Git. Isolate incompatible FOCUS,
Cellpose, StarDist and Ultrack environments; do not upgrade shared study runtimes.

Prioritize actual mask generation, the representation ablation and full region
tracking. Bound screening and integration debugging; do not spend the study on
another framework or data-download survey. Run both pure Ultrack and mask/native
hybrid comparisons so a poor new linker does not hide useful segmentation evidence.
Keep no-op fallback and separate candidate-generation, linking and division gains.

Every complete variant needs all 199 clips, fresh official matching and divisions,
strict output validity, and exact aggregate weighting. No partial folder intersection
may be called full evaluation. Learned finalists need same-recipe replication;
freeze a single primary export rather than picking whichever seed looks best.

Finish with final_report.md, dashboard.html, all comparison CSVs, measured resources,
model/source manifests, an image-to-graph inference package, and CONTINUATION.md.
Update handover status with a pointer to actual result status. Run fresh-image
pilots with annotations, old predicted graphs, and network access unavailable.
Record genuine failure or incomplete arms rather than fabricating masks or gains.
Commit/push only an explicit sanitized v6 allowlist to this branch. Do not merge,
submit to Kaggle, publish a notebook/forum post, rent hardware, or start paid APIs.
