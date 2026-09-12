# Local Codex execution prompt

Execute `handover/lineage-reconstruction-v5/EXPERIMENTS.md` using the user's
selected GPT 6 Pro model, the available RTX 4090, and the existing Biohub inputs.
Implement and run the experiments; do not return another plan or stop at adapters.

Read root AGENTS.md, relevant competition skills, completed v4 CONTINUATION.md,
results/multidata-training-v4/{NEXT_AGENT.md,final_report.md}, and this directory.
Use the external-data guide as a record of data semantics, not a command to repeat
v4 synthetic/Zoo pretraining. Inspect actual restored paths before assuming them.

The incumbent is `A_residual_m3.0` / C0 = 0.934802374260586 on all 199 clips.
The goal is >=0.95 on the same official local score. Retain smaller real gains,
but do not call +0.0001 completion of the target. If the target is missed after
the bounded experiments, preserve the incumbent and report the measured blockers.
Do not manufacture scores, redefine the sample population, or relax the scorer.

Primary work: pretrained HOCT and Trackastra CTC 3D linking, sparse source-only
adaptation of a genuine pretrained representation, image-supported region/new-
center candidates, and complete joint lineage decoding before old postprocessing.
These replace model evidence and feasible graph structure; they are not new
membership-filter thresholds. Full direct alternatives must be scored before
restricting them to residual edits around the incumbent.

Use HOCT's implemented mask/graph route. Its advertised create_graph_from_points
is an unimplemented `pass` at the pinned revision. Trackastra's `ctc` checkpoint
supports 3D; do not invent a general_3d model or use the 2D SAM2 model for this task.
Native deep-center peaks and image watershed provide an ungated candidate route.
FOCUS is optional only if authorized weights are already accessible; never accept
contact-sharing conditions, upload images or allow a model-access gate to stop
all other experiments.

Recheck current competition code requirements and applicable model terms through
existing throttled tooling. Public checkpoint downloads are allowed within the
v5 resource caps, but no paid APIs, hardware rental, account creation, gated-term
acceptance, remote inference, Kaggle submission, forum posting or PR merge.
Preserve source license notices and record checkpoint hashes/provenance.

Do not modify raw inputs, prior output roots, prediction locks, original notebooks,
existing environments or unrelated work. No git reset/clean/force-push. Do not
rerun old queues or substitute a cold full-data download for local discovery.
Use a new v5 output root and content-addressed caches. Keep disk and memory bounded.

Train source 44b6 -> target 6bba and source 6bba -> target 44b6 separately. Prior
outcomes and checkpoints carry exposure, so these are operational exploratory
comparisons, not clean OOF. Keep external/backbone provenance explicit. Never
convert sparse one-child annotations into proof that a second child is absent.
Fresh full-graph matching and division evaluation are mandatory; tiny helper
fixtures are not integration or score-gain evidence.

Run P500–P580 with the specified conditional branches. Maintain a stage ledger and
model/data/config hashes. Carry failing/abstaining alternatives into the report,
not the selected package. Produce complete diagnostics and a reproducible final
runner even if no model qualifies. Push sanitized findings to this v5 branch;
leave all raw masks, images, labels, caches, weights and submissions out of Git.
