# Execute the single v6 plan

Read root AGENTS.md, the latest local v5 continuation/results, and
`handover/segmentation-tracking-v6/EXPERIMENTS.md`.
Then implement and execute that plan on this branch using the existing RTX 4090.

Focus on integrating a learned instance segmenter and Ultrack, retaining masks
and bounding boxes through tracking. Use FOCUS-3D if its authorized assets are
available locally; Cellpose is the one fallback. Do not expand into another
model survey, external-data campaign, or large training/threshold sweep.

Use the existing repository structure and local assets. No ZIPs, patches,
activation steps, automatic external downloads, or additional handover branches.
Record unavailable dependencies and continue independent runnable stages.
Do not pretend a classical watershed is a successful learned-tool integration.

Preserve the C0 incumbent, prior results, active jobs and unrelated user changes.
Implement the adapters, run the bounded comparisons and real-image checks, and
write measured results plus CONTINUATION.md. Commit and push only sanitized v6
code/configs/reports to this branch. Do not merge or submit to Kaggle.
Do not return another plan instead of executing this one.
