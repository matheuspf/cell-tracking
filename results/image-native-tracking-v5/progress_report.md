# Image-native tracking v5 — execution progress

Updated 2026-09-10T14:02:40.818865+00:00. 1 of 18 registered complete configurations are scored.
C0 is freshly reproduced at 0.934802374260586; the target is 0.95 (+0.015197625739414).
Current eligible export: C0 at 0.934802374260586. Selection remains provisional while execution continues.

The offline [dashboard](dashboard.html) contains the official complete-clip comparisons, source learning curves, candidate coverage and replication gate.
Only complete 199-clip variants enter the comparison. Oracles are separately identified as truth-assisted feasibility diagnostics.

The native tracker is the installed 2,076,706-parameter U-Net/transformer, with a 1,496,320-parameter temporal 3D encoder. N1 fits its association module for 8,000 updates per source; N2 warm-starts N1 and updates the actual encoder for 12,000 more, at a tenfold smaller learning rate. Both recipes repeat with seed 314159. The detector output layer remains frozen; N2 is an association-representation experiment at fixed coordinates, followed by the proposal factorial.

HOCT runs the pinned official 6,252,593-parameter general_v1 JIT with 19 genuine region/position features and 288-dimensional edge embeddings. H1 fits its linear edge probe on supported source transitions. H2 fits one source-only residual calibration with the unchanged native evidence. Hard ILP consistency is disabled so missing C0 links do not become negatives. The ctc_v0 model is a source-only diagnostic within the 18-complete-configuration cap.

P1 discovers full-field native peaks with real image contrast, assigns new IDs, derives image-supported watershed morphology, and rebuilds candidate features at the new coordinates. PDC uses continuous frozen DeepCenter confirmation. The image ablation removes proposal confidence. The rolling five-frame MILP compares births, continuation, bifurcation and incumbent explanations under ownership and one-cell/two-cell exclusion constraints. Timing aliases on predicted paths share a maximum complete-explanation choice. C0 fork predecessor/daughter/grandchild edges are protected in operational inference. Oracles relax protection explicitly.

Native comparison tensors use the primary checkpoint without the incumbent's secondary model/eight-view harmonic ensemble. Their full-frame two-frame inputs use the exact installed downsampling, quantiles and positional/indexing conventions. This intentional change is isolated by N0. The complete inherited C0 path and pre-ILP arrays were independently reproduced on two full density-selected clips.

Both direct adaptation directions use only their own source labels and source calibration. All source transitions with represented endpoints are eligible; missing parents and possible unannotated second daughters remain censored. Public checkpoints, C0 teachers and repeated embryo use prevent an independent biological-generalization claim. Seed replication measures training sensitivity, not embryo independence.

The server reboot interrupted execution after the native log reached update 2,941. Hash checks recovered 122 image shards, 120 HOCT shards, all C0 results and optimizer/RNG state at update 2,000. The last 941 updates were repeated. Prior critical hashes remained unchanged. The crash cause is unavailable from container kernel logs. Resumption uses tmux, two bounded GPU lanes and shared CPU-pool locking; native resume snapshots are now every 250 updates.

Detailed GT identities, optical-review images, raw data, checkpoints and submissions remain local. No Kaggle submission or notebook publication is performed.
