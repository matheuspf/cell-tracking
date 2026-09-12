# Implementation contracts

## Additive package and explicit paths

Implement `tools/lineage_reconstruction_v5/` and tests under
`tests/lineage_reconstruction_v5/`. Provide a new wrapper and CLI with explicit
`--data-root`, `--study-parent`, `--output`, `--model-root`, `--source-model` and
interpreter selection. Do not alias another package's hard-coded OUT global or
replay v4 queue machinery. Proposed stage commands do not exist in this handover;
build them before documenting them as executable.

Reuse the actual v3/v4 packaged image runner, metric adapter, graph readers,
provenance maps and serialization validators where contracts agree. In particular,
`tools/multidata_training_v4/models.py::Detector` is an offset/query model, not
full-image detector inference. Use the original full-frame DeepCenter source and
weights for new peaks. Locate the patched native predictor through the existing
package manifest; do not assume unmodified upstream scripts reproduce its output.

## Modules to implement

- `evidence.py`: native heatmaps and pre-threshold edge logits, deterministic
  window aggregation, physical-grid mapping, checkpoint/source hashes.
- `regions.py`: image-conditioned foreground/seeded watershed, new maxima and
  competing split/merge supports, mask-ID/candidate-ID joins and failed seeds.
- `pretrained.py`: isolated HOCT/Trackastra adapters, model cache, licensing and
  numerical feature/API parity receipts; no implicit model downloads in inference.
- `supervision.py`: source-only one-to-one reference matching, supported incoming
  supervision, sparse unknown masks, group weighting and provenance. Never
  imported by inference.
- `adapt.py`: head and genuine backbone adaptation with parameter/gradient receipts,
  deterministic source exposure, checkpoint resume and fixed source model choice.
- `lineage.py`: native-only/new-only/combined full candidate optimization, explicit
  lineage states and segmentation conflicts, deterministic overlap reconciliation.
- `run.py`: image-only cold inference, canonical CSV export and baseline fallback.
- `evaluate.py` and `report.py`: fresh official counts, strict complete-set
  aggregation, source/target results, runtime and offline report/dashboard.

## Instance supports and physical coordinates

The native measured data had TZYX=(100,64,256,256), spacing ZYX=(1.625,.40625,.40625)
um. Re-read each metadata file. Coordinate transforms must include axes, voxel
spacing, crop origin, strides and interpolation conventions. Do not interpret a
voxel distance as a micrometre distance. A network input resize changes coordinates
and shape descriptors together; adapting coordinates alone is not unit parity.

A0 regions are proxy shapes from image watershed, not segmentation ground truth.
Use actual signal to set foreground; do not fabricate equal-size balls as the
primary input to models trained on region shape. A ball/Voronoi support can be a
named control to diagnose how strongly region quality matters. Log empty seeds,
center collisions, truncated supports and feature nonfinites. Fixed-node evaluation
must retain exactly the incumbent points; a failed support can use an explicit
fallback representation or reject that fixed-node variant, not silently delete it.

Both models receive actual supported region features. Audit upstream REGIONPROPS,
coordinate normalization, missing-feature handling and physical scaling. Where
an API lacks anisotropy, use one documented common grid for images AND masks and
invert exactly at output. Never use the 2D model because its API is easier.

Mask labels may repeat across frames; candidate IDs must be stable per video.
Retain `(frame,mask_label)->candidate_id` mappings and model-output mappings even
when model APIs relabel tracks. Map edges through identities, not nearest GT.
Freeze exact candidate populations for fixed-node comparison; new masks/centers
belong to A1. Separate feature centroids from exported node centers in the receipt.

Do not save all raw 4D masks uncompressed across 199 clips. Stream native frames
and model windows, persist compact region features/points/scores, and retain a few
bounded audit overlays locally. uint32 masks for the entire collection would be
far larger than the existing available disk. Fail on budget overflow rather than
deleting prior artifacts. Use source-hash-aware on-demand reconstruction.

## Partial-label training, not false background supervision

Represent association targets as positive, supported negative or unknown.
For a matched target with a known annotated predecessor, alternate source
candidates can be contrasted against that predecessor, subject to ambiguity masks.
A matched parent with one recorded child does not certify nondivision or falsify
all other possible daughters. A parent with two established children under the
binary-tree contract can rule out a third. Unknown starts/ends of annotations
are not observed biological birth/death. Censored windows remain masked.

Keep all annotated positive alternatives inside training only if they exist in
the prediction-derived candidate set. No GT-centroid injection into deployed
populations. Record missing-positive coverage rather than making up a feature
row. One-to-one matching and ambiguity masks must be computed separately for
each new candidate universe. Do not reuse labels attached to old IDs/coordinates.

Prefer a masked incoming categorical loss with an explicit birth/null option
where its label is supported, and separate partial outdegree/fork supervision.
Do not row-normalize over two true daughters as if only one can be correct.
Use source fine-tuning to learn competition-relevant transitions; unknown regions
can receive soft frozen-model consistency only as a clearly marked regularizer.
Hard current-ILP labels on unknown edges are OFF in the main adapted arm.

Third-party training code often assumes complete CTC truth. Integrating sparse
Biohub means implementing masks before the loss, not merely making CTC-shaped
files. Test that logits in unknown supervised slots have zero direct loss gradient.
Attention/context gradients may still flow through unknown candidates; distinguish
that from giving them hard labels. Check birth, division and terminal-frame cases.

The reference `sparse_edge_targets` is intentionally conservative. It illustrates
the semantics, not the full official matching procedure or a dense training loop.

## Model adaptation and controlled evidence fusion

Record starting checkpoints and exact trainable parameter names/counts. Compare
zero-shot, adapted head and adapted transformer when supported. Log nonzero
finite gradients and checkpoint differences on the backbone. A scripted model
that cannot train may remain a frozen feature extractor; do not report its head
probe as end-to-end image training. Do not silently retrain a tiny scratch MLP
and call it HOCT/Trackastra adaptation.

Model inputs never include annotated IDs, ground-truth degrees, match distances,
annotation masks, count estimates, oracle actions or GT-bearing caches. No target
labels may enter direct fits, source calibration or teacher selection. Existing
incumbent public exposure remains disclosed; it does not justify new leakage.

A common score is needed before combining native and new logits. Fit temperature/
scale and optional bias using supported source data and the fixed recipe; state
that selective support is not calibration on all real cells. Do not multiply raw
probabilities from different normalization domains. Always retain a new-model-only
and native-only control. Different windows of the same underlying edge are
correlated views; normalize their evidence and deduplicate exact repeated views.

## Joint lineage and competing segmentation states

A feasible selected graph obeys indegree <=1, outdegree <=2 and forward consecutive
frames. Node selection is explicit for A1. Overlapping competing region hypotheses
cannot all be selected; impose exclusion constraints using predicted supports.
Birth/death, continuation and division costs must share a consistent sign/scale.
Tiny enumeration checks must show both true division and correct continuation can
win. A fixed 1.2 penalty against bounded [0,1] edge reward and free births is not
acceptable without a separate event term that actually changes that dominance.

For large clips, solve time windows with sufficient overlap and carry ownership
constraints across seams. Candidate selection and edge/link fusion must not create
extra confidence just because the same event occurs in multiple windows. A
model may use longer temporal context, but final edges still connect t to t+1.
Do not interpolate blind bridging nodes to legalize skip links; re-detect missing
frames from images or abstain with a recorded fallback.

If inference is an edit-based hybrid, evaluate entire fork timing/path neighborhoods.
The v3 audit showed that an edge globally counted as FP may still supply valid
local division evidence. Never prune a daughter path based solely on global edge
FP flags or short-track heuristics. Full replacements are scored as full graphs.

## Required tests and acceptance receipts

The included handover tests are synthetic reference checks. Local Codex must add:
1. Upstream model smoke tests; verified checkpoint SHA; complete feature schemas;
   nonzero learned edge outputs; scripted/Python path parity where available.
2. Native-to-network-to-native roundtrips, anisotropy, crop offsets, mask relabeling,
   duplicate/empty seeds, center-feature versus output-center policies and dense
   new-peak scanning (not query-only refinement).
3. Sparse positive/two-child/single-child/unknown/birth/death gradients, dense-source
   controls, train/target path guards, group sampling and reproducible checkpoint
   restart. Label access is forbidden before inference model/input loading.
4. Full-lineage fixture enumeration: one true division, false fork, owner displacement,
   duplicate segmentation alternatives, skip-frame rejection, conflict-free window
   seams and permutation invariance. Enabling a head must change feasible decisions
   in a positive fixture; disabling it must recover the declared control.
5. Full 199-sample count/score identity with the current official scorer, no NaNs,
   no missing samples or changed GT/count metadata, plus cold renamed-image runs.
6. Runtime/VRAM/RAM/disk and model download receipts, inference sockets denied,
   raw-input/old-study hashes unchanged, archive CRC and staged-file privacy audit.

A model that outputs the incumbent byte-for-byte is valid fallback, not a measured
new capability. Report acceptance, candidate attrition and fallback counts, plus
actual GT-identity changes after fresh matching, not just counts of edited pairs.
