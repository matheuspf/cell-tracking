# Dataset decisions and supervision contracts

Use the existing `docs/external-data-guide/dataset_inventory.json` as the
source-of-record for paths, sizes and captured hashes. Do not redownload merely
because a fresh checkout lacks ignored arrays. Discover relocated roots before
restoring anything; never modify originals. W400 emits the actual runtime index.

## Selected roles

| Source | v4 role | What it must not supervise |
|---|---|---|
| Synthetic static, 1,539 volumes | Native-resolution center detector/encoder pretraining; dense synthetic heatmaps and subvoxel localization | Temporal edges, divisions, real annotation membership, instance masks |
| Synthetic six-frame, 2,174 clips | Dense synthetic edge/fork/no-fork training plus temporal image pretraining | Real division prevalence; nine observed frames; clone IDs as tracklets |
| Zoo zebrafish | First real geometry/trajectory pretraining source; weak link and fork supervision | Image appearance, native neural probabilities, assumed physical/time calibration |
| Zoo fly, ascidian, worm, beetle | Separately ablated source-balanced geometry transfer after use/provenance audit | Independent zebrafish validation; exhaustive biological negatives |
| Zoo mouse | Optional weak continuity/geometry pretraining only | A huge bank of confirmed nondivision events: this export records zero forks |
| Seven RIKEN archives | Inventory/label audit; optional spatial-layout stress tests only after use review | Known tracks/parenthood, paired images, dense masks or inferred GT from nearest neighbors |

Primary actual training must include synthetic images and eligible Zoo trajectories,
not only one type of data. If Zoo use cannot be cleared, complete the synthetic
training and comparison, record the exact blocker, and retain the model. Source
permission gates are per source; no new gated agreement is accepted automatically.
The synthetic release's CC0 declaration and Zebrahub organizer clearance are
recorded in the guide; check the archived statements and originating data terms.
The viewer's software license does not license every hosted dataset. RIKEN's
recorded CC BY-NC-SA terms remain a use-case question, not permission inferred
from the competition's general external-data allowance.

## Canonical representations

**Static.** Original `volume[64,256,256]`, fractional `centroids[N,3]`, spacing
ZYX=(1.625,.40625,.40625) um. Start at native resolution. The old loader's
pooled-static route is stride slicing, not mean pooling. Select resampling
explicitly and apply the identical transform to coordinates/spacing and the
matching real-data branch. Do not use 7 um as a heatmap radius by default.

**Sequences.** Original `volumes[6,64,64,64]`; source coordinates still live on
native [64,256,256]. Use prepared `zyx_pooled` or divide only Y/X by four;
image spacing is (1.625,1.625,1.625) um. Keep fractional coordinates until
submission serialization. Already-pooled images must never receive a second /4
image transform. Upsampling is an ablation, not recovery of unreleased detail.
`edges` index node rows. `source_clone_id` is a lineage shared by daughters,
not an identity feature. Use graph-derived tracklets. Time remains frame index;
no unsupported conversion to seconds. Final-frame nondivision targets are masked.

**Zoo.** Use exactly one enriched prepared graph per source acquisition. Base ZIP,
enriched store, rebuilt beetle ZIP and extra ascidian/worm CSVs are overlapping
representations. Keep a canonical representation ID; hashes and provenance groups
must prevent cross-partition alias leakage. Edges are direct adjacent-frame links
from checked parent metadata, not `tracks_to_tracks` reachability or the viewer's
colour-coded `divisions` array. Resolve row indices versus IDs before loading.

No downloaded Zoo images exist. Use a geometry tower with a feature set genuinely
available in both Zoo and Biohub: relative point displacements, neighborhood
configuration, temporal masks and candidate competition. Derive tracking evidence
from corrupted observations and a fixed label-blind proposal builder, not from
the clean target graph. The clean graph is a teacher target, never an input hint
such as true parent ID, clone ID, GT degree or future GT path.

Physical scales/cadence are unverified. Attempt the bounded source metadata audit
first. Sources with verified axes and internally consistent coordinates can enter
an explicitly unitless geometry pretext arm: per-axis robust centering/scaling,
relative distances/ranks and frame offsets. This is not calibrated physical motion
and may discard shape information. Use the same normalization on real candidates;
report it separately from the verified-um arm. Disable absolute um/second features
for unverified sources. Global scalar normalization alone cannot repair unknown
anisotropy. Never copy Biohub spacing into Zoo metadata.

**RIKEN.** Audit the other six HDF5/XML structures in bounded chunks, without
extracting all 51.8 GB at once. Numeric frame groups exclude metadata groups;
IDs may be frame-local measurements. A position feature or nearest neighbor is
not an explicit temporal label. If genuine correspondence fields are found,
verify uniqueness/time/semantics against the source before proposing a future
supervised arm; record that as a protocol amendment, not an automatic promotion.
Otherwise keep all seven out of supervised edge/event/image training. Missing
or broad FWHM/box measurements are not segmentation masks. Do not stall primary
training for this optional audit.

## Sampling and contamination

Synthetic examples are grouped by original example/generation identity. All crops,
noise variants, frame windows and rerenders of one example remain in one partition.
Use the prepared holdout only as a generator sanity holdout; derive validation/test
subpartitions without returning held-out examples to training. In the runtime
manifest record actual splits and seeds, not just requested proportions.

Each Zoo acquisition and its aliases is a source group. A contiguous held-out time
block with a feature-context purge can test within-acquisition transfer, but is not
an independent embryo. Complete lineage/overlap groups should remain together
where feasible. If the graph forms a giant group, use leave-one-acquisition/species
out for the relevant diagnostic instead of claiming independent random windows.
Measure zebrafish-only versus additional-species transfer; six species are not six
replicates of the target domain. Do not train on all Zoo and then advertise its
windows as independent test examples.

The published simulator used 44b6 density/tissue calibration. Primary v4 runs may
use the downloaded release in both operational directions, with that dependency
recorded. A reduced-exposure diagnostic trains on released synthetic -> source
44b6 -> target 6bba without using 6bba-derived calibration. For target 44b6, a
source-6bba-only regeneration or no-synthetic comparison is needed for the analogous
claim; the released files do not qualify. Do not use that caveat to block the
operational study. Public checkpoint/teacher exposure remains in every inherited
lane. Never label the complete experiment clean OOF.

Sample domains first, then acquisitions/examples, then observed event groups.
Do not sample uniformly over the 11.85 million zebrafish nodes and drown out the
real Biohub fine-tuning or smaller sources. Cap the contribution of each event
bag; multiple daughter/path alternatives do not create independent positives.
Use explicit masks for weak/unknown supervision and class/domain denominators.

## Domain-gap controls

The release has no internal births/deaths and no missing observations. Add
observation-level dropout, duplicate peaks, bounded localization jitter, spatial
crop entry/exit and missing-frame masks. These corrupt the observation, not the
known simulated biological truth. A daughter outside the crop censors the observed
pair target; it is not a false biological division. Never bridge a missing frame
with a submitted multi-step edge. Do not reverse a division into a biological merge.

Image augmentation uses source-only Biohub image statistics: PSF/blur, gain,
background, shot/read noise, depth attenuation and subvoxel displacement. Separate
synthetic appearance randomization from geometric motion corruption. All channels,
views, coordinates and daughters receive consistent spatial transforms. Preserve
original examples and record seeds. The generator's mixed voxel/physical velocity
implementation and per-frame radius changes require measured augmentation checks.
No target-embryo labels or density estimates enter a source-specific simulator.

Optional later arm: render short Zoo trajectory windows with an explicitly
synthetic, source-calibrated image renderer. This supplies simulated appearance
paired with weak experimental geometry, not missing real microscopy. Keep it
separate from the primary image pretraining and measure its incremental benefit.
Do not require this arm or new image downloads to complete W400-W490.
