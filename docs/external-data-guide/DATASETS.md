# Downloaded external datasets: labels, provenance and inventory

This is the portable record for a new agent reading the branch without the
ignored `work/` directory. It describes the **2026-09-08 download and preparation
snapshot**, documented on 2026-09-09. All 11 messages in
[discussion 732103](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103)
were archived. All linked dataset contents were recovered, including the public
contents behind a broken Tribolium ZIP link. No external-data training or transfer
gain experiment was performed during this preparation work.

[dataset_inventory.json](dataset_inventory.json) contains the exact original and
prepared paths, source URLs, per-resource counts, file sizes, SHA-256 receipts,
notebook identities and inspection scope. Paths in that file are repository
relative unless explicitly marked as the canonical competition input. Raw files
and generated labels are intentionally absent from Git; the information and
maintained preparation code are in the branch.

## Available label types and competition representation

The synthetic release has paired images and exact simulated centre/temporal
labels. The Zoo exports have experimental trajectories and reconstructed
divisions but **no downloaded images**, with unresolved physical calibration
and biological label quality. RIKEN provides measurements and positions; the
inspected sample has no explicit temporal links. A sequence of positions is not
itself a set of supervised cell identities or parent-child relationships.
None of the downloaded external resources provides dense semantic or instance
segmentation masks. These facts describe available information; selection and
training use remain decisions for the next agent.

Competition inputs remain at
`/kaggle/input/competitions/biohub-cell-tracking-during-development` (`data/` is an
alias). The target is cell centres plus directed temporal edges. Prepared centres
retain fractional precision; final submission rows use native integer coordinates.
The optional stride-four image preprocessing below is not a metric
requirement. The evaluator's 7 µm node-matching tolerance is not a temporal-link
radius or a prescribed heatmap width.

## Synthetic release: two related collections

Source: José Freitas's
[builder notebook](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)
and [explorer notebook](https://www.kaggle.com/code/josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking).
Original root: `work/biohub-forum-archive/downloads/kaggle/biohub_synthetic/`.
The builder output is one release; the explorer reuses it and is not a second
independent dataset. Original notebooks were downloaded and exported to Python,
not executed. Their Kaggle notebook IDs are 129280459 and 129280032 respectively;
these are notebook IDs, not pinned version numbers. No immutable Kaggle version
number was captured. The inventory records source/metadata hashes for this copy.

- **Static:** `static/vol_00000.npz` through `vol_01538.npz`: 1,539 images,
  12,916,348,212 bytes and 423,853 centre labels. `volume` is `uint16` ZYX,
  shape `(64,256,256)`; `centroids` is floating `(N,3)` ZYX in native voxels;
  `voxel_um` is `(1.625,0.40625,0.40625)` µm/voxel. There are no temporal edges
  in these one-frame examples.
- **Sequences:** `sequences/seq_0000.npz` through `seq_2173.npz`: 2,174 six-frame
  clips, 6,951,054,644 bytes; 4,056,226 nodes, 3,460,295 temporal edges and
  165,267 division parents. `volumes` is `uint16` TZYX `(6,64,64,64)`;
  `nodes` is float32 `(N,5)` containing `[t,z,y,x,track_id]`; `edges` is integer
  `(E,2)` indexing node rows; `divisions` indexes parent nodes. Source node
  coordinates still refer to the native `(64,256,256)` grid: divide **only Y/X
  by four** to align labels with the released images. The reader does not sample
  sequence images again. The source `voxel_um_pooled` field is `(1.625,1.625,1.625)` µm
  per image voxel. Native-resolution sequence images were not released.

Original `track_id` is a clone/lineage ID that daughters inherit; it is not a
unique cell segment ID. The audit found 480,640 repeated `(t, source_clone_id)`
combinations beyond the first occurrence. The prepared graph-derived
`tracklet_id` breaks after forks. Node and track IDs are file local, so identical
numbers in different samples do not denote the same biological identity.

Prepared files are
`work/biohub-data-guide/prepared/synthetic/vol_XXXXX_labels.npz` and
`seq_XXXX_labels.npz`, indexed by
`work/biohub-data-guide/prepared/synthetic_manifest.json`. They include
`zyx_native`, `zyx_pooled` and `zyx_um`; sequence files add `node_id`, `t`,
`edges`, `division_parent_ids`, `tracklet_id`, and `source_clone_id`.
`tools/biohub_external_data/data_adapter.py::load_synthetic` loads the original
image with aligned points. Its default static preprocessing is XY **stride
sampling**, not block averaging; pooled spacing is `(1.625,1.625,1.625)` µm.
`points_to_heatmap` creates a Gaussian centre target, not a segmentation mask.
The loader exposes final-frame censoring via `division_target_observed`.

Distribution and provenance limitations:

- Dense simulated labels, short six-frame clips, simplified imaging and
  deliberately oversampled divisions differ from sparse experimental labels.
  No relationship between synthetic loss and competition score was measured.
- The generator fits density/tissue shape to ten videos from the first sorted
  embryo, `44b6`. Its fixed motion/render calibration has not been fully traced
  to source folds. Using this release when `44b6` is an evaluation target would
  carry target-derived distribution information. The release is derived from
  competition training data and is not independent external biology.
- Prepared synthetic splits (3,359 train / 354 holdout samples) are generator
  sanity splits only. Whole example IDs are assigned deterministically by hash;
  this does not establish biological validation or independence between releases.
- The local competition snapshot has 133,318 annotated nodes and **151 annotated
  parent forks** over 199 samples from two embryos. The forum's approximate 304
  events and asserted 0.26% real rate do not establish a reliable biological
  prior. Synthetic 165,267 / 4,056,226 = 4.07% is descriptive, with final-frame
  censoring. The author's proposed 15.7× correction is not a validated real-data
  prevalence estimate or a parameter selected by this handover.

The author declares **CC0** in discussion 732103. This is a recorded source
declaration for the captured release.

### Generator behavior and measured motion

The captured metadata reports a planar tissue shell, real detected-count range
140–417, and claimed calibration constants of 1.86 µm/frame median motion,
0.30 persistence and 7.24 µm sister separation. These are author values, not
equivalent to a complete physical validation of the exported graphs.

Inspection of the captured builder's `run_dataset_build` shows deterministic
seeds `100000+i` for static samples, `500000+j` for sequence dynamics, and
`900000+j*97+t` for frame rendering. Initial requested counts are sampled from
the calibrated count distribution and multiplied by 1.25, before placement
exclusions. The default split probability is 0.05 per cell per transition.
Cells continue or divide; the sequence loop has no explicit death or independent
entry/exit process. It clips updated centers to native ZYX bounds
`[4,12,12]` through `[59,243,243]`. Frame rendering resamples radii for the
supplied centers, so persistent identity does not imply constant rendered
appearance. These are properties of the released generator, not selected
settings for a future model.

The source adds its continuation velocity directly to native voxel positions,
while converting sister offsets from micrometres by dividing by voxel spacing.
Thus its metadata calibration numbers alone do not establish that exported
physical displacement distributions match the stated values. A descriptive
audit of all 2,174 prepared sequence graphs measured:

- **3,129,761 nondivision edges:** physical step median 1.6510 µm, 10th/90th
  percentiles 0.5566 / 4.5932 µm, range 0–56.8753 µm.
- **165,267 sister pairs at their first frame:** separation median 7.0789 µm,
  10th/90th percentiles 4.9244 / 9.1702 µm, range 0.3384–14.6798 µm.
- **Zero** nodes after frame 0 without a parent, and zero nodes before frame 5
  without a child. Missing context occurs at the clip boundaries; this graph
  does not supply internal appearance/disappearance examples.

Distances use the prepared float32 `zyx_um`, Euclidean distance and one count per
continuation edge or dividing parent. The audit is in
[verification_summary.json](verification_summary.json), reproducible with
[audit_synthetic_motion.py](../../tools/biohub_external_data/audit_synthetic_motion.py).
It is a label-geometry measurement, not a training experiment or a choice of
motion thresholds.

## Virtual Embryo Zoo: six experimental tracking exports

Original root: `work/biohub-forum-archive/downloads/virtual-embryo-zoo/`.
For each species, we have a base `tracks_SPECIES_bundle.zarr.zip` and a complete
enriched `tracks_SPECIES_attributes_bundle.zarr/` store. They overlap and are
not separate training examples. Prepared graph path:
`work/biohub-data-guide/prepared/zoo/SPECIES_graph.npz`.
The counts below refer to the prepared enriched graph; **nodes are observations
across time, not unique cells**. Counts and bytes for every representation are in
the JSON inventory.

- **Zebrafish / Danio rerio:** 522 frames, 11,851,323 nodes, 11,770,767 edges,
  398,662 tracklets and 159,053 division parents. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/danior)
  links the Lange/Zebrahub study and describes Ultrack tracking by the original
  authors. Its species matches the competition. An organizer
  [explicitly permits public Zebrahub resources and confirms no hidden-test overlap](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330).
  That statement does not provide physical calibration for this viewer export.
- **Drosophila melanogaster:** 301 frames, 1,877,876 nodes, 1,872,888 edges,
  154,508 tracklets and 74,760 division parents. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/drosophilam)
  identifies Amat, Nature Methods (2014), and the original authors' GMM tracking.
- **Mouse / Mus musculus:** 532 frames, 7,349,203 nodes, 6,988,673 edges,
  360,530 tracklets and **zero recorded division parents**. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/musm)
  identifies McDole, Cell (2018). The graph records continuity links; zero forks
  does not prove that no biological divisions occurred.
- **Ascidian / Phallusia mammillata:** 88 frames, 17,052 nodes, 16,988 edges,
  732 tracklets and 334 division parents. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/phalussiam)
  identifies Guignard, Science (2020), and Astec tracking. Original viewer point
  fields include radius in addition to ZYX; the prepared graph preserves ZYX,
  while radius remains in the source store/CSV.
- **Worm / C. elegans:** 379 frames, 110,899 nodes, 110,897 edges,
  1,332 tracklets and 665 division parents. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/celegans)
  identifies Moyle, Nature (2021), and Linajea tracking.
- **Beetle / Tribolium castaneum:** 80 frames, 305,622 nodes, 302,460 edges,
  25,100 tracklets and 10,969 division parents. [Source page](https://virtual-embryo-zoo.sf.czbiohub.org/dataset/triboliumc)
  credits Akanksha Jain and Ultrack. The advertised enriched ZIP returned HTTP
  404; its complete public folder (1,151 files) was mirrored and zipped locally.
  `tracks_tribolium_attributes_bundle.zarr.zip` is that **local replacement**,
  not the upstream ZIP. The separate base ZIP downloaded normally.

All six are Zarr v2 viewer stores, unlike the competition's Zarr v3 images.
`points` is a padded per-frame array with source-declared ZYX fields (and optional
radius). Padding is approximately `-9999.9`. `tracks_to_points` is a CSR mapping
from tracklets to point IDs and raw source coordinates. A point ID decodes as
`t * max_points_per_frame + slot`. Preparation retains source units; means/extents
are display metadata, not proof that coordinates need inverse normalization.
Physical scale and frame duration are still unverified per species.

`tracks_to_tracks` contains a lineage reachability closure, **not direct edges**.
Its encoded column metadata identifies the immediate parent tracklet (positive
values are one-based IDs; roots decode to -1). The converter validates that
metadata, joins adjacent observations inside each tracklet, and connects parent
endpoints to child starts only at adjacent times. All six exports had zero
within-track gaps and zero nonadjacent parent links under these checks.
The viewer's `divisions` attribute encodes display colours; it is not a binary
training target. Some continuous viewer attributes are normalized too.

Prepared NPZ fields are `node_id`, `source_point_id`, `t`, `zyx_source`,
`tracklet_id`, `parent_tracklet`, `edges`, and `division_parent_ids`.
`parent_tracklet` has one entry per tracklet, with -1 for roots. Node/track IDs
are file local. These are original tracking results plus a checked conversion;
graph integrity does not prove biological annotation accuracy.

Additional downloaded files `ascidian_tracks_withSize.csv` and
`elegans_tracks.csv` represent the same source tracks as their Zoo stores.
Time, coordinates and parent relationships matched those prepared graphs
exactly. They provide an independent **format check**, not independent data.

No Zoo microscopy images were downloaded. The archived species pages offer
possible later acquisition paths: mouse [IDR0044](https://idr.openmicroscopy.org/webclient/?show=project-502),
ascidian [Figshare](https://figshare.com/s/765d4361d1b073beedd5#/articles/8223890),
worm [Zenodo 6460375](https://zenodo.org/records/6460375), and beetle
[Cell Tracking Challenge 3D datasets](https://celltrackingchallenge.net/3d-datasets/).
Those links have not been audited for image-to-export correspondence, costs,
availability or licenses in this task. They are pointers, not extra acquired
datasets. Originating study terms have not been resolved across all six exports;
the inTRACKtive software's MIT license is not a blanket license for hosted data.

## RIKEN / SSBD: seven zebrafish measurement archives

[Project 5-Keller-FishEmbryo / SSBD-000005](https://ssbd.riken.jp/database/project/5-Keller-FishEmbryo/)
publishes seven BDML 3.0 ZIPs. All seven downloaded ZIPs passed published SHA-256
and ZIP CRC checks. Their combined compressed size is **9,780,046,632 bytes**.
Original root: `work/biohub-forum-archive/downloads/ssbd/bdml/`.

- `zebrafish_animal_a_bdml3.0.zip` — 1,545,317,368 bytes; inventoried only.
- `zebrafish_animal_b_bdml3.0.zip` — 1,848,355,379 bytes; inventoried only.
- `zebrafish_animal_c_bdml3.0.zip` — 621,616,807 bytes; full structure/count audit.
- `zebrafish_dorsal_bdml3.0.zip` — 974,122,136 bytes; inventoried only.
- `zebrafish_in_toto_mzoep_bdml3.0.zip` — 1,922,708,812 bytes; inventoried only.
- `zebrafish_in_toto_wt_bdml3.0.zip` — 1,651,857,532 bytes; inventoried only.
- `zebrafish_ventral_bdml3.0.zip` — 1,216,068,598 bytes; inventoried only.

Each contains one `zebrafish_NAME_bd5.h5` and one `zebrafish_NAME_bdml3.0.xml`.
The JSON inventory lists exact member names/uncompressed sizes and each ZIP's
published SHA. The ZIP and separately hosted raw HDF5/XML are equivalent
representations and do not add independent data. No image dataset is listed by
this project. Independence and relationships between the seven views/specimens
have not been established by this archive audit.

**Inspected animal C only:** 821 frames, 3,447,269 point measurements, coordinates
in micrometres and 90-second time steps. HDF5 `data/<numeric frame>/object/0`
contains structured `ID,t,entity,x,y,z` fields; numeric values can be encoded as
byte strings. The converter selects numeric frame groups: `featureDef`, `objectDef` and
`scaleUnit` are metadata, not frames. `data/<frame>/feature/0` contains
`ID,fID,value` measurements. IDs appear frame-specific; persistent identity is
not established. No explicit temporal or parent-child edge labels were found in
this sample. Other six archives have not had the same label audit.

The 14 feature definitions are two nucleus fluorescence measurements (arbitrary
units), six nucleus-core bounding-box endpoint coordinates, and six fluorescence
FWHM endpoint coordinates (micrometres). FWHM values may be missing or unusually
broad; neither FWHM nor boxes are dense cell-boundary masks. Missing features
remain missing in the local illustrations; no calibrated size/intensity prior
has been fitted from these data.

`work/biohub-data-guide/prepared/riken_animal_c_first10_frames.npz` contains only
the first ten frames / 1,312 points with `t`, `zyx_um` and `measurement_id`.
Its displayed `t=0` maps to source measurement time index 1. Full feature
definitions and inspection scope are included in the committed
[inventory](dataset_inventory.json), under the `riken_animal_c` record.
Nearest-neighbour correspondences, if added later, would be inferred pseudo-labels
rather than supplied ground-truth edges or divisions.

The source declares **CC BY-NC-SA**. Compatibility of a particular training/prize
use has not been established here. The captured competition winner clause exempts
input data/models from relicensing, so the MIT winner requirement alone does not
establish a conflict. General external-data permission does not replace source
terms. This is a record of the saved sources, not a fresh legal determination.

## Verification, restoration and provenance

Download verification recorded **37,743 files / 30,877,642,679 bytes**, with zero
incomplete files and zero errors. That is a storage total including overlapping
base/enriched Zoo stores, the local beetle ZIP, metadata and forum attachment;
it is not unique scientific training volume. Prepared verification recorded
3,725 files / 394,701,884 bytes, including manifests and CSV format demo. Every
synthetic label file and Zoo graph passed coordinate/graph checks. The RIKEN
preview passed structure/finite-coordinate checks. These receipts come from the
saved 2026-09-08 verification, not a new rehash of the entire archive.

Preparation code is in `tools/biohub_external_data/`; pins are in
`tools/biohub_external_data/requirements.txt`. Use Conda `cell-tracking`,
`PYTHONNOUSERSITE=1`, and the isolated dependencies under
`work/biohub-data-guide/python-deps`. `data_adapter.py` itself needs NumPy;
preparation additionally needs Zarr/numcodecs, HDF5/h5py and Pillow. Keep those
versions separate from the active solution's training environment until tested.
The converter reference is pinned to inTRACKtive commit
`8afc30b5c0a42632ad971e76dbdb559286bec0bf`; inspected metric reference is
`075fc5f5a52d11077f9dc2b074644618f26939e2`. These pins describe the audit, not a
claim that upstream is unchanged.

On this machine, inputs and prepared labels already exist: run the loader/checks
from [README.md](README.md), rather than downloading them again. On a fresh
clone, restore originals to the JSON paths, then run the maintained preparation
and verification scripts. An authenticated manual synthetic download is:

```sh
kaggle kernels output josefreitasalvesneto/biohub-synthetic-dataset \
  -p work/biohub-forum-archive/downloads/kaggle --page-size 100
```

Kaggle's CLI listing showed implausible NPZ byte sizes in this snapshot; verify
against the author's manifest and captured release hashes. The public RIKEN and
Zoo source/download URLs are recorded per resource in the JSON. For the broken
Tribolium enriched ZIP, use its public directory or the working base ZIP; simply
retrying the 404 URL cannot restore that missing upstream file. Public directory
listings can be requested as JSON with `Accept: application/json`.
[ACCESS.md](ACCESS.md) provides the complete preparation prerequisites and the
per-source retrieval routes without relying on ignored archive reports. The
recorded counts and hashes describe this snapshot, not a future upstream release.
