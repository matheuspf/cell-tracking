# Access, preparation and schema reference

This document makes the September 8, 2026 snapshot understandable without the
original machine. All counts and availability statements below describe that
capture. Public URLs may change. The committed
[inventory](dataset_inventory.json) identifies the captured files and inspection
scope; it is not evidence that every upstream resource is unchanged today.

## What a fresh checkout contains

The checkout contains this documentation, the aggregate inventory and verification
receipts, and authored preparation/reader code. It contains no original image
volumes, Zoo stores, RIKEN archives, generated NPZ arrays, model weights, private
transcript, or signed download URLs. The full 37,743-entry download receipt and
3,725-entry prepared checksum list remain local artifacts; compact fingerprints,
per-archive hashes and the meaningful validation outcomes are included here.

Historical original paths are rooted at `work/biohub-forum-archive/`; historical
derived paths are rooted at `work/biohub-data-guide/`. Those paths are relative
to the repository. The separately maintained official competition data lives at
`/kaggle/input/competitions/biohub-cell-tracking-during-development` on the
preparation machine. Restoring external labels with `--labels-only` does not need
that official input or an ignored reference snapshot.

## Retrieval coverage and automatic-access limitations

The archived discussion contained the original post and all ten reported replies.
There were **no detected missing messages** in that snapshot; completeness cannot
cover deleted, private or subsequently added messages. The data-bearing messages
were:

- **3507381**, José Freitas, August 1: the synthetic builder and explorer notebooks.
- **3521291**, hengck23, September 5: the Virtual Embryo Zoo site and one screenshot.
- **3521851**, hengck23, September 7: the RIKEN/SSBD project.
- **3521156**, José Freitas, September 5: label clarification. A division is one
  parent fork, and daughters share the source `track_id`. This adds semantics,
  not another dataset.

The remaining replies were 3521235, 3521140, 3521070, 3510364, 3510754, 3509286
and 3509458. They contained discussion, questions or acknowledgements, with no
additional downloadable dataset link in the capture. Their method suggestions
are community ideas, not organizer requirements or experimental evidence in
this branch. In particular, a reply called the Zoo resource synthetic; inspection
of the source pages and stores identified experimental tracking exports.

The only unretrieved advertised dataset artifact was the **exact enriched
Tribolium ZIP**, which returned HTTP 404. All 1,151 files in its public directory
were retrieved and a replacement ZIP was built locally. The replacement preserves
the recovered content; it does not establish the missing upstream ZIP's hash or
byte identity. No manual action was needed to complete that content recovery.

No images paired with the Zoo or RIKEN tracks were downloaded. Separate image
links on some study pages are listed in [DATASETS.md](DATASETS.md), with unverified
correspondence. They were not failed downloads and are not part of the acquired
inventory. Six RIKEN archives are downloaded but lack an in-depth label audit;
that is an inspection gap, not an access failure.

## Per-source acquisition routes

These are restoration instructions if the inputs are absent. They do not select
any dataset for training, and they were not rerun as part of this documentation
commit.

### Synthetic images and original notebooks

The public [builder output](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)
contains the `biohub_synthetic/` directory with `static/`, `sequences/`,
`manifest.json` and `metadata.json`. The
[explorer](https://www.kaggle.com/code/josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking)
uses that same output; downloading both does not provide two datasets.

With the Kaggle CLI installed in an appropriate environment and authentication
configured outside Git, the output command is:

```sh
mkdir -p work/biohub-forum-archive/downloads/kaggle
kaggle kernels output josefreitasalvesneto/biohub-synthetic-dataset \
  -p work/biohub-forum-archive/downloads/kaggle --page-size 100
```

After extraction, the required path is
`work/biohub-forum-archive/downloads/kaggle/biohub_synthetic/manifest.json`.
Notebook sources can be obtained without execution using `kaggle kernels pull`
for the two owner/slug identifiers above, or the notebook's download interface.
The inventory includes their notebook IDs, declared inputs and source hashes.
Notebook IDs **129280459 / 129280032 are not immutable version numbers**. No
immutable output version was recorded, so a later download may differ.

The captured API output listing gave implausible NPZ byte sizes. The original
download used paginated output URLs, the author's manifest sizes and ZIP CRC
checks; listing sizes alone were not treated as trustworthy. If automatic access
fails due to login, an expired output URL or API behavior, the same public notebook
can be opened while signed in to Kaggle and its output downloaded manually.
Fresh signed output URLs may be needed. Authentication material and signed URLs
are not part of this handover.

### Virtual Embryo Zoo

Each `zoo` record in the inventory has a `source_page`, a `base_bundle.url`, and
an `enriched_store.url`. Downloading the base ZIP preserves the original compact
bundle; the implemented graph conversion reads the **enriched directory** at
the recorded path. The two are overlapping representations.

At capture time, enriched stores could be mirrored by recursively requesting
their directory listings with `Accept: application/json`. A root-listing example
using the exact committed URL is:

```sh
python - <<'PY'
import json
from pathlib import Path
from urllib.request import Request, urlopen

inventory = json.loads(Path('docs/external-data-guide/dataset_inventory.json').read_text())
entry = next(x for x in inventory['zoo'] if x['species_key'] == 'tribolium')
request = Request(entry['enriched_store']['url'], headers={'Accept': 'application/json'})
with urlopen(request, timeout=60) as response:
    print(response.read().decode())
PY
```

This command lists a directory; it does not mirror the store. A mirror must retain
all nested files, including dot-prefixed `.zgroup`, `.zarray` and `.zattrs`, and
preserve the directory layout. The acquisition scripts used bounded concurrency,
retries and pauses between listing requests. The committed aggregate record gives
each tree's file count, bytes and fingerprint for comparison after restoration.
The two extra CSVs have their own direct URLs and SHA-256 hashes in
`additional_csvs`; the full verifier uses both.

If directory listing access changes, manual access through the species page or
its download link may still work. The working base ZIP is a separate fallback
for inspecting that representation. It is not a promise that it can replace the
enriched directory without a converter/schema check. Retrying the same missing
Tribolium enriched ZIP cannot reconstruct its absent upstream bytes; the public
directory was the successful recovery route in this snapshot.

### RIKEN / SSBD

The [project page](https://ssbd.riken.jp/database/project/5-Keller-FishEmbryo/)
and each `riken[].download_url` identify seven public BDML 3.0 ZIPs. Save each
ZIP to its inventory `path`. Every record includes the source-published SHA-256,
recorded local SHA-256, byte size and ZIP member names/uncompressed sizes.

For a downloaded archive, ordinary tools can check its identity and integrity:

```sh
sha256sum work/biohub-forum-archive/downloads/ssbd/bdml/zebrafish_animal_c_bdml3.0.zip
unzip -t work/biohub-forum-archive/downloads/ssbd/bdml/zebrafish_animal_c_bdml3.0.zip
```

The first value can be compared with `published_sha256` in the matching inventory
record. If automation fails, the same ZIP links can be downloaded through the
project page. The HDF5 and XML exposed separately are the members of those ZIPs,
not new acquisitions. All seven compressed ZIPs total 9,780,046,632 bytes;
their members total about 51.8 GB uncompressed. Preparation extracts only animal
C's HDF5, requiring 3,199,666,392 additional bytes of cache space. No credentials
were required for these public URLs in the recorded download.

## Reproduce the implemented preparation

These commands run from the repository root. On the original machine, the data
and isolated dependencies already exist. On another machine, after acquiring the
originals, activate an existing Python 3.12 Conda environment or create the named
environment if absent. Package installation belongs inside that environment.

```sh
conda activate cell-tracking
export PYTHONNOUSERSITE=1
python -m pip install --target work/biohub-data-guide/python-deps \
  -r tools/biohub_external_data/requirements.txt
export PYTHONPATH="$PWD/work/biohub-data-guide/python-deps"

python tools/biohub_external_data/prepare_data.py --labels-only
python tools/biohub_external_data/data_adapter.py --sample seq_0000
python tools/biohub_external_data/data_adapter.py \
  --export-demo work/biohub-data-guide/prepared/synthetic_format_example.csv
python tools/biohub_external_data/verify_data.py
python tools/biohub_external_data/audit_synthetic_motion.py
```

The pinned dependency directory is isolated from the current solution's CUDA
runtime. Preparation reads all synthetic files, every restored enriched Zoo
directory and animal C's ZIP. Its complete-snapshot prerequisites are:

- `downloads/kaggle/biohub_synthetic/manifest.json` and all files listed there.
- All six `downloads/virtual-embryo-zoo/*attributes_bundle.zarr/` trees.
- `downloads/ssbd/bdml/zebrafish_animal_c_bdml3.0.zip` (or its previously extracted
  cache). The other six RIKEN ZIPs are inventory inputs, not processed by this step.
- Both Zoo source CSVs for the subsequent full verification.

`--labels-only` writes the synthetic label NPZs, synthetic manifest/audit, six Zoo
graph NPZs/manifest and animal C preview/audit. It does not copy original images,
train a model, generate an inference submission, or use official competition data.
Zoo conversion can require several GB of RAM. The six-species/full-release
verification intentionally asserts the captured complete inventory; it is not
a generic validator for a selected subset or a changed upstream release.

Without `--labels-only`, the historical preparation entry point also builds
local presentation previews. That mode requires the official input
`train/44b6_0113de3b.zarr` and corresponding GEFF plus the uncommitted
`work/biohub-data-guide/reference/local-train-audit.json`. It is not the fresh
checkout restoration route. The optional HTML builder and presentation assets
are not required to read this handover or run label-only preparation.

## Reader and prepared schemas

The full dtype/shape inventory is in [dataset_inventory.json](dataset_inventory.json).
The following explains the meanings that are easy to lose when adapting a loader.

**Synthetic static:** `zyx_native`, `zyx_pooled`, and `zyx_um` are float32 `(N,3)`.
The first two are coordinates in the native and XY-stride-four grids. For an
illustrative center `[8,120,200]` in native ZYX, the pooled coordinate is
`[8,30,50]` and the physical coordinate is `[13,48.75,81.25]` µm. This arithmetic
is not a measured example or a model target choice.

**Synthetic sequence:** those same arrays are accompanied by int64
`node_id[N]`, `t[N]`, `edges[E,2]`, `division_parent_ids[D]`, `tracklet_id[N]`,
and `source_clone_id[N]`. Node IDs equal file-local row indices. Each edge links
adjacent frames. At a fork, both daughters inherit the source clone ID, while
their prepared tracklet IDs differ. `division_parent_ids` counts parents once,
not daughters or edges. The current graph conversion starts a new tracklet at
each root and immediately after a fork.

`load_synthetic(sample_id, grid="pooled", source_root=None)` returns these
prepared arrays with `image`, `zyx_image`, `voxel_um_zyx`, `sample_id`, `kind`
and `split`. Images are float32 TZYX after division by 65,535. Static images gain
a time dimension and are sampled `[:, :, ::4, ::4]` in the default grid;
sequence images are already sampled. `grid="native"` is supported for static
images only and rejects sequences. `source_root` can relocate the original
`biohub_synthetic/` folder, but prepared labels/manifests still resolve under
the checkout's `work/biohub-data-guide/`. Manifest `source` paths are relative
to that guide directory, not to the caller's current working directory.

For sequences, the reader also returns Boolean `is_division_parent[N]` and
`division_target_observed[N]`. The second is false at frame 5 because no future
frame was released. It is a one-step observation mask, not a completeness mask
for arbitrary longer event windows. Static examples have no temporal target.
The default reader is an inspection/reference implementation that reparses the
manifest and opens NPZs per call; no optimized training data pipeline is provided.

`points_to_heatmap` is an optional derived-label example: fractional centers,
Gaussian sigma in µm, truncation at three sigma, and maximum combination where
targets overlap. Its default `sigma_um=2.0` is an implementation default from the
illustration, not a selected training hyperparameter or the evaluator's radius.

**Zoo:** int64 `node_id[N]`, `source_point_id[N]`, `t[N]`, `tracklet_id[N]`,
`parent_tracklet[K]`, `edges[E,2]`, `division_parent_ids[D]`, and float32
`zyx_source[N,3]`. `parent_tracklet` is indexed by tracklet ID, not node row;
roots are `-1`. Prepared edges reference node rows. Source point IDs encode
`t * max_points_per_frame + slot`, where the per-frame capacity comes from the
source store. Original per-point display attributes are not copied into these
NPZs; they remain in the source Zarr stores. Their names/metadata are recorded
in the committed inventory. Positions have not been converted to µm.

**RIKEN animal C excerpt:** int64 `t[N]` (0 through 9), float32 `zyx_um[N,3]`,
and Unicode `measurement_id[N]`. There are no prepared graph edges, masks or
feature vectors in that NPZ. Features remain in the original HDF5; joining
`data/<frame>/feature/0` to `object/0` uses the measurement `ID` within that
frame and `fID` from the 14 committed feature definitions. Display frame 0 maps
to source object time 1; a 90-second interval is not an absolute acquisition
timestamp. In the inspected example, a fluorescence FWHM extent could be about
291 µm along Y while the corresponding nucleus-core box spanned about 15 µm.
Missing/outlier widths were preserved; no replacement masks were invented.

**CSV helper:** `prediction_rows` demonstrates the ten submission columns
`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`. It takes native
coordinates, rounds with NumPy `rint`, and requires edges indexing the supplied
node array at consecutive times. The demo uses `SYNTHETIC_FORMAT_DEMO` and is not
a submission. This helper has no image shape argument, no full degree/duplicate
validation and no inference loop; final submission validation is a separate
existing tracker responsibility.

## What was verified, and how the inventory was compiled

[verification_summary.json](verification_summary.json) retains the recorded
checks for all 3,713 synthetic label sets, all six Zoo graphs, exact ascidian/worm
CSV correspondence after ID remapping, static stride equivalence, sequence grid
alignment, final-frame censoring and animal C's ten-frame excerpt. Graph checks
establish internal structural consistency, not biological correctness. Download
CRC and SHA checks establish file integrity, not label accuracy or independence.

The summary also retains the earlier local visual/browser audit: 140 checked
PNGs, six Zoo selections, ten RIKEN frames, 46 working local links/anchors and
four viewport widths. Those receipts describe the local presentation at that
time, not files bundled into this checkout. Illustrative trajectory windows
were selected to show readable structures and do not estimate biological rates.

Its separate September 9 synthetic motion audit measures all continuation-edge
lengths and first-frame sister separations in physical coordinates, plus internal
birth/death counts. The committed audit script reads labels only. It does not
generate new simulations or compare models.

The committed `build_inventory.py` compiles these compact records from existing
archive verification, source-download lists, notebook metadata and prepared audit/
checksum files. It writes only to `work/biohub-data-guide/branch_inventory/`;
updating the committed snapshot is a separate documentation action. It requires
the original local receipts and does not bootstrap missing data or perform a
new hash pass over the raw archive.

For directory collections, the content fingerprint is SHA-256 of UTF-8 JSON
containing sorted `[relative_path_within_tree, bytes, sha256]` records, with
`separators=(',', ':')` and `ensure_ascii=True`. It was compiled from the saved
per-file hashes. Individual archive/sample hashes are direct recorded SHA-256s;
a reconstructed ZIP has its own identity. Regeneration with another compression
library can change NPZ/ZIP bytes even when decoded arrays agree.

## Source declarations and recorded reference pins

The synthetic author's CC0 declaration and clone/division clarification are in
discussion 732103. The Zoo software license is MIT, but that does not identify
all hosted studies' data licenses. In
[discussion 734330](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330),
HOST **Thibgolds** stated on August 13 that public Zebrahub resources were allowed
and had no hidden-test overlap. This is a Zebrahub-specific organizer statement,
not a general physical-calibration or label-accuracy assurance.

The captured [competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
permit external data subject to public/reasonable accessibility and retain source
obligations. Their winner clause exempts input data/pretrained models from the
code-relicensing requirement. RIKEN's source declares CC BY-NC-SA; intended
training/prize-use compatibility remains unresolved here. These are dated source
observations, not a new legal conclusion or blanket approval for future use.

The inspected metric revision is `075fc5f5a52d11077f9dc2b074644618f26939e2`;
the inspected inTRACKtive converter revision is
`8afc30b5c0a42632ad971e76dbdb559286bec0bf`. Their source semantics are summarized
in the committed documents and implemented conversion. Neither pin implies the
upstream service or current competition rules have remained unchanged.
