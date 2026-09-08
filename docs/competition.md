# Competition reference

Official source: [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).
Facts below were checked against the local official-page mirror on 2026-09-08.

## Objective and evaluation

Detect fluorescently labeled zebrafish embryo cells in 3D microscopy videos and
link detections across time, including cell divisions. The submission describes
a graph of nodes and temporal edges.

The score combines adjusted edge Jaccard and 0.1 times division Jaccard. Node
matching uses physical centroid distance; sparse annotations and an adjustment
for predicted node counts affect scoring. Read
[`reference/overview/evaluation.md`](../reference/overview/evaluation.md) for the
published metric and its link to organizer implementation details. Organizer
metric changes are indexed in [`reference/forum/INDEX.md`](../reference/forum/INDEX.md).

## Data schema

Source: [`reference/overview/data-description.md`](../reference/overview/data-description.md).

- `train/*.zarr`: Zarr v3 image groups with an array at `0/`, axis order
  `(T, Z, Y, X)`, `uint16`, typically `(100, 64, 256, 256)`.
- `train/*.geff`: sparse tracking graphs paired by sample name. Node IDs live
  at `nodes/ids`, integer centroid coordinates at `nodes/props/{t,z,y,x}/values`,
  and source/target pairs at `edges/ids`. Root GEFF metadata includes an
  estimated total node count.
- `test/*.zarr`: visible example clips copied from training. Kaggle substitutes
  hidden test data during submission reruns; the official description estimates
  hidden-test size at approximately the training-set size.
- `sample_submission.csv`: example of the required node/edge row format.

Physical voxel scale: `(z, y, x) = (1.625, 0.40625, 0.40625)` micrometers.
Use actual `0/zarr.json` metadata for dimensions and codecs. Sparse unlabeled
cells should not automatically be treated as negatives. The first underscore
segment of each sample name identifies an embryo. The hidden train/test split
is embryo-disjoint; visible example test copies must not be used as independent
validation.

Run `python tools/inspect_data.py` to regenerate `work/data-inventory.json`.
The complete extracted member list and sizes are in `work/data-manifest.json`.

The official file inventory fetched on 2026-09-08 contains 24,886 files totaling
87,609,892,618 bytes (81.6 GiB): 199 training image volumes with 199 paired GEFF
annotation graphs, four visible test volumes, and the sample submission CSV.
All 203 image arrays were inspected and have shape `(100, 64, 256, 256)` and
dtype `uint16`. The sample submission has 20 rows: 12 nodes and eight edges.

## Submission

Source: [`evaluation.md`](../reference/overview/evaluation.md) and
[`code-requirements.md`](../reference/overview/code-requirements.md).

Submit a Kaggle notebook that writes `submission.csv`, runs with internet
disabled, and completes within the current 12-hour CPU/GPU limit. The header is:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Node rows contain integer voxel coordinates and `source_id=target_id=-1`.
Edge rows contain source/target node IDs with unused node fields set to `-1`.
Dataset names match test directory stems, every test dataset must appear, and
`id` is a consecutive throwaway index. Read the official pages before building
submission validation or inference code.

## Dates and limits at snapshot

| Item | Value |
|---|---|
| Entry and team merger deadline | 2026-09-22 23:59 UTC |
| Final submission deadline | 2026-09-29 23:59 UTC |
| Maximum team size | 5 |
| Daily submissions | 5 |
| Notebook runtime | 12 hours CPU or GPU |

Dates come from [`timeline.md`](../reference/overview/timeline.md); other limits
come from the official rules and competition metadata. Consult current
[`rules.md`](../reference/overview/rules.md) for external-data allowances,
eligibility, and winner requirements.
