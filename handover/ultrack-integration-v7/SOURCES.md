# Evidence and source pins

Inspected 2026-09-12. Repository inspection used the connected GitHub API;
upstream documentation was also checked on the web. Runtime execution and
competition data access were not performed during planning.

## Repository evidence

Base: [`fb5521629eb41c8c485b291a5bcf344944c113ae`](https://github.com/matheuspf/cell-tracking/commit/fb5521629eb41c8c485b291a5bcf344944c113ae),
current merged main at inspection. Its history includes measured v6 commit
`06e2deedf23f5d7108423d14466316e608c93be1` and the parallel handover merges.

- [Measured v6 report](../../results/segmentation-tracking-v6/final_report.md): C0/P0 scores, zero learned masks, no Ultrack score, runtime/disk blockers, fresh-inference checks and validation limitations.
- [V6 continuation](../segmentation-tracking-v6/CONTINUATION.md): local runtimes, artifact paths, source checkpoints, current inference commands and exact unimplemented boundaries.
- [V6 adapter](../../tools/segmentation_tracking_v6/ultrack_adapter.py): bare image array, missing scale, eager label materialization, export and native-writer gaps.
- [P0 controls](../../tools/segmentation_tracking_v6/controls.py): source-trained complete-native residual, supported-label semantics, frozen-fork decoder and official evaluator call.
- [Common paths/metadata](../../tools/segmentation_tracking_v6/common.py): native axes, spacing, inventory and resource conventions.
- [Competition snapshot](../../docs/competition.md) and [root instructions](../../AGENTS.md): data/submission schema and reference-refresh obligations. Current competition limits must be rechecked locally; this plan does not assert that a September 8 snapshot is current.

## Upstream code (pin, not a floating latest dependency)

Ultrack: [`5c94d845eb0a7b78c8dc24492ef00f218a467995`](https://github.com/royerlab/ultrack/commit/5c94d845eb0a7b78c8dc24492ef00f218a467995).
The latest-commit search reported this commit, dated 2026-08-13. Verify actual
source identity at installation. The releases/latest API instead returned the
2024 `zebrahub_publication` snapshot; do not conflate publication releases with
current source or package versions.

| Inspected file at that pin | Relevant contract |
|---|---|
| [pyproject.toml](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/pyproject.toml) | Python >=3.11,<3.14; Zarr >=3; python-mip; gurobipy dependency distinct from license. |
| [trackingconfig.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/config/trackingconfig.py) | CBC/GUROBI choices, signed costs require identity, default power=4, windows, gaps/time limits. |
| [dataconfig.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/config/dataconfig.py) | Working directory, SQLite, metadata and write behavior. |
| [segmentationconfig.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/config/segmentationconfig.py) | Size limits are voxel counts, contour/foreground thresholds and workers. |
| [linking/processing.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/linking/processing.py) | Image channel sequence; physical scale; target-shift convention; overlap/distance links and add_links. |
| [mip_solver.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/solve/solver/mip_solver.py) | Actual python-mip/CBC backend, objective construction and solver parameters. |
| [tracks_layer.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/core/export/tracks_layer.py) | Dataframe id, parent_id, track_id and parent_track_id are distinct. |
| [utils/edge.py](https://github.com/royerlab/ultrack/blob/5c94d845eb0a7b78c8dc24492ef00f218a467995/ultrack/utils/edge.py) | Single or multiple label inputs; foreground union, averaged boundaries and per-frame conversion. |

## Documentation and method

- [Official overview/workflow](https://royerlab.github.io/ultrack/): foreground/contours, hierarchy, linking, optimization and exports.
- [Official API](https://royerlab.github.io/ultrack/api.html): add_links, add_new_node, add_flow, selected-mask export, annotation flags and source-inspection entry points.
- [Official installation](https://royerlab.github.io/ultrack/install.html): isolated environments; GPU image processing; solver license distinct from package installation.
- [Official configuration](https://royerlab.github.io/ultrack/configuration.html): limits and objective parameters.
- [Method paper, Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02778-0): joint selection under segmentation uncertainty. Published results on other datasets do not predict this repository's score.

Proposed arm order, search ceilings, resource ceilings, promotion gates and
hybrid design are planning decisions, not claims from those sources. No new
accuracy result, hardware benchmark, package installation or valid Ultrack
runtime is claimed by this planning commit.
