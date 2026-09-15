Cellpose cpDINO-ViT-B with ultrack, 14 September 2026

This study measures the **full official competition graph metric** on complete
100-frame clips: adjusted edge Jaccard plus 0.1 times division Jaccard. It extends
the six previously selected detector pilot clips to all their frames, three
clips per embryo. Its five annotated divisions give only a small division sample.
This is an exploratory local experiment, with no claim of hidden-test accuracy.

The detector is frozen `cpdino-vitb`, the strongest new pretrained checkpoint in
the repository's [detector screen](../../docs/detector-benchmark-20260914.md).
Its checkpoint SHA256 is `3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f`.
The existing Cellpose adapter runs unchanged, with native ZYX input, metadata
anisotropy, orthogonal 3D reconstruction, eager BF16 and batch size 8. No training
or learned association model runs in this study. Inherited pretraining exposure
remains unresolved, as documented in the prior development review.

Ultrack receives Cellpose instance labels through `labels_to_contours`, creates
its actual hierarchical segmentation hypotheses, links by mask IoU within
15 micrometers, and jointly selects observations and temporal edges with CBC.
The [final recipe](../../configs/cellpose-ultrack-windowed-v1.json) retains the documented
native-volume area/frontier settings and stock link/birth/death/division weights.
The solver uses 20-frame temporal windows with overlap 5, a 180-second limit per
window, and a 0.1% relative gap target. It exports one complete 100-frame graph.
Every actual solve records status, achieved gap, objective
and bound. A feasible time-limited result is not represented as proven optimal.
The first full clip exposed an upstream hierarchy failure on a six-voxel
disconnected region. Before any full-graph score was inspected, `min_area_factor`
was set to 2.5: with `min_area=20`, ultrack removes connected components smaller
than its eight-voxel hierarchy minimum. The failed run is retained under
`work/cellpose-ultrack-20260914/diagnostics/`.

An initial [full-clip solve recipe](../../configs/cellpose-ultrack-v1.json)
scored 0.9488359062 on `44b6_81c256f0`, but CBC could not find a feasible solution
on the denser `44b6_8f5ab931` within 180 seconds. Temporal windowing was then
applied uniformly to all six clips, including rerunning the first clip. This
decision addressed solver execution; no mask or association weights were tuned.
The first two clips reuse their verified, unchanged hypothesis/link databases.
Original complete graphs, scoring receipts and the failed solve are retained in
`work/cellpose-ultrack-20260914/full-clip/`. Final summary figures use the uniform
windowed recipe; they do not mix the initial full-clip result with windowed runs.

The adapter audits SQL coordinates against the serialized linker points and
independently recomputed rounded native-mask centroids **before linking**.
Pinned ultrack rounds centroids before SQL insertion; no coordinates are changed.
Final CSV coordinates use `np.rint`, consistent with the detector benchmark.
Actual observation `id`/`parent_id` defines the graph; track IDs are never
substituted for node identity.

Segmentation and tracking read an image-only panel and Cellpose outputs. GEFF
annotations and total-node estimates enter a separate evaluation process only.
The unmodified organizer metric is pinned to
`075fc5f5a52d11077f9dc2b074644618f26939e2`, verified against upstream main on
14 September 2026. Fresh assignment is audited for one-to-one matching, time
consistency and the physical 7 micrometer gate. The aggregate is independently
cross-checked and rejects incomplete cohorts. The raw Cellpose center recall
and endpoint coverage are reported separately as detector diagnostics.

From the repository root, using existing isolated environments:

```sh
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python -m tools.cellpose_ultrack.prepare

PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /kaggle/envs/detector-screen-cellpose/bin/python -m tools.detector_screen.cellpose_adapter \
  --panel work/cellpose-ultrack-20260914/panel.json --role tracking \
  --output work/cellpose-ultrack-20260914/predictions/cellpose_cpdino_vitb \
  --artifacts work/cellpose-ultrack-20260914/cellpose

PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/mpf/.conda/envs/cell-tracking/bin/python -m tools.cellpose_ultrack.finish
```

`finish` may run concurrently with inference. It waits for all frames of a clip,
then runs tracking and evaluation in their respective isolated runtimes. Both
jobs preserve the shared detector-screen GPU lock. A resume checks existing
input/graph fingerprints. Masks, databases, CSVs, predictions and detailed
receipts remain in ignored `work/cellpose-ultrack-20260914/`; compact evidence is
published to `results/cellpose-ultrack-20260914/` only for the complete panel.

After all six clips have scored, publish the final checks and report:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.validate
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.diagnose_divisions
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.report
```

Validation checks exact parity with the 12 earlier pilot masks and combines the
six validated per-clip CSVs into `work/cellpose-ultrack-20260914/submission.csv`.
This CSV contains the evaluated local training clips. Division diagnostics use
the official parent/daughter matching windows and are computed after prediction.

Run a bounded integration check after its Cellpose frames exist:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=/home/mpf/code/kaggle/ultrack \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m tools.cellpose_ultrack.track \
  --datasets 44b6_81c256f0 --frames 10

PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m pip install \
  --target work/cellpose-ultrack-20260914/test-deps pytest==9.1.1
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/work/cellpose-ultrack-20260914/test-deps" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m pytest -q tests/test_cellpose_ultrack.py

PYTHONNOUSERSITE=1 PYTHONPATH="/home/mpf/code/kaggle/ultrack:$PWD/work/cellpose-ultrack-20260914/test-deps" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m pytest -q tests/test_cellpose_ultrack.py -k hierarchy
```

The integration graph is excluded from full-metric reporting. The graph checks
reject duplicate IDs/edges, missing parents, merges, nonadjacent links and more
than two daughters. Actual SQL exclusions and CSV round trips are also checked.
Tests include a perfect synthetic division, a deleted daughter edge, and extra
unmatched nodes that change the official node-count adjustment.

`prepare --scope assessment` or `--scope all` supports a separately rooted larger
experiment. Use `--root` consistently across preparation, inference arguments,
tracking and evaluation; scopes cannot overwrite an already frozen panel. The
600-volume pilot alone does not establish all-199 performance or a Kaggle
12-hour runtime. The comparison contains no matched full-clip FOCUS3D run.

The follow-up [matched error analysis](../../docs/cellpose-ultrack-error-analysis-20260914.md)
rescores selected v3 and public Harmonic Fusion on the exact six clips. Reproduce
that audit without changing their graphs:

~~~sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.analyze_errors
~~~

The bounded event-cost study changes only signed appearance, disappearance and
division costs. Its three configurations are declared together before new
scoring. Preparation copies each database, geometry metadata and stage receipts,
and links the unchanged image/mask arrays. Input bank hashes exclude only solver
selection/parent fields; all region data, candidate links, weights and exclusion
constraints must remain identical after solving.

~~~sh
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_costs prepare

for event_arm in division-surcharge reference-events no-division-control
do
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mpf/code/kaggle/ultrack \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m tools.cellpose_ultrack.track \
    --root "work/cellpose-ultrack-event-costs-20260914-$event_arm" \
    --config "configs/cellpose-ultrack-event-costs-20260914-$event_arm.json"
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
    /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.evaluate \
    --root "work/cellpose-ultrack-event-costs-20260914-$event_arm" \
    --config "configs/cellpose-ultrack-event-costs-20260914-$event_arm.json"
done

PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_costs collect
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_report
~~~

The report command updates the existing [Cellpose/ultrack canvas](/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/cellpose-ultrack-20260914.canvas.tsx) with the matched
baseline audit and all three controls. These are development ablations on the
already inspected panel; the no-division arm is a diagnostic, not a division
model. The selected v3 baseline remains unchanged.

Five real-CBC tests check that a second link competes with a new track, that
equal birth/division weights cancel locally, and that an extra division cost
can retain a strong fork while suppressing a weak one:

~~~sh
PYTHONNOUSERSITE=1 PYTHONPATH="/home/mpf/code/kaggle/ultrack:$PWD/work/cellpose-ultrack-20260914/test-deps" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m pytest \
  tests/test_cellpose_ultrack.py -k real_solver_fork -q
~~~

The subsequent temporal-image extension uses the existing opposite-embryo
v4 C4 optical models. It recomputes image embeddings at every current region
centroid using the unchanged physical patch adapter. It evaluates all compatible
unordered daughter pairs in the existing link bank, with no top-k pair cap.
An explicit pair variable requires the actual two outgoing links and contributes
a signed image score. A zero-image control uses the identical pair bank and
constraints. The original generator temperature, synthetic initialization/replay
and historical upstream exposure remain documented; these scores are not
calibrated biological division probabilities.

Pairs are pruned only if they are dominated under both declared objectives.
Dropping the weaker edge and starting the second daughter as a new track must
strictly improve the objective while preserving every selected node and its
downstream path. Full tracking uses the frozen extra division cost of −0.011.
The optical pair-versus-no-fork logit, divided by the existing model temperature,
is clipped to ±8 and multiplied by 0.01. No threshold sweep or new model fitting
is performed.

~~~sh
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.learned_divisions prepare
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  /kaggle/envs/cell-tracking-notebooks/bin/python \
  -m tools.cellpose_ultrack.learned_divisions infer

for pair_arm in pair-control pair-image
do
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mpf/code/kaggle/ultrack \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m tools.cellpose_ultrack.track \
    --root "work/cellpose-ultrack-learned-divisions-20260914-$pair_arm" \
    --config "configs/cellpose-ultrack-learned-divisions-20260914-$pair_arm.json"
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
    /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.evaluate \
    --root "work/cellpose-ultrack-learned-divisions-20260914-$pair_arm" \
    --config "configs/cellpose-ultrack-learned-divisions-20260914-$pair_arm.json"
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
    /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.diagnose_divisions \
    --root "work/cellpose-ultrack-learned-divisions-20260914-$pair_arm"
done
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.learned_results
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_report
~~~

Run the real-CBC test command above with the selector changed to real_solver
to include the four explicit pair-variable tests. Each optical inference also
checks parity with the original six-candidate model API and daughter permutation
invariance. The final collector verifies every exported fork’s exact parent and
two daughters against the scored pair evidence.

The separately declared persistence follow-up adds one selected continuation
edge for each daughter, except at the end of the clip. It is motivated by the
completed optical-arm audit: 9/29 evaluable false divisions have an immediately
terminating daughter; the recovered true event has both daughters continue once.
It reuses the frozen optical evidence and retains a zero-image matched control.
It is an exploratory follow-up on the reused pilot, not an independent test.

The implementation handles anchored windows explicitly: fixed future edges
qualify or reject daughters at the right edge, and daughters of a previously
committed fork must continue at the left edge. A full-graph check rejects any
exported interior daughter that lacks continuation.

~~~sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.branch_audit
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.persistent_divisions prepare
for persistence_arm in pair-control pair-image
do
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/mpf/code/kaggle/ultrack \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /kaggle/envs/cell-tracking-ultrack-v6/bin/python -m tools.cellpose_ultrack.track \
    --root "work/cellpose-ultrack-persistent-divisions-20260914-$persistence_arm" \
    --config "configs/cellpose-ultrack-persistent-divisions-20260914-$persistence_arm.json"
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
    /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.evaluate \
    --root "work/cellpose-ultrack-persistent-divisions-20260914-$persistence_arm" \
    --config "configs/cellpose-ultrack-persistent-divisions-20260914-$persistence_arm.json"
done
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.persistent_divisions collect
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_report
~~~

The real_solver test selector now runs 18 actual CBC fixtures: event costs,
pair identity and bonuses, required selected continuations, missing
continuations, clip endings, and both anchored window boundaries.

The current error audit freshly matches the completed image/persistence,
matched-control, stock and selected-v3 graphs on the same six clips. It verifies
saved scores and input-bank hashes before classifying missed links and endpoint
matching failures. Near-neighbor distances are diagnostic; they do not label an
unmatched object as a false cell.

~~~sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m tools.cellpose_ultrack.analyze_adapted
PYTHONNOUSERSITE=1 /home/mpf/.conda/envs/cell-tracking/bin/python \
  -m tools.cellpose_ultrack.event_report
~~~

See the [current error findings](../../docs/cellpose-ultrack-current-errors-20260914.md)
and compact [audit result](../../results/cellpose-ultrack-persistent-divisions-20260914/error-analysis.json).
