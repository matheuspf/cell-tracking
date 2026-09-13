# Expanded public946 revision 2 runner

The scientific registration remains in
`handover/public946-minimal-generalization-v1/`. This implementation materializes
the pinned public v29 source, never an earlier study's simplified model. It adds
the registered scalar-response, association-probability and graph hooks, and
keeps inference and official evaluation in separate processes.

```sh
bash scripts/run_public946_minimal.sh run --resume --workers 4
```

The default interpreter is the existing
`/kaggle/envs/cell-tracking-notebooks/bin/python`. Set `PUBLIC946_PYTHON` to change
the controller interpreter and pass `--runtime` for its worker interpreter.
`--data`, `--artifacts`, `--archive`, and `--out` accept explicit paths. Outputs
default to `/kaggle/working/cell-tracking/public946-minimal-generalization-v1/revision-2`.
No command installs models, trains networks, publishes notebooks, or submits to Kaggle.

Stages are `preflight`, `audit`, `pilot`, `controls`, `singles`, `combinations`,
`transfers`, `robustness`, `package`, and `report`. `singles --arm E04` resumes a
particular registered arm. Completed results require matching immutable source,
input, model, recipe and output fingerprints. Completed jobs retain their actual
recorded resource allocation when resuming; new jobs use the requested worker
count. Record resource-only scheduler amendments before changing concurrency.

The configured limits are 96 summed worker device-hours, 22 GiB total allocated
GPU memory, 48 GiB new scratch and a 10 GiB filesystem reserve. Four full-cohort
workers each receive a 5.5 GiB allocation cap on the same GPU. The first nine
baseline jobs used two workers at 11 GiB each, before a recorded resource-only
scheduler amendment based on measured memory and utilization. Pilots run
serially. Summed worker wall time is a conservative accounting quantity, including
CPU work; `telemetry.py` separately records sampled per-process SM utilization.
Package wall time, recorded failures and unscored smoke workers also count toward
the conservative device-time budget. The first full baseline clips showed that
retaining every arm's dense matrices would threaten the scratch cap. Full B0 and
all pilot probabilities remain intact; other completed arms retain exact compact
candidate/offset evidence, full-matrix hashes and source/target universes, without
quantization. Compaction journals recover interrupted writes. E01 cannot reuse a
compacted cache. Fresh-finalist DeepCenter maps are clip-local and retain hashes.

## Source-resolved recipes

- B0 retains all original public flags; B1 changes only motion relinking.
- E01 applies atomic degree-preserving, fork-free native-probability vetoes.
- E02 fits three raw-logit values per axis at original peaks, then refines only
  surviving, unrelocated original detections before the public smoother.
- E03 is audited against the actual integer feature coordinates. Its interpolation
  implementation has an exact integer fast path and generic fractional fixtures.
- E04 adds raw-image shift `(0,2,2)`, reflection padding and inverse grid sampling
  at `q+(0,0.5,0.5)`. The unshifted features and public retention guard are retained.
- E05 reflects the full raw X axis and physical/node positions, recomputes the
  existing primary feature TTA and secondary identity features, and averages the
  final normalized parent columns. Detection IDs and coordinates stay fixed.
- E06's actual checkpoint context is two frames. There is exactly one observed
  context containing a given adjacent pair; no five-frame replacement is introduced.
- E07 anchors forks, their predecessors and daughters. Unanchored smoothing uses
  public arithmetic within the nonbranching interior; fork vertices terminate
  support and are excluded from an unanchored node's fit.
- E08 takes the even-count median of the eight inverse-aligned scalar responses
  separately per model, retaining the original feature streams and inter-model mix.

The public harmonic operation combines forward/reverse association evidence.
Secondary associations use the source's calibrated low-margin logit mix;
detections use its aligned linear blend and candidate-retention guard. These
details supersede the planner's tentative source description, without changing B0.

`execution_lock.json` freezes recipes, source semantics, public assets and
scientific code before comparative scoring. Status and decisions are separate.
Correctness failures are retained under `engineering_failures/`; no method is
retuned after a disappointing score. E01 cannot become a novel winner if the
full graphs prove identical to known B1.

## Validation

```sh
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /kaggle/envs/cell-tracking-notebooks/bin/python -m unittest discover \
  -s tests -p 'test_public946_minimal*.py' -v
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/public946-minimal-generalization-v1 -p 'test_*.py' -v
```

All-199 scoring uses fresh per-arm official matching, run-level weighted
aggregation, division micro-aggregation, and a separate original-export score.
Every delivered graph uses natural rounding and identical bounds clipping.
The worker's `original_nodes` array retains rounded coordinates before clamping;
the separate evaluator reconstructs v29's actual CSV `max(0, round(value))`
before scoring the original export. Its upper-bound violations remain visible.
Historical half-tie lookup corrections are never imported into inference.

Standalone notebooks are ignored outputs under `packages/`, accompanied by
public input dependencies, Python exports, diffs and hashes. Each actual notebook
is executed on the four full metadata-selected pilots. Full scientific-worker
execution and actual notebook execution scopes are stated separately. These
reused local embryos cannot establish clean generalization or a new LB score.
