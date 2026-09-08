# Start here: completed experiment and follow-up findings

The user requested actual local execution of `CODEX_PROMPT.md`, then asked how
much it improves over the best downloaded public notebook, whether FOCUS-3D can
help, and finally asked to upload all findings to this branch. **The v1 local
experiment is complete, including its report and offline dashboard.** Its
implementation and initial measured result are in commit `21cd0d7`. Do not
restart the original prompt or present its historical plan as unfinished work.

## Read these artifacts

- [Measured results and public-baseline answer](../../results/annotation-selection-v1/README.md).
- [Full final report](../../results/annotation-selection-v1/report.md).
- [Offline HTML dashboard](../../results/annotation-selection-v1/dashboard.html):
  open the downloaded file in a browser; it embeds all chart data and images.
  GitHub displays its source rather than executing it.
- [FOCUS-3D technical assessment, sources and follow-up](FOCUS3D.md).
- [Reproduction and implementation](../../docs/annotation-selection-v1.md).
- [Portable artifact hashes](../../results/annotation-selection-v1/bundle_manifest.json)
  and [local-only evidence inventory](../../results/annotation-selection-v1/LOCAL_ARTIFACTS.md).

## What the results mean

All 199 clips were evaluated in both held-out embryo directions: 71 from 44b6
and 128 from 6bba. The preregistered primary rule is a seven-leaf boosted tree
using all allowed features, coherent tracklet ranking by the 0.9 quantile,
and requested 90% node retention. On the provenance-clean classical baseline,
pooled official local score rose **0.674116 → 0.676907 (+0.002791)**. Both
directions improved: +0.012257 and +0.001056. All 199 prediction files reproduced
byte-for-byte with annotation-file access blocked before outer evaluation.

The measured stronger-public-baseline answer is **no demonstrated gain**.
Harmonic Fusion v29 is tied at **0.946** with 942 TTA v1 in the downloaded
public-score snapshot. Only Harmonic Fusion was run across all 199 local clips.
Its checkpoint provenance uses supplied embryos, so this is a contaminated
diagnostic lane. The frozen primary filter **reduced** its pooled local score
**0.911774 → 0.873322 (−0.038452)**. None of the 72 learned/confidence settings
beat identity in pooled local score. Even the best hindsight setting scored
0.911222 (−0.000552). These local scores and the 0.946 public leaderboard score
use different evaluation populations; do not add the classical gain to 0.946.

The clean gain is mostly count adjustment (+0.007121), partly offset by graph
damage (−0.004331). It exceeds exact-cost confidence control by +0.005636 pooled,
but does not identify latent annotator preference among true cells. There are
only two embryos, one conservative overlap group per embryo and 73 confirmed
overlapping clip pairs. Independent source inner tuning and informative
within-embryo bootstrap intervals were unavailable; fixed preregistered settings
were used. The decision remains **promising_but_uncertain**. Hindsight oracles
are diagnostic headroom, not deployable methods or forecasts.

There are 133,318 annotated observations, 128,883 annotated edges, 151 divisions
and 4,725,117 supplied estimated observations. Their ratio is 2.8215%; the mean
per-clip ratio is 6.1261%. Overlapping observations are not unique cells. True
all-cell prevalence remains unknown. The two 48-item blinded audit/census packs
exist locally with manual fields blank; no human labels were fabricated.

FOCUS-3D was assessed from its code, model metadata and competition rules.
**No FOCUS model was downloaded or run; its uplift and local runtime remain
unknown.** It returns per-frame 3D masks and needs a centroid/tracker adapter.
The nuclei checkpoint is about 4.47 GB with automatic Hugging Face access
gating. The practical follow-up is a bounded local cached-model benchmark,
then direct inference or pseudo-label teacher evaluation as timing permits.
The Space is a reference implementation, not a measured extension of v1.

## Existing execution and preservation boundaries

Raw data, notebook originals, existing environments and unrelated user work
were preserved. A dedicated study environment inherits notebook packages:

- Data: `/kaggle/input/competitions/biohub-cell-tracking-during-development`.
- Study Python: `/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python`.
- Notebook runtime: `/kaggle/envs/cell-tracking-notebooks`.
- Inspection Python: `/home/mpf/.conda/envs/cell-tracking/bin/python`.
- Sealed outputs: `/kaggle/working/cell-tracking/annotation-selection-v1`.
- Working files/logs: `work/annotation-selection-v1/`.
- Notebook mirrors: `notebooks/downloaded/`; selected versions and input
  references are recorded in `configs/notebooks.json`.

The sealed manifest records 11,014 files and 30.72 GiB, including 26.52 GiB of
patch pixels. Those artifacts, raw annotations, crop viewers and model weights
are not in public Git. A fresh checkout has complete code and aggregate evidence;
reproduction still requires the existing local input/model stores or separately
obtained competition inputs. The portable bundle was exported from verified
sealed source hashes without changing the source artifact store.

Original validation passed 29 handover helper tests, 22 study tests, 7 inspection
tests (2 optional skips), and 102 upstream metric fixtures. Offline dashboard
validation exercised 36 lane/embryo/model/policy combinations and both audit
viewers. The portable dashboard has its own publication validation receipt;
the original validation receipt hashes the original HTML. The image probe ran
eight small CUDA fits on the RTX 4090, 30,000 source candidates per fit, six
epochs, two seeds: 8.13 GPU-synchronized seconds total, about 123 MiB peak
allocated fit VRAM. This is a bounded probe, not FOCUS training throughput.

Public notebook scheduling required four independent shards. The preserved
adapter retained 27 notebook function bodies byte-for-byte; two independent
pilot graphs and 48 completed serial-prefix graphs matched exactly. One partial
terminal graph was excluded from the parity comparison, not from the 199-clip
score. Six original centers in five clips lie one voxel beyond the image Z
boundary: original graph geometry was retained and feature sampling reflected
at borders. Detector confidence was missing for 383,339 public nodes (9.27%)
and explicitly filled with zero plus a missingness flag. Failures and repair
receipts remain local and are described in the final report.

No Kaggle submission, notebook publication, forum post, remote model inference,
paid API call or hardware rental was performed. The current user explicitly
requested committing and pushing these findings to the existing branch.
Unrelated changes to root `AGENTS.md`, root `README.md`, `Untitled`, the external
data guide/handover/tools, and its resume script belong to user work and were
excluded from this findings commit.

Any follow-up must use a new experiment version and acknowledge that these
outer outcomes have been inspected. Preserve v1 labels, models, prediction
locks and outputs. Do not retune against these outcomes and call them untouched
validation. See FOCUS3D.md for source pins and the concrete adapter/runtime
questions the next experiment should resolve.
