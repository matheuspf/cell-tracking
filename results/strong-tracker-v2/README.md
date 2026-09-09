# Strong tracker v2: completed measured study

V200–V270 are complete. Decision: **significant_local_gain**. Bypassing the original motion relinker, with strict image bounds, raises pooled local score from **0.911774 to 0.934206 (+0.022432)**. Both embryo directions improve: **44b6 +0.019091; 6bba +0.022928**.

This is an exploratory result on reused embryos with contaminated public upstream checkpoints. The winner was selected after inspection; it is not a hidden-test forecast or an increment to the historical 0.946 leaderboard score.

- [Measured report](v2_report.md)
- [Self-contained offline dashboard](dashboard.html) — download/open in a browser; all 104 settings, both embryos, failure census and training curves
- [All aggregate operating points](operating_points.csv)
- [Validation receipt](validation_receipt.json)
- [Implementation and reproduction](../../docs/strong-tracker-v2.md)

The full local artifact root is `/kaggle/working/cell-tracking/strong-tracker-v2/`. It includes all 20,696 per-sample score rows, detailed failure censuses, 199 selected prediction graphs, frozen models, resumable checkpoints, logs, source snapshots and the complete artifact manifest. `seconds` in operating points is summed evaluator time; cached repair timings are separately labelled in `runtime_summary.json`. Neural inference was reused, so these are not fresh full-notebook or hidden-set runtime measurements.

Only code/configuration and aggregate evidence are committed here. Detailed GT identifiers, coordinates, microscopy, model weights and predictions remain local. The v1 store and unrelated work are preserved. Copy the full v2 output root, ignored `work/strong-tracker-v2/`, and this Git commit out before destroying the Vast instance; reproduction also requires the existing v1 artifacts, official evaluator and competition inputs.
