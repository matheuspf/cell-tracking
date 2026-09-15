# Incumbent comparison — 14 September 2026

Updated: 2026-09-15T03:20:43.667119+00:00

Open the ordinary HTML report at http://127.0.0.1:8770/report.html. It refreshes progress while the local server runs; the saved report also opens directly in a browser.

## Released validation replay

The released seed-314159 checkpoint scores **0.9779740160**, versus published **0.9779747766** (difference -0.0000007606). This is near-exact, not bit-exact reproduction.

Its 200-batch prefix covers 1,600 eligible two-frame windows from 19 of the 40 listed monitoring clips. Detection recall is 9,024/9,226 = 97.810535%, exactly matching the published recall. Edge accuracy is 3,864,585/3,865,104 = 0.9998657216.

The complete 40-clip monitor gives **0.9753165131**, recall 97.544632% and edge accuracy 0.9998669216. The cap and complete-monitor scores must not be mixed.

These are the upstream edge-classification-accuracy × detected-node-recall proxy, using greedy 5 µm matching and the original proposal rules. They are **not the competition score**. The monitor is fully included in the 199-clip training set and contains only embryo 44b6. Pair windows count frames repeatedly; these denominators are not unique cells.

## Detector comparison

All rows below use the same 400 assessment frames, 40 clips and 2,383 annotations, native integer coordinates, and optimal one-to-one matching. The HTML includes every radius from 1–7 µm, both embryos separately, and close-pair counts.

- **Cellpose ViT-B**: recall at 1/2/3/7 µm = 21.36% / 63.58% / 71.76% / 95.89%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 11/12.
- **Refiner · source only · 20260914**: recall at 1/2/3/7 µm = 23.12% / 62.44% / 74.99% / 95.89%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 11/12.
- **Refiner · source only · 314159**: recall at 1/2/3/7 µm = 24.76% / 64.92% / 76.12% / 95.76%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 12/12.
- **Constant offset · source only**: recall at 1/2/3/7 µm = 28.79% / 69.79% / 77.51% / 96.01%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 12/12.
- **Public incumbent ensemble**: recall at 1/2/3/7 µm = 26.86% / 71.84% / 85.82% / 99.50%; 116,533 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 11/12.
- **Refiner · both embryos · 20260914**: recall at 1/2/3/7 µm = 64.04% / 85.02% / 87.75% / 96.06%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 12/12.
- **Refiner · both embryos · 314159**: recall at 1/2/3/7 µm = 64.33% / 85.27% / 87.87% / 96.06%; 156,226 candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: 12/12.

Both-embryo heads retain the original 3,000-update query-training recipe and two fixed seeds. They use 2,268 natural queries and 27,088 jitter/center queries across the 199-clip population. This matches the incumbent’s eligible clip population, **not its exact temporal sampling, observation count, optimization budget, architecture, or pretraining**. These measurements quantify training-exposure effects; they do not prove generalization.

The residual head keeps proposal counts/confidence unchanged and can move an integer center by at most 3 µm. Its headroom is therefore smaller than that of a new detector that can add, split, or remove cells.

The exposure comparison is large: seed 20260914 gains 12.76 percentage points at 3 µm when both embryos are used for fitting; seed 314159 gains 11.75 percentage points at 3 µm when both embryos are used for fitting; both remain below the incumbent at 7 µm. This supports further work on transfer and missed-cell coverage; it does not establish a superior detector for unseen embryos.

## Full competition-score experiment

Nine methods are registered on six complete 100-frame clips: incumbent, Cellpose, source-only constant offset, both source-only refiner seeds, both both-embryo refiner seeds, incumbent + Cellpose, and incumbent + source-only refiner seed 20260914. Unions use the previously fixed 2 µm suppression radius.

Every method uses fresh native features at its own predicted centers and the same public primary association checkpoint. The upstream greedy linker uses source-normalized softmax, probability >0.5, at most one parent and two children, and no motion gate. Node counts follow the detector; they are not forced equal. All graphs are frozen before evaluation with the pinned official metric. This is a controlled native pipeline, not a reproduction of the full production Harmonic pipeline.

- incumbent: official local score **0.92274973**, 92,475 nodes, 3871/3996 available GT edges linked.
- cellpose: official local score **0.81337339**, 111,082 nodes, 3510/3859 available GT edges linked.
- constant: official local score **0.81744035**, 111,082 nodes, 3526/3874 available GT edges linked.
- source-20260914: official local score **0.79425062**, 111,082 nodes, 3484/3883 available GT edges linked.
- source-314159: official local score **0.80051184**, 111,082 nodes, 3535/3889 available GT edges linked.
- both-20260914: official local score **0.80939383**, 111,082 nodes, 3563/3879 available GT edges linked.
- both-314159: official local score **0.81479055**, 111,082 nodes, 3580/3878 available GT edges linked.
- incumbent+cellpose: official local score **0.63825793**, 136,847 nodes, 3226/4007 available GT edges linked.
- incumbent+source-20260914: official local score **0.60722698**, 136,300 nodes, 3134/4008 available GT edges linked.

The six clips and public association checkpoint are training-exposed. This stage tests score changes and extra-proposal tradeoffs on a fixed downstream pipeline; it is not an independent generalization claim.

## Native training with each embryo excluded

Two whole-model fits start from random initialization: 44b6 → 6bba (71 source clips, 6,315 windows), and 6bba → 44b6 (128 source clips, 12,392 windows). Each retains the native batch size 8, AdamW 1e-4, original losses and brightness/flip augmentations. The fixed final epoch is 400. Target evaluation waits until both final checkpoints exist.

Source-only workers reject reads of the other embryo and inherited checkpoints. Input and source hashes are frozen. Cached normalized FP32 images reproduce the original augmented half-precision storage round trip exactly on the checked windows. Runs save model, optimizer and RNG state, and release the GPU every eight updates.

- 44b6 → 6bba: **running**, epoch 17/400, 13,048 updates, 3,264/6,315 windows in current epoch; peak reserved 13.41 GiB. Last progress 2026-09-15T03:20:39.355423+00:00.
- 6bba → 44b6: **running**, epoch 9/400, 13,048 updates, 5,248/12,392 windows in current epoch; peak reserved 15.09 GiB. Last progress 2026-09-15T03:20:32.935408+00:00.

The full campaign currently projects roughly **8.5 GPU-days remaining**, based on the latest batches. This is a throughput estimate, not a promised finish time; contention adds wall time. Both source-only fits have saved resumable checkpoints.

Only two independent public embryos exist. Report both directions and paired per-clip differences; repeated/overlapping crops and previously inspected panels cannot establish broad biological generalization.

## Reproducibility

- Plans and receipts: `results/incumbent-comparison-20260914/`.
- Heavy artifacts, checkpoints and logs: `work/incumbent-comparison-20260914/`.
- Resume native fits with the notebook runtime: `python -m tools.incumbent_comparison.train_native run --source 44b6` (or `6bba`). Per-source locks prevent duplicate workers.
- Resume complete-clip stages: `python -m tools.incumbent_comparison.run_full`. Failed stages are logged; completed banks are verified and reused.
- Serve/update report: `python -m tools.incumbent_comparison.report --serve --port 8770`.
- The source-only native target inference/evaluation stage is deferred until both 400-epoch fits finish; it is not currently an automatically scheduled worker.
