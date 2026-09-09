# External datasets: branch handover

This directory is the self-contained information package for an agent deciding
whether and how to use the external resources downloaded from Kaggle
[discussion 732103](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103).
It records the September 8, 2026 data snapshot and the current tracker at
`f67ba60` on `handover/strong-tracker-v3`, documented September 9.
**No training strategy or dataset selection is made here. No external-data
training or transfer evaluation was performed in this preparation work.**

Everything needed to understand the inventory, labels, preparation and measured
branch results is committed. A fresh checkout does **not** contain the original
datasets, generated arrays, model weights, local HTML presentation or private
conversation. Historical `work/` paths identify artifacts on the preparation
machine; their presence is not assumed on another machine.

## Contents

- [Dataset descriptions](DATASETS.md): every downloaded collection, counts,
  label meaning, source quality, overlap, units, source terms and inspection gaps.
- [Current branch context](BRANCH_CONTEXT.md): selected v3 model, actual results,
  failed controls, feature/crop interfaces and existing validation limitations.
- [Access and preparation](ACCESS.md): public retrieval routes, failed automatic
  access and manual alternatives, exact preparation commands, file schemas and
  what the code does and does not implement.
- [Machine-readable inventory](dataset_inventory.json): all six Zoo exports,
  all seven RIKEN archives, both synthetic collections, original notebook
  identities, file sizes, SHA-256 receipts, array metadata and feature definitions.
- [Verification summary](verification_summary.json): recorded download,
  coordinate, graph, source-CSV, image-alignment and presentation checks, with
  their scope. These checks establish preparation integrity, not model accuracy.

## What exists

The synthetic release contains **1,539 static images with 423,853 centers** and
**2,174 six-frame sequences with 4,056,226 nodes, 3,460,295 edges and 165,267
division parents**. All 3,713 examples have prepared labels. Images and labels
are paired, but the sequence image grid differs from the original coordinate
grid and source `track_id` denotes a clone shared by daughters.

The **six Virtual Embryo Zoo exports** contain experimental tracking results for
zebrafish, fly, mouse, ascidian, worm and beetle. All have prepared graphs. No
paired microscopy was downloaded; physical coordinate scale and frame duration
remain unverified. Base ZIPs, enriched stores and the two extra CSVs include
overlapping representations of the same tracks.

The **seven RIKEN/SSBD zebrafish archives** contain HDF5/XML measurements. All
passed archive integrity and published checksum checks. Only animal C received a
full structural/count audit: 821 frames, 3,447,269 point measurements and 14
feature definitions. Its prepared excerpt is ten frames / 1,312 points. No
explicit temporal or parent-child links were found in that inspected sample.
The other six archives have not had the same label audit.

None of these downloads supplies dense cell masks. The original storage total,
**37,743 verified files / 30,877,642,679 bytes**, includes duplicate
representations and the locally reconstructed beetle ZIP; it is not unique
scientific data volume. The prepared directory contained 3,725 files /
394,701,884 bytes at the recorded verification.

## Relation to the solution in this branch

The selected policy is **v3 `A_residual_m3.0`**, with local pooled score
**0.934802374260586**, compared with v2's **0.9342063149703403**.
These are repeated, operational exploratory evaluations of 199 supplied clips
from two embryos, with documented upstream label exposure; they are not an
independent transfer test or leaderboard result.
[BRANCH_CONTEXT.md](BRANCH_CONTEXT.md) connects each available label type to the
actual inputs the current models consume and identifies missing interfaces.
That factual comparison leaves all model, training, data-selection and future
evaluation decisions to the next agent.

## Code and artifact locations

The committed [NumPy reader](../../tools/biohub_external_data/data_adapter.py),
[preparation code](../../tools/biohub_external_data/prepare_data.py),
[data verifier](../../tools/biohub_external_data/verify_data.py),
[motion audit](../../tools/biohub_external_data/audit_synthetic_motion.py),
[inventory compiler](../../tools/biohub_external_data/build_inventory.py) and
[dependency pins](../../tools/biohub_external_data/requirements.txt) describe and
reproduce the implemented conversion. Instructions and prerequisites are in
[ACCESS.md](ACCESS.md).

On the original machine, source downloads reside in
`work/biohub-forum-archive/`, and derived labels, checks and the illustrated HTML
guide reside in `work/biohub-data-guide/`. The HTML contains microscopy slices,
point overlays, a synthetic fork, Zoo trajectories and RIKEN measurements. It is
an optional local artifact; this handover does not depend on opening it. Its
earlier experimental suggestions are not decisions for this branch.
