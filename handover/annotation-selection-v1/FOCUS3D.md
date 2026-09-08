# FOCUS-3D follow-up: evidence and next experiment

Verified **2026-09-08 UTC**. This is a research handover, separate from the
completed annotation-selection v1 experiment. **No FOCUS-3D installation, model
download, inference, pseudo-label generation, or score-gain measurement was
performed. Its improvement over our public baseline remains unknown.**

## What is available

The [official project](https://github.com/yu-lab-vt/FOCUS-3D) segments individual
cells in 3D fluorescence images. The [online Space](https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D)
returns an instance-label volume; it does not provide temporal cell identities
or division edges. A local integration therefore needs **per-frame masks →
centroids → the existing temporal tracker and division logic**. Better cell
separation and localization are hypotheses to test, not established benefits.

Source versions inspected:

- GitHub: `5c4b53f743a0fbbae056e2c1a139895ae819f069`.
- Hugging Face model: `115258efcc9ee44e69db3902bce2511d0ae24e2f`.
- Hugging Face Space: `bdff0d36e44c58959f90e3cada03b82f9ada06fa`.

The repository's [root license](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/LICENSE)
is BSD-3-Clause; the [nested segmentation implementation](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/src/focus3d/segmentation/FOCUS3D/LICENSE)
has an MIT license. The [model card](https://huggingface.co/Qinghua-thu/FOCUS-3D)
states Apache-2.0 for the weights. Preserve these notices when packaging code
or weights. The model API reports automatic gating (`gated: auto`); downloading
requires Hugging Face sign-in and agreement to share contact information.
Access acceptance was not performed in this study.

The [model file inventory](https://huggingface.co/api/models/Qinghua-thu/FOCUS-3D/tree/main)
lists `model_final.pth` and `model_final_nuclei.pth`, each **4,468,804,784 bytes
(about 4.47 GB)**, plus a membrane checkpoint. Start by evaluating the nuclei
variant for nuclear fluorescence. Do not assume that the general and nuclei
files are identical merely because their byte sizes match. Training-data
overlap with Biohub embryos is **unknown**; generalization claims require an
actual provenance audit.

## Adapter and runtime findings

The [Space input reader](https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D/blob/bdff0d36e44c58959f90e3cada03b82f9ada06fa/backend/io.py)
expects a single-channel grayscale **ZYX** TIFF. Competition clips inspected
in v1 are **TZYX**, with shape `(100, 64, 256, 256)` and voxel sizes
`(1.625, 0.40625, 0.40625)` µm. Read the actual metadata for each input, process
one time point at a time, and start from Z/XY ratio **4.0**. Preserve coordinate
order and physical units when converting restored masks into centers.

The [upstream inference implementation](https://github.com/yu-lab-vt/FOCUS-3D/blob/5c4b53f743a0fbbae056e2c1a139895ae819f069/src/focus3d/segmentation/FOCUS3D/inference.py)
interprets cell radius in original XY pixels. Its reference radius is 15 pixels:
XY target size is approximately `original_size * 15 / cell_radius`. With
`z_ratio > 1`, this resampling leaves Z unchanged; it does not automatically
make the volume isotropic. Labels are restored to the original shape using
nearest-neighbor interpolation. Small radius settings can enlarge the XY
volume substantially. Measure radius from source images and record the choice
before examining validation outcomes.

That Linux `infer_volume` implementation builds and loads the predictor on
each call. A naive 100-frame loop would repeatedly load a large model. The
[Space adapter](https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D/blob/bdff0d36e44c58959f90e3cada03b82f9ada06fa/backend/inference.py)
instead initializes models once and passes a cached model into its
`inference_win` runtime. Check the chosen runtime's API and cache one selected
predictor; establish output parity before changing inference plumbing. The
Space uses patch batch size 12. **Batch size 1 on the RTX 4090 is a plausible
pilot configuration, not a verified fit or throughput estimate.** Measure
startup separately from warmed inference, peak VRAM/RAM, stitching time and
complete per-clip runtime. The Space's remote hardware and time allocation do
not establish local or Kaggle feasibility.

## Direct inference versus a teacher

The earlier conversation inspected [community discussion 738217](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217):
a participant reported direct FOCUS inference timing out, and the discussion
suggested using it as a teacher. Treat this as **community experience**, not an
organizer ruling or our benchmark. The local forum index lists the topic, but
its full body was not present when this handover was written; retrieve it with
the repository's throttled forum extractor before relying on finer details.

A proposed alternative is **offline FOCUS masks → dense pseudo-labels → a
smaller point detector → existing tracking/division detection**. Pseudo-labels
can contain false splits, merges and missing cells; sparse GEFF annotations
are not an exhaustive true-cell denominator. Preserve original annotations
and quantify pseudo-label quality independently. Direct inference and teacher
distillation both need measurement; neither has a demonstrated score uplift.

The [official code requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements)
in the local 2026-09-08 snapshot allow publicly available pretrained models,
disable internet and cap CPU/GPU notebooks at **12 hours**. The
[competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
also require reasonable external-resource accessibility. Recheck these before
submission packaging, including how the checkpoint's access conditions apply.
A submission would require locally packaged weights and dependencies; calling
the hosted Space during submission inference is incompatible with offline
execution.

## Next-agent checklist

1. Read the [completed v1 results](../../results/annotation-selection-v1/README.md)
   and [reproduction notes](../../docs/annotation-selection-v1.md). The small
   classical-baseline gain did not transfer to Harmonic Fusion. Do not add it
   to a public leaderboard score.
2. Create a **new experiment version and isolated environment**, retaining
   exact source/checkpoint hashes and dependency pins. The current request
   authorizes preserving findings on the branch; this document does not
   initiate a new FOCUS run, accept gated-access conditions, upload microscopy
   data to the Space, or authorize a Kaggle submission.
3. For a follow-up pilot, freeze representative source-only
   frames, radius/normalization settings, resource limits and stop criteria.
   Obtain authorized checkpoint access; keep downloads and derived masks in
   ignored `work/` or `/kaggle/working/cell-tracking/`, outside Git. Benchmark
   one cached model at batch size 1 and verify coordinate/shape restoration.
4. Compare matched centers, localization, splits/merges, count ratio and
   end-to-end official graph/division score against the **same** frozen
   baseline and clips. Report runtime and failures alongside quality. Use the
   timing evidence to choose direct inference or a teacher experiment.
5. Keep v1 models, labels, splits, prediction locks and results immutable. Its
   outer outcomes have already been seen: do not retune v1 or call those clips
   untouched validation. Establish a new validation design with explicit
   embryo overlap and checkpoint-provenance limitations; use genuinely
   untouched validation where available. Preserve all existing data,
   environments, notebook originals and unrelated studies' work.
