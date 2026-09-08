# Interactive competition field guide

An offline guide for an experienced Kaggler who is new to Biohub cell tracking.
Seven chapters cover the task, actual microscopy, lineage graphs, scoring,
embryo validation, notebook submission, and sourced forum context. The page has
no external scripts, fonts, image requests, or runtime dependencies.

The generated page is `/kaggle/working/cell-tracking/guide/index.html`.
Open it directly in a browser, or serve it locally:

```sh
conda activate cell-tracking
PYTHONNOUSERSITE=1 python -m http.server 8765 --bind 127.0.0.1 \
  --directory /kaggle/working/cell-tracking/guide
```

Then visit <http://localhost:8765>.

## Rebuild

The tracked source is `template.html`; the builder embeds a small real-data
preview into that template. Once the canonical data is extracted:

```sh
PYTHONNOUSERSITE=1 python tools/build_competition_guide.py
```

The initial build used a CRC-verified excerpt of the in-progress local archive,
stored under ignored `work/guide-inputs`, without modifying the download:

```sh
PYTHONNOUSERSITE=1 python tools/build_competition_guide.py \
  --data-root work/guide-inputs
```

That cache contains only the selected training sample's GEFF graph, image
metadata, image chunks for t=0,20,40,60,80,99, and sample CSV. Missing image chunks
are explicitly rejected, so Zarr fill values cannot masquerade as a real frame.
The preview stores every fourth Z plane and full-depth max projections with
fixed 1st–99.9th percentile display normalization. Its JPEG images are for
visualization; they are not model inputs. All raw data and generated HTML stay
outside Git.

## Evidence and maintenance

- Official pages were refreshed on 2026-09-08 with the repository extractor.
- Clip/embryo counts come from `work/data-api-files.json`, or the complete
  extracted canonical image directory if that inventory is unavailable.
- Actual array and graph metadata, node arrays, edges, and the CSV header were
  read from the selected input files.
- Organizer metric/baseline source is pinned to
  `075fc5f5a52d11077f9dc2b074644618f26939e2` (current when checked).
- Cached forum bodies were read, including host/staff patch and rescore notices,
  the two-embryo clarification, and recent participant count-metric discussions.
- Recommendations and simulated scenarios are labeled separately from evidence.
  The arithmetic sandbox is not an implementation of the official graph scorer.

Time-sensitive editorial facts in the template are deliberately dated. Rebuilding
images does not re-verify those facts; refresh the relevant reference sources and
review the prose before changing the date or metric pin.

Validation: Python lint; Chromium interaction checks for chapter navigation,
microscopy controls, graph scenarios, score arithmetic and zero-event input,
physical-distance gates, embryo splits, runtime estimates, quiz feedback, CSV
download, checkpoint persistence, mobile overflow, and offline loading.
