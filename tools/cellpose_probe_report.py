"""Build a portable browser report and numerical Markdown evaluation."""
from __future__ import annotations

import json
from statistics import mean

from tools.cellpose_models_probe import REPO, WORK, OUT, read, write, sha

LABELS = {
    'incumbent':'Incumbent', 'cpdino-vitb':'Cellpose DINO-B', 'cpdino':'Cellpose DINO-L',
    'cpsam_v2':'Cellpose SAMv2', 'vitb+vitl':'DINO-B + DINO-L', 'vitb+sam':'DINO-B + SAMv2',
    'all_cellpose':'All three Cellpose', 'incumbent+vitb':'Incumbent + DINO-B',
    'incumbent+all_cellpose':'Incumbent + all Cellpose'}


def pct(n, d):
    return f'{100*n/d:.2f}%' if d else 'not estimable'


def main():
    metrics=read(OUT/'pilot-metrics.json')
    plan=read(OUT/'embedding-plan.json')
    audit=read(REPO/'results/incumbent-provenance-20260914/audit.json')
    pooled={r['method']:r for r in metrics['groups'] if r['embryo']=='pooled'}
    runtime=[]
    for model in ('cpdino','cpsam_v2'):
        receipts=[]
        for folder in (WORK,WORK/'adjacent'):
            receipts.extend(read(p) for p in sorted((folder/'predictions'/model).glob('*.json')))
        assert len(receipts)==24
        runtime.append({'method':model,'frames':len(receipts),
                        'mean_seconds':mean(r['processing_seconds_excluding_queue'] for r in receipts),
                        'total_seconds':sum(r['processing_seconds_excluding_queue'] for r in receipts),
                        'peak_reserved_gib':max(r['gpu_peak_reserved_bytes'] for r in receipts)/2**30})
    payload={'metrics':metrics,'audit':audit,'runtime':runtime,'labels':LABELS,
             'embedding_plan_sha256':sha(OUT/'embedding-plan.json'),'metrics_sha256':sha(OUT/'pilot-metrics.json')}
    write(OUT/'report-data.json',payload)
    html=(REPO/'tools/cellpose_probe_report.html').read_text()
    html=html.replace('__REPORT_DATA__',json.dumps(payload,allow_nan=False).replace('<','\\u003c'))
    (OUT/'report.html').write_text(html)
    base=pooled['cpdino-vitb']; inc=pooled['incumbent']; large=pooled['cpdino']; combined=pooled['all_cellpose']; inc_union=pooled['incumbent+all_cellpose']
    lines=['# Additional Cellpose models and native embeddings', '',
        'The checkpoint-linked provenance audit confirms overlap between the incumbent secondary model’s published training list and every local assessment clip. This establishes training exposure in our benchmark. No hidden competition-test label leak was established. See the [full audit](incumbent-provenance-20260914.md).', '',
        f'**Evaluation:** {base["frames"]} volumes, 12 adjacent pairs (25→26 and 75→76), six clips, two embryos, {base["gt_nodes"]} annotated nodes. Predictions, proposal banks and native scores were frozen before this evaluation. This is an exploratory pilot on previously inspected public training clips. **No complete-clip competition score is measured here.**', '',
        'Open [the standalone HTML report](../results/cellpose-embedding-probe-20260914/report.html) in a browser. It includes embryo filters, detector curves, association controls, counts and downloadable evidence. It needs no TSX or JavaScript build.', '',
        '## Findings and decision', '',
        f'- **DINO-L is a modest detector improvement in this pilot:** R3 rises from {pct(base["matches"]["3"],base["gt_nodes"])} to {pct(large["matches"]["3"],large["gt_nodes"])} and R7 from {pct(base["matches"]["7"],base["gt_nodes"])} to {pct(large["matches"]["7"],large["gt_nodes"])}. Candidates fall from {base["candidates"]:,} to {large["candidates"]:,}. Native-image parent retrieval rises by only one edge, {base["association"]["ensemble_image"]["correct_parent_top1"]}/88 to {large["association"]["ensemble_image"]["correct_parent_top1"]}/88. This is not convincing evidence of a tracking-score gain.',
        f'- **Naive unions improve tight recall but hurt association ranking:** the three-Cellpose union reaches R3 {pct(combined["matches"]["3"],combined["gt_nodes"])}, yet image parent retrieval falls to {combined["association"]["ensemble_image"]["correct_parent_top1"]}/88, versus {base["association"]["ensemble_image"]["correct_parent_top1"]}/88 with DINO-B. Incumbent plus all Cellpose raises R3 from {pct(inc["matches"]["3"],inc["gt_nodes"])} to {pct(inc_union["matches"]["3"],inc_union["gt_nodes"])}, but image parent retrieval falls from {inc["association"]["ensemble_image"]["correct_parent_top1"]}/88 to {inc_union["association"]["ensemble_image"]["correct_parent_top1"]}/88. R7 stays at 100% and candidates rise {inc["candidates"]:,} → {inc_union["candidates"]:,}.',
        '- **The native model can embed the new detections, but its image features do not consistently beat geometry.** On DINO-B, geometry gets 82/88 parents correct and the image ensemble gets 81/88; on DINO-L the figures are 80/88 and 82/88. On SAMv2 they are 83/88 and 80/88. These controls provide no broad association improvement from simply swapping proposal generators.',
        '- **Keep DINO-L as a candidate for a wider detector test; do not promote these unions from this pilot.** More proposals are allowed and can help. They need learned selection or compatible instance-level association, followed by a fixed complete-clip scorer comparison. This experiment does not establish that increasing proposals must worsen the final metric.',
        '- **There are no annotated close pairs or divisions in these selected frames.** Small-radius recall and nearby proposal competition are measured, but close-cell separation and division accuracy remain untested here. The earlier 400-frame panel and a division-focused, independently frozen panel are needed to cover these failure modes.', '',
        '## Detector comparison on the identical 24 frames', '',
        'Recall uses integer submission coordinates and the official distance matcher at each radius. The counts below are matched annotations / total annotations. Unannotated proposals remain competitors; sparse labels do not supply a conventional false-positive rate.', '']
    for method,r in pooled.items():
        rec='; '.join(f'{k} µm: **{pct(r["matches"][str(k)],r["gt_nodes"])}** ({r["matches"][str(k)]}/{r["gt_nodes"]})' for k in (1,2,3,5,7))
        lines.append(f'- **{LABELS[method]}**, {r["candidates"]:,} candidates: {rec}.')
    lines += ['', 'The union rule preserves the base bank and appends alternate-model points farther than 2 µm from every retained center. Within an added bank, original confidence determines order. No cross-model confidence calibration or GT-based filtering is used. Counts are deliberately allowed to change.', '',
        '## Association with the same proposal banks', '',
        'The existing TemporalUNet encoder produces 32-dimensional image features at every new center. Each bank is passed separately through the trained node transformer with its own candidate context. Geometry-only scores use negative physical distance. The zero-image control preserves the trained transformer, positions and coordinates. Its distribution shift means it is an ablation, not a retrained geometry model.', '',
        'The primary result below is **correct top-1 parent / all eligible GT edges**: missed endpoints count as failures. Conditional parent accuracy is also shown among edges whose two endpoints were matched within 7 µm. This is a ranking diagnostic, not a globally consistent tracking graph or a competition score. A fixed 15 µm motion gate applies to all controls. The ensemble uses the notebook’s calibrated low-margin blend: secondary weight at most 0.15, margin 0.35, temperature 1.', '']
    for method,r in pooled.items():
        summaries=[]
        for name,title in [('distance','geometry'),('ensemble_zero','zero image'),('ensemble_image','image embeddings')]:
            a=r['association'][name]
            summaries.append(f'{title}: **{a["correct_parent_top1"]}/{a["eligible_gt_edges"]} ({pct(a["correct_parent_top1"],a["eligible_gt_edges"])})**; conditional {pct(a["correct_parent_top1"],a["supported_all_edges"])} on {a["supported_all_edges"]} available edges')
        lines.append(f'- **{LABELS[method]}** — '+'. '.join(summaries)+'.')
    lines += ['', '## Tight localization and ambiguous neighborhoods', '',
        'A close annotated pair has centers separated by at most 7 µm. Both must receive distinct detections under the official matching. These annotations are sparse, so this does not enumerate every crowded region.', '']
    for method,r in pooled.items():
        c=r['crowded']
        pair_text=(f'Both members recovered within 3 µm in {c["close_pairs_both_at_3"]}/{c["close_gt_pairs"]} pairs; within 7 µm in {c["close_pairs_both_at_7"]}/{c["close_gt_pairs"]}. ' if c['close_gt_pairs'] else '')
        lines.append(f'- **{LABELS[method]}**: {pair_text}{c["gt_with_multiple_candidates_at_7"]}/{r["gt_nodes"]} annotations have multiple proposals within 7 µm; {c["duplicates_integer"]} duplicate integer points.')
    divisions=base['association']['distance']['eligible_gt_divisions']
    lines += ['', f'The selected pairs contain **{divisions} annotated division events**. '+('That is too little evidence to establish division performance.' if divisions<10 else 'Top-2 daughter recovery is included in the JSON evidence.'), '',
        '## Per-embryo results', '']
    for r in metrics['groups']:
        if r['embryo']=='pooled':continue
        a=r['association']['ensemble_image']
        lines.append(f'- **{r["embryo"]} / {LABELS[r["method"]]}**: {r["candidates"]:,} proposals; R3 {pct(r["matches"]["3"],r["gt_nodes"])} ({r["matches"]["3"]}/{r["gt_nodes"]}); R7 {pct(r["matches"]["7"],r["gt_nodes"])} ({r["matches"]["7"]}/{r["gt_nodes"]}); image parent retrieval {a["correct_parent_top1"]}/{a["eligible_gt_edges"]} ({pct(a["correct_parent_top1"],a["eligible_gt_edges"])}) versus geometry {r["association"]["distance"]["correct_parent_top1"]}/{a["eligible_gt_edges"]}.')
    lines += ['', '## Runtime and provenance', '']
    for r in runtime:
        lines.append(f'- **{LABELS[r["method"]]}**: {r["mean_seconds"]:.2f} seconds/volume, {r["peak_reserved_gib"]:.2f} GiB peak CUDA reservation across {r["frames"]} volumes. Includes inference, reconstruction and writing outputs; excludes the shared GPU queue. Runtime is machine-specific.')
    lines += ['',
        'The additional checkpoints come from [MouseLand’s official Hugging Face repository](https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837): `cpdino` (DINOv3 ViT-L) and `cpsam_v2` (SAM ViT-L). We verified complete learned-weight loading and the downloaded SHA256 hashes. Source and recipes are pinned in [pilot-plan.json](../results/cellpose-embedding-probe-20260914/pilot-plan.json); the [adjacent extension](../results/cellpose-embedding-probe-20260914/adjacent/pilot-plan.json) adds the immediate next frames without altering that recipe.', '',
        'The original embedding-plan draft incorrectly assumed that the existing pilot frames were adjacent. It was archived without execution; the corrected plan adds t26/t76 before any scores were inspected. Detector inference was already running under its unchanged plan.', '',
        'The native encoders retain the incumbent’s training exposure. The Cellpose release does not provide an image-level training manifest sufficient to certify independence from this competition data. Freezing predictions before this evaluation prevents new label-dependent edits; it does not erase earlier exposure or make this pilot an independent embryo test.', '',
        'Increasing proposals can recover missing endpoints and improve small-radius localization, but it also adds competing associations and can affect the final scorer’s count penalty. A full-clip run must measure the combined result. These pilot percentages cannot be substituted for that final metric.', '',
        'For generalization, train the detector and any association encoder with the target embryo excluded from the start, then compare complete graphs with fixed selection/tracking controls and report both embryos separately. Fine-tuning only a new head on top of the all-train incumbent is not an embryo-held-out experiment.', '',
        '## Reproduction and evidence', '',
        'Use `/kaggle/envs/detector-screen-cellpose/bin/python` for detector runs and `/kaggle/envs/cell-tracking-notebooks/bin/python` for native embeddings/evaluation, with `PYTHONNOUSERSITE=1` and two BLAS threads. The shared GPU lock is honored per volume/pair.', '',
        '1. `python -m tools.cellpose_models_probe prepare`; `detect --model cpdino` and `detect --model cpsam_v2`.',
        '2. `python -m tools.cellpose_adjacent_probe prepare`; `detect --model cpdino` and `detect --model cpsam_v2`.',
        '3. `python -m tools.cellpose_embedding_probe prepare`, then `banks`, `native`, `evaluate`.',
        '4. `python -m tools.cellpose_probe_report`.', '',
        'The [validation receipt](../results/cellpose-embedding-probe-20260914/validation.json) verifies all 216 banks and 216 native matrices, independently recounts 180 nodes and 88 edges directly from the canonical GEFF files, and records browser/offline/mobile checks. Re-run the numeric checks with `python -m tools.validate_cellpose_probe`.', '',
        f'- Metrics SHA256: `{payload["metrics_sha256"]}`.',
        '- [Machine-readable metrics](../results/cellpose-embedding-probe-20260914/pilot-metrics.json), [bank hashes](../results/cellpose-embedding-probe-20260914/banks-lock.json), [native prediction hashes](../results/cellpose-embedding-probe-20260914/native-lock.json), [embedding plan](../results/cellpose-embedding-probe-20260914/embedding-plan.json).',
        '- The earlier 400-frame baseline (Cellpose R3 71.76%, R7 95.89%; incumbent R3 85.82%, R7 99.50%) is a different cohort. The numbers in this report must be compared within the 24-frame cohort.', '']
    (REPO/'docs/cellpose-embedding-results-20260914.md').write_text('\n'.join(lines))
    print(json.dumps({'html':str(OUT/'report.html'),'markdown':'docs/cellpose-embedding-results-20260914.md','metrics_sha256':payload['metrics_sha256']}))


if __name__=='__main__':main()
