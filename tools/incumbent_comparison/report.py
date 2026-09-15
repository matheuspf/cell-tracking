"""Portable HTML and Markdown results, with an optional local live-status server."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path

from .common import REPO, WORK, OUT, read, write, now

LABELS = {'incumbent': 'Public incumbent ensemble', 'baseline': 'Cellpose ViT-B',
          'constant': 'Constant offset · source only', '20260914': 'Refiner · source only · 20260914',
          '314159': 'Refiner · source only · 314159', 'both-20260914': 'Refiner · both embryos · 20260914',
          'both-314159': 'Refiner · both embryos · 314159'}


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def collect():
    old = read(REPO / 'results/cellpose-refine-v1/report-data.json')
    curves = []
    for method, data in old['methods'].items():
        for group in data['integer']:
            curves.append(dict(method=method, label=LABELS[method], embryo=group['embryo'], gt=group['gt'],
                               candidates=group['predicted'], recall={str(k): group['recall'][f'{k}.0'] for k in range(1, 8)},
                               close={k: dict(pairs=v['pairs'], both_at3=v['both_matched']['3.0']) for k, v in group['close_pairs'].items()}))
    refit_path = OUT / 'refit-assessment.json'
    if refit_path.exists():
        for group in read(refit_path)['groups']:
            method = f"both-{group['seed']}"
            curves.append(dict(group, method=method, label=LABELS[method]))
    launch = read(WORK / 'native-launch.json')
    training = []
    for source in ('44b6', '6bba'):
        p = WORK / f'native-{source}-progress.json'
        if not p.exists():
            continue
        row = read(p)
        folder = WORK / 'models/native' / f'source-{source}-seed-314159'
        final = folder / 'final.json'
        row['complete'] = final.exists()
        record = next(r for r in launch['processes'] if r['source'] == source)
        row['pid'] = record['pid']
        row['process_running'] = alive(row['pid'])
        row['fraction'] = 1. if row['complete'] else (row['epoch'] - 1 + row['epoch_windows'] / row['windows_per_epoch']) / row['target_epochs']
        row['remaining_compute_days_estimate'] = max(0, math.ceil(row['windows_per_epoch'] / 8) * row['target_epochs'] - row['updates']) * row['latest_seconds_per_update'] / 86400
        row['checkpoint_exists'] = (folder / 'resume.pt').exists()
        training.append(row)
    progress_path = WORK / 'full-clips/progress.json'
    driver_path = WORK / 'full-clips-run.json'
    full = dict(progress=read(progress_path) if progress_path.exists() else None,
                driver=read(driver_path) if driver_path.exists() else None)
    if (OUT / 'full-clips-evaluation.json').exists():
        full['evaluation'] = read(OUT / 'full-clips-evaluation.json')
    if full['driver']:
        full['driver']['process_running'] = alive(full['driver']['pid'])
    return dict(updated_utc=now(), published=read(OUT / 'replay-plan.json')['published_reference'],
                replay={k: {n: v for n, v in read(OUT / filename).items() if n != 'window_order'}
                        for k, filename in [('cap200', 'released-secondary-cap200.json'), ('full40', 'released-secondary-full-monitor.json')]},
                curves=curves, refits_assessed=refit_path.exists(), native_training=training,
                native_launch=launch, full_clips=full,
                scope='The historical monitor and both-embryo refits reuse training labels. Source-only refiner adaptation excludes the target embryo, while cpDINO pretraining ancestry remains unresolved. Native source-only runs start the entire model from random weights. Only two public embryos are available; all panels were previously inspected.')


def markdown(data):
    cap, full = data['replay']['cap200'], data['replay']['full40']
    lines = ['# Incumbent comparison — 14 September 2026', '', f"Updated: {data['updated_utc']}", '',
             'Open the ordinary HTML report at http://127.0.0.1:8770/report.html. It refreshes progress while the local server runs; the saved report also opens directly in a browser.', '',
             '## Released validation replay', '',
             f"The released seed-314159 checkpoint scores **{cap['product']:.10f}**, versus published **{data['published']['product']:.10f}** (difference {cap['product_delta_from_published']:+.10f}). This is near-exact, not bit-exact reproduction.", '',
             f"Its 200-batch prefix covers {cap['window_count']:,} eligible two-frame windows from {cap['clip_count']} of the 40 listed monitoring clips. Detection recall is {cap['counts']['gt_matched']:,}/{cap['counts']['gt_total']:,} = {100*cap['node_recall']:.6f}%, exactly matching the published recall. Edge accuracy is {cap['counts']['correct']:,}/{cap['counts']['total']:,} = {cap['accuracy']:.10f}.", '',
             f"The complete 40-clip monitor gives **{full['product']:.10f}**, recall {100*full['node_recall']:.6f}% and edge accuracy {full['accuracy']:.10f}. The cap and complete-monitor scores must not be mixed.", '',
             'These are the upstream edge-classification-accuracy × detected-node-recall proxy, using greedy 5 µm matching and the original proposal rules. They are **not the competition score**. The monitor is fully included in the 199-clip training set and contains only embryo 44b6. Pair windows count frames repeatedly; these denominators are not unique cells.', '',
             '## Detector comparison', '',
             'All rows below use the same 400 assessment frames, 40 clips and 2,383 annotations, native integer coordinates, and optimal one-to-one matching. The HTML includes every radius from 1–7 µm, both embryos separately, and close-pair counts.', '']
    for row in [r for r in data['curves'] if r['embryo'] == 'pooled']:
        values = ' / '.join(f"{100*row['recall'][str(k)]:.2f}%" for k in (1, 2, 3, 7))
        close = row['close']['7']
        lines.append(f"- **{row['label']}**: recall at 1/2/3/7 µm = {values}; {row['candidates']:,} candidates. Close GT pairs ≤7 µm recovered as two separate cells within 3 µm: {close['both_at3']}/{close['pairs']}.")
    lines += ['', 'Both-embryo heads retain the original 3,000-update query-training recipe and two fixed seeds. They use 2,268 natural queries and 27,088 jitter/center queries across the 199-clip population. This matches the incumbent’s eligible clip population, **not its exact temporal sampling, observation count, optimization budget, architecture, or pretraining**. These measurements quantify training-exposure effects; they do not prove generalization.', '',
              'The residual head keeps proposal counts/confidence unchanged and can move an integer center by at most 3 µm. Its headroom is therefore smaller than that of a new detector that can add, split, or remove cells.', '',
              '## Full competition-score experiment', '',
              'Nine methods are registered on six complete 100-frame clips: incumbent, Cellpose, source-only constant offset, both source-only refiner seeds, both both-embryo refiner seeds, incumbent + Cellpose, and incumbent + source-only refiner seed 20260914. Unions use the previously fixed 2 µm suppression radius.', '',
              'Every method uses fresh native features at its own predicted centers and the same public primary association checkpoint. The upstream greedy linker uses source-normalized softmax, probability >0.5, at most one parent and two children, and no motion gate. Node counts follow the detector; they are not forced equal. All graphs are frozen before evaluation with the pinned official metric. This is a controlled native pipeline, not a reproduction of the full production Harmonic pipeline.', '']
    if data['refits_assessed']:
        pooled = {r['method']: r for r in data['curves'] if r['embryo'] == 'pooled'}
        line = 'The exposure comparison is large: '
        for seed in ('20260914', '314159'):
            trained = pooled['both-' + seed]
            excluded = pooled[seed]
            line += f"seed {seed} gains {100*(trained['recall']['3']-excluded['recall']['3']):.2f} percentage points at 3 µm when both embryos are used for fitting; "
        line += 'both remain below the incumbent at 7 µm. This supports further work on transfer and missed-cell coverage; it does not establish a superior detector for unseen embryos.'
        position = lines.index('## Full competition-score experiment')
        lines[position:position] = [line, '']
    if 'evaluation' in data['full_clips']:
        for method, groups in data['full_clips']['evaluation']['summaries'].items():
            row = groups['pooled']
            lines.append(f"- {method}: official local score **{row['score']:.8f}**, {row['proposals']:,} nodes, {row['linked_gt_edges']}/{row['available_gt_edges']} available GT edges linked.")
    else:
        progress = data['full_clips']['progress']
        lines += [f"Status: **{progress['stage']}**, {progress['completed']}/{progress['total']} ({progress['updated_utc']}). No full-clip score is available yet." if progress else 'Full-clip job has not started.']
    lines += ['', 'The six clips and public association checkpoint are training-exposed. This stage tests score changes and extra-proposal tradeoffs on a fixed downstream pipeline; it is not an independent generalization claim.', '',
              '## Native training with each embryo excluded', '',
              'Two whole-model fits start from random initialization: 44b6 → 6bba (71 source clips, 6,315 windows), and 6bba → 44b6 (128 source clips, 12,392 windows). Each retains the native batch size 8, AdamW 1e-4, original losses and brightness/flip augmentations. The fixed final epoch is 400. Target evaluation waits until both final checkpoints exist.', '',
              'Source-only workers reject reads of the other embryo and inherited checkpoints. Input and source hashes are frozen. Cached normalized FP32 images reproduce the original augmented half-precision storage round trip exactly on the checked windows. Runs save model, optimizer and RNG state, and release the GPU every eight updates.', '']
    for row in data['native_training']:
        status = 'complete' if row['complete'] else ('running' if row['process_running'] else 'worker stopped; inspect log')
        lines.append(f"- {row['source']} → {row['target']}: **{status}**, epoch {row['epoch']}/{row['target_epochs']}, {row['updates']:,} updates, {row['epoch_windows']:,}/{row['windows_per_epoch']:,} windows in current epoch; peak reserved {row['peak_reserved_bytes']/2**30:.2f} GiB. Last progress {row['updated_utc']}.")
    days = sum(r['remaining_compute_days_estimate'] for r in data['native_training'])
    lines += ['', f"The full campaign currently projects roughly **{days:.1f} GPU-days remaining**, based on the latest batches. This is a throughput estimate, not a promised finish time; contention adds wall time. Both source-only fits have saved resumable checkpoints.", '',
              'Only two independent public embryos exist. Report both directions and paired per-clip differences; repeated/overlapping crops and previously inspected panels cannot establish broad biological generalization.', '',
              '## Reproducibility', '',
              '- Plans and receipts: `results/incumbent-comparison-20260914/`.',
              '- Heavy artifacts, checkpoints and logs: `work/incumbent-comparison-20260914/`.',
              '- Resume native fits with the notebook runtime: `python -m tools.incumbent_comparison.train_native run --source 44b6` (or `6bba`). Per-source locks prevent duplicate workers.',
              '- Resume complete-clip stages: `python -m tools.incumbent_comparison.run_full`. Failed stages are logged; completed banks are verified and reused.',
              '- Serve/update report: `python -m tools.incumbent_comparison.report --serve --port 8770`.',
              '- The source-only native target inference/evaluation stage is deferred until both 400-epoch fits finish; it is not currently an automatically scheduled worker.', '']
    return '\n'.join(lines)


def build():
    data = collect()
    write(OUT / 'report-data.json', data)
    template = Path(__file__).with_suffix('.html').read_text()
    rendered = template.replace('__DATA__', json.dumps(data, allow_nan=False).replace('<', '\\u003c'))
    path = OUT / 'report.html'
    temp = path.with_suffix('.html.tmp')
    temp.write_text(rendered)
    temp.replace(path)
    (REPO / 'docs/incumbent-comparison-20260914.md').write_text(markdown(data))
    return rendered, data


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path not in ('/', '/report.html', '/status.json'):
            self.send_error(404)
            return
        try:
            rendered, data = build()
            body = json.dumps(data).encode() if path == '/status.json' else rendered.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if path == '/status.json' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as error:
            self.send_error(500, str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    build()
    if args.serve:
        # Serial request handling avoids simultaneous snapshot-file writers.
        from http.server import HTTPServer
        HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
