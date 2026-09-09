"""Independent official scoring of the fresh-image selected reproduction."""
from .common import *


def one(task):
    ctx, sample, variant, expected_hash = task
    from .evaluate import scorer_hash
    from annotation_selection.metric_adapter import evaluate_graph
    name = sample['dataset']
    path = ctx.out / 'fresh_candidates' / variant / f'{name}.npz'
    row = next(r for r in ctx.eval_rows() if r['dataset'] == name)
    gtpath = ctx.v1 / 'evaluation/gt' / f'{name}.npz'
    stamp = dict(graph_sha256=sha(path), gt_sha256=sha(gtpath),
                 estimated_total=row['estimated_total'], physical_scale=sample['physical_scale'],
                 scorer_sha256=scorer_hash(ctx), code_sha256=sha(__file__))
    if stamp['graph_sha256'] != expected_hash:
        raise ValueError('Frozen fresh prediction changed')
    target = ctx.out / 'evaluation/fresh_selected_scores' / variant / f'{name}.json'
    if target.exists():
        old = read_json(target)
        if old['inputs'] != stamp:
            raise ValueError('Fresh-score fingerprint drift')
        return name + ' exact cache'
    graph, gt = load_graph(path), load_graph(gtpath)
    validate(graph['nodes'], graph['edges'], sample['image_shape'])
    result, _, _ = evaluate_graph(name, graph['nodes'], graph['edges'],
        gt['nodes'], gt['edges'], sample['physical_scale'], row['estimated_total'])
    result.update(variant=variant, embryo=sample['embryo'])
    write_json(target, dict(inputs=stamp, result=result,
        scope='Independent evaluation of completed fresh-image reproduction; no new inference policy'))
    return name


def run(ctx, variant, workers=2):
    import pandas as pd
    from .evaluate import scorer_hash
    from annotation_selection.metric_adapter import aggregate
    samples = ctx.samples()
    names = {s['dataset'] for s in samples}
    root = ctx.out / 'fresh_candidates' / variant
    if {p.stem for p in root.glob('*.npz')} != names or len(names) != 199:
        raise ValueError('Fresh selected reproduction must cover all 199 expected clips')
    hashes = {name: sha(root / f'{name}.npz') for name in sorted(names)}
    lockpath = ctx.out / 'fresh_selected_score_lock.json'
    old = read_json(lockpath) if lockpath.exists() else {}
    lock = dict(created=old.get('created', now()), variant=variant, hashes=hashes,
                scorer_sha256=scorer_hash(ctx), input_manifest_sha256=sha(ctx.out/'inputs.json'))
    write_json(lockpath, lock, immutable=True)
    list(run_pool(one, [(ctx, s, variant, hashes[s['dataset']]) for s in samples], workers))
    rows = [read_json(ctx.out / 'evaluation/fresh_selected_scores' / variant / f'{n}.json')['result']
            for n in sorted(names)]
    points = pd.read_csv(ctx.out / 'operating_points.csv')
    output = {}
    fields = ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']
    for embryo in ['44b6', '6bba', 'pooled']:
        subset = [r for r in rows if embryo == 'pooled' or r['embryo'] == embryo]
        expected_names = [s['dataset'] for s in samples if embryo == 'pooled' or s['embryo'] == embryo]
        result = aggregate(subset, expected_names)
        cached = points[(points.variant == variant) & (points.embryo == embryo)].iloc[0]
        baseline = points[(points.variant == 'incumbent') & (points.embryo == embryo)].iloc[0]
        result['delta_v2'] = result['score'] - float(baseline.score)
        result['delta_cached_selected'] = result['score'] - float(cached.score)
        result['count_parity'] = all(result.get(k, result['counts'].get(k)) == int(cached[k]) for k in fields)
        result['score_parity'] = abs(result['delta_cached_selected']) <= 1e-10
        output[embryo] = result
    summary = dict(variant=variant, samples=199, results=output, fresh_whole_graph_matching=True,
        comparison_only_annotation_access=True, annotations_used_for_inference=False,
        fingerprint=sha(lockpath), all_counts_and_scores_match=all(
            r['count_parity'] and r['score_parity'] for r in output.values()))
    write_json(ctx.out / 'fresh_selected_score_summary.json', summary)
    pd.DataFrame(rows).to_csv(ctx.out / 'fresh_selected_score_rows.csv', index=False)
    return summary


if __name__ == '__main__':
    import argparse
    from .context import RunContext
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(RunContext.default(), args.variant, args.workers)
