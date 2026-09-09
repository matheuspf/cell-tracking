"""Fresh full-graph scoring of impossible-inference source feasibility probes."""
from .common import *

def one(task):
    ctx,sample,label,path=task
    from annotation_selection.metric_adapter import evaluate_graph
    name=sample['dataset'];g=load_graph(path);gtpath=ctx.v1/'evaluation/gt'/f'{name}.npz';gt=load_graph(gtpath)
    row=next(r for r in ctx.eval_rows() if r['dataset']==name)
    stamp=dict(graph=sha(path),gt=sha(gtpath),estimate=row['estimated_total'],scorer=ctx.metric_revision)
    dest=ctx.out/'evaluation/oracle_scores'/label/f'{name}.json'
    if dest.exists():
        old=read_json(dest)
        if old['inputs']!=stamp:raise ValueError('Oracle input drift')
        return name+' cached'
    validate(g['nodes'],g['edges'],sample['image_shape'])
    result,_,_=evaluate_graph(name,g['nodes'],g['edges'],gt['nodes'],gt['edges'],sample['physical_scale'],row['estimated_total'])
    result.update(variant=label,embryo=sample['embryo'])
    write_json(dest,dict(inputs=stamp,result=result,scope='Heuristic source training feasibility; impossible inference, never eligible for promotion'))
    return name+' '+label

def run(ctx,args):
    from annotation_selection.metric_adapter import aggregate
    label=args.variant or 'source_association_oracle';root=ctx.out/'evaluation'/label
    expected=[r['dataset'] for r in ctx.samples()]
    if {p.stem for p in root.glob('*.npz')}!=set(expected):raise ValueError('Oracle graphs not complete')
    list(run_pool(one,[(ctx,s,label,root/f'{s["dataset"]}.npz') for s in ctx.samples()],args.workers))
    rows=[read_json(ctx.out/'evaluation/oracle_scores'/label/f'{n}.json')['result'] for n in expected]
    base={r['embryo']:r['score'] for r in __import__('pandas').read_csv(ctx.out/'operating_points.csv').to_dict('records') if r['variant']=='incumbent'}
    results={}
    for embryo in ['44b6','6bba','pooled']:
        subset=[r for r in rows if embryo=='pooled' or r['embryo']==embryo]
        a=aggregate(subset,[r['dataset'] for r in subset]);a['delta_v2']=a['score']-base[embryo];results[embryo]=a
    write_json(ctx.out/f'{label}_score_summary.json',dict(label=label,samples=199,results=results,
        scope='Source-only heuristic feasibility; not global bound, not deployable, not promotion eligible'))

if __name__=='__main__':
    import argparse
    from .context import RunContext
    p=argparse.ArgumentParser();p.add_argument('--variant',default='source_association_oracle');p.add_argument('--workers',type=int,default=2)
    run(RunContext.default(),p.parse_args())
