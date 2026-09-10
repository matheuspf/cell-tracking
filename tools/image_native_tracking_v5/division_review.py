"""Post-freeze official division-identity regret and a local optical review pack."""
import html,time,sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from .common import *


def identities(task):
    variant,row=task;name=row['dataset'];dest=OUT/'division_identities'/variant/f'{name}.json'
    measured=read(OUT/'evaluation'/variant/f'{name}.json')
    if dest.exists():
        r=read(dest);assert r['graph_hash']==measured['graph_hash'];return r
    from annotation_selection.metric_adapter import make_graph
    from tracking_cellmot import division_metrics as dm
    pred=graph(name,variant);gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    pg,pr=make_graph(pred['nodes'],pred['edges']);gg,gr=make_graph(gt['nodes'],gt['edges'])
    result=dm.score_divisions(pg,gg,tuple(row['physical_scale']),7.)
    scores={str(gr[k]):bool(v) for k,v in result.scores.items()}
    assert sum(scores.values())==measured['division_tp']
    assert len(scores)-sum(scores.values())==measured['division_fn']
    assert len(result.fp_forks)==measured['division_fp']
    r=dict(dataset=name,embryo=row['embryo'],variant=variant,graph_hash=measured['graph_hash'],scores=scores,
        fp_forks=[int(pr[k]) for k in result.fp_forks],evaluation_only_GT_IDs=True)
    write(dest,r);return r


def render(name,variant,divider,kind,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import zarr
    gt=arrays(V1/'evaluation/gt'/f'{name}.npz');base=graph(name);new=graph(name,variant)
    parent=next(n for n in gt['nodes'] if int(n[0])==divider);t0=int(parent[1]);z,y,x=map(int,parent[2:])
    volume=zarr.open_group(str(DATA/'train'/f'{name}.zarr'),mode='r')['0']
    zs=slice(max(0,z-8),min(64,z+9));ys=slice(max(0,y-40),min(256,y+41));xs=slice(max(0,x-40),min(256,x+41))
    frames=list(range(max(0,t0-1),min(100,t0+3)));fig,axes=plt.subplots(len(frames),3,figsize=(10,3.1*len(frames)),squeeze=False)
    maps=[{int(n[0]):n for n in g['nodes']} for g in [gt,base,new]];old=set(map(int,base['nodes'][:,0]))
    for i,t in enumerate(frames):
        raw=np.asarray(volume[t,zs,ys,xs]);proj=raw.max(axis=0);lo,hi=np.quantile(proj,[.01,.995]);hi=max(hi,lo+1e-6)
        for j,(g,title,color) in enumerate([(gt,'Sparse provided GT','#00f4ff'),(base,'C0','#ffd65c'),(new,variant,'#ed79ff')]):
            ax=axes[i,j];ax.imshow(proj,cmap='gray',vmin=lo,vmax=hi,extent=[xs.start,xs.stop,ys.stop,ys.start])
            n=g['nodes'];visible=(n[:,1]==t)&(n[:,2]>=zs.start)&(n[:,2]<zs.stop)&(n[:,3]>=ys.start)&(n[:,3]<ys.stop)&(n[:,4]>=xs.start)&(n[:,4]<xs.stop)
            local=n[visible];ax.scatter(local[:,4],local[:,3],s=72,facecolors='none',edgecolors=color,linewidths=1.2)
            if j==2:
                added=local[np.array([int(v) not in old for v in local[:,0]],bool)]
                ax.scatter(added[:,4],added[:,3],s=120,marker='+',color='#66ff78',linewidths=1.2)
            local_ids=set(map(int,local[:,0]))
            for a,b in g['edges']:
                if int(b) not in local_ids:continue
                aa=maps[j][int(a)];bb=maps[j][int(b)]
                if zs.start<=aa[2]<zs.stop:
                    ax.plot([aa[4],bb[4]],[aa[3],bb[3]],color=color,alpha=.7,linewidth=1)
            ax.set_xlim(xs.start,xs.stop);ax.set_ylim(ys.stop,ys.start);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{title} · frame {t}',fontsize=10)
    fig.suptitle(f'{name} · {kind}\nRaw Z maximum projection; circles are detections, segments enter from previous frame, green + are new nodes',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.955]);fig.savefig(destination,dpi=110);plt.close(fig)
    return dict(dataset=name,variant=variant,kind=kind,frames=frames,raw_projection='maximum over 17-or-fewer Z slices',
        sha256=sha(destination),bytes=destination.stat().st_size,GT_overlay=True,new_labels_created=False)


def run(wait=False):
    expected=set(read(OUT/'execution_protocol.json')['variants'])
    from .report import csv_rows
    while True:
        scores=pd.DataFrame(csv_rows('ablation_scores.csv'));complete=set(scores[scores.embryo=='pooled'].variant)
        if complete==expected:break
        if not wait:raise RuntimeError('All frozen graph comparisons are required')
        time.sleep(30)
    if wait:os.execv(sys.executable,[sys.executable,'-m','image_native_tracking_v5.division_review'])
    variants=sorted(v for v in complete if not v.startswith('Oracle_'))
    rows=inventory();tasks=[(v,r) for v in variants for r in rows]
    with cpu_batch(),ProcessPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(identities,tasks,chunksize=2))
    by={(r['variant'],r['dataset']):r for r in records};summaries=[]
    for v in variants:
        for em in ['44b6','6bba','pooled']:
            summary=dict(variant=v,embryo=em,samples=0,GT_divisions_gained=0,GT_divisions_lost=0,still_missed=0,still_recovered=0,FP_forks=0)
            for row in rows:
                if em!='pooled' and row['embryo']!=em:continue
                b=by['C0',row['dataset']]['scores'];n=by[v,row['dataset']]['scores'];assert set(b)==set(n)
                summary['samples']+=1;summary['FP_forks']+=len(by[v,row['dataset']]['fp_forks'])
                for k in b:
                    key='still_recovered' if b[k] and n[k] else 'GT_divisions_lost' if b[k] else 'GT_divisions_gained' if n[k] else 'still_missed'
                    summary[key]+=1
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(OUT/'division_regret.csv',index=False)
    # Hindsight examples are descriptive and selected only after graph freezing.
    family_variants={'H':['H_general_J','H_probe_J','H_probe_native_J'],
        'N':['N_head_J','N_backbone_J'],'P':['P_union_native_J','P_union_N_J','P_DC_native_J','P_DC_N_J']}
    pooled=scores[scores.embryo=='pooled'].set_index('variant');pack=OUT/'optical_review';pack.mkdir(exist_ok=True)
    chosen=[]
    for family,choices in family_variants.items():
        variant=max(choices,key=lambda v:float(pooled.loc[v,'score']))
        for em in ['44b6','6bba']:
            events=[]
            for row in rows:
                if row['embryo']!=em:continue
                b=by['C0',row['dataset']]['scores'];n=by[variant,row['dataset']]['scores']
                for k in sorted(b,key=int):
                    if b[k] and n[k]:continue
                    kind='gained official division' if n[k] else 'lost official division' if b[k] else 'still missed official division'
                    events.append((kind,row['dataset'],int(k)))
            selected=[]
            for kind in ['gained official division','lost official division','still missed official division']:
                candidates=sorted(e for e in events if e[0]==kind)
                if candidates:selected.append(candidates[0])
            for kind,name,divider in selected[:2]:
                dest=pack/f'example_{len(chosen)+1:02d}.png';r=render(name,variant,divider,kind,dest)
                r.update(file=dest.name,family=family,embryo=em);chosen.append(r)
    cards=''.join(f'<article><h2>{html.escape(r["family"]+" · "+r["embryo"]+" · "+r["kind"])}</h2><img src="{r["file"]}" alt="Microscopy comparison for {r["dataset"]}"></article>' for r in chosen)
    (pack/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V5 local optical review</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:15px;background:#101827;color:#e9effa}img{max-width:100%;height:auto}article{border-top:1px solid #41506b;margin-top:30px}</style><h1>V5 post-freeze optical review</h1><p>Local microscopy and sparse GT overlays. These images are diagnostics, not new labels or manual validation. Examples use the highest pooled primary configuration in each family after predictions froze; a lack of gains is not concealed. Z projections can overlap different depths. No model or decoder is changed from this review.</p>'+cards)
    write(OUT/'optical_review_receipt.json',dict(at=now(),complete=True,examples=chosen,
        selection='after all 199-clip graph configurations froze; first available gained, lost, then still-missed official event per family/embryo, at most two',
        annotation_identities_published=False,raw_images_local_only=True,human_judgments_invented=False,new_labels=False))
    print('Official division identity regret and local optical review completed',len(chosen),'examples',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();run(a.wait)
