"""Competition-gate detection evidence; deliberately no temporal-link prediction."""
from __future__ import annotations
import argparse,hashlib,json,logging,sys,time,warnings
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'work/detector-screen-20260914'
sys.path.insert(0,str(REPO/'tools'))
from annotation_selection.metric_adapter import make_graph
from tracksdata.metrics import DistanceMatching
import tracksdata as td

SPACING=np.array([1.625,.40625,.40625])
RADII=[5.,6.,7.]
EVALUATION_VERSION=3

def matches(centers,gt,t,radius):
    if not len(centers) or not len(gt):return {}
    nodes=np.column_stack([np.arange(len(centers)),np.full(len(centers),t),centers])
    g,rev=make_graph(nodes,np.empty((0,2)));truth,tr=make_graph(gt,np.empty((0,2)))
    g.match(truth,matching=DistanceMatching(max_distance=radius,scale=tuple(SPACING),optimal=True))
    pairs=g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    result={rev[int(i)]:tr[int(j)] for i,j in pairs.iter_rows() if j is not None and j!=-1}
    if len(result)!=len(set(result.values())):raise ValueError('Non-bijective tracksdata matching')
    truth_by_id={int(r[0]):r[2:] for r in gt}
    for predicted_id,truth_id in result.items():
        distance=float(np.linalg.norm((centers[predicted_id]-truth_by_id[truth_id])*SPACING))
        if not np.isfinite(distance) or distance>radius+1e-8:
            raise ValueError(f'tracksdata returned an out-of-gate match: prediction={predicted_id}, GT={truth_id}, distance={distance}um, gate={radius}um')
    return result

def evaluate_frame(row,path,truth,baseline_n):
    payload=np.load(path);centers=np.asarray(payload['centers_zyx'],dtype=np.float64).reshape(-1,3)
    scores=np.asarray(payload['scores'],dtype=np.float64).reshape(-1)
    if len(centers)!=len(scores) or not np.isfinite(centers).all() or not np.isfinite(scores).all():raise ValueError(path)
    unrounded=centers.copy();centers=np.clip(np.rint(centers),0,np.array(row['shape'])-1)
    result=dict(key=row['key'],dataset=row['dataset'],embryo=row['embryo'],time=row['time'],role=row['role'],
        gt=len(truth),predicted=len(centers),duplicates_after_rounding=len(centers)-len(np.unique(centers,axis=0)),
        clipped_centers=int(np.any(np.rint(unrounded)!=centers,axis=1).sum()),baseline_candidates=baseline_n)
    m={str(r):matches(centers,truth,row['time'],r) for r in RADII}
    result['matches']={r:len(v) for r,v in m.items()};result['matched_gt_at_7']=list(m['7.0'].values())
    truth_by_id={int(r[0]):r[2:] for r in truth}
    result['assigned_distances_um_at_7']=[float(np.linalg.norm((centers[i]-truth_by_id[j])*SPACING)) for i,j in m['7.0'].items()]
    if len(truth) and len(centers):
        d=cdist(truth[:,2:]*SPACING,centers*SPACING);nearest=d.min(axis=1)
        result['nearest_coverage_at_7']=int((nearest<=7).sum())
        result['gt_multiple_candidates_at_7']=int(((d<=7).sum(axis=1)>=2).sum())
        result['nearest_in_5_to_7_um']=int(((nearest>5)&(nearest<=7)).sum())
    else:result.update(nearest_coverage_at_7=0,gt_multiple_candidates_at_7=0,nearest_in_5_to_7_um=0)
    ordered=np.argsort(-scores,kind='stable')
    result['budgets']={}
    for multiplier in [.5,.75,1.,1.25,1.5]:
        n=min(len(centers),int(round(baseline_n*multiplier)))
        idx=ordered[:n]
        bm=matches(centers[idx],truth,row['time'],7.)
        result['budgets'][str(multiplier)]=dict(predicted=n,matched=len(bm),matched_gt_at_7=list(bm.values()))
    receipt=path.with_suffix('.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());result['runtime_receipt']=r
    return result

def edge_availability(rows,gt,budget=None):
    ids_by_clip={};frame_times={}
    for r in rows:
        ids=r['matched_gt_at_7'] if budget is None else r['budgets'][budget]['matched_gt_at_7']
        ids_by_clip.setdefault(r['dataset'],set()).update(ids)
        frame_times.setdefault(r['dataset'],set()).add(r['time'])
    eligible=covered=0
    for name,ids in ids_by_clip.items():
        clip=gt['clips'][name];times={int(r[0]):int(r[1]) for r in clip['nodes']}
        for a,b in clip['edges']:
            if times[a] in frame_times[name] and times[b] in frame_times[name]:
                eligible+=1;covered+=a in ids and b in ids
    return dict(gt_edge_endpoints_available=covered,gt_edges_in_panel=eligible,
                edge_endpoint_availability=covered/eligible if eligible else None)

def aggregate(rows,gt,baseline_rows):
    out=[];base={r['key']:r for r in baseline_rows}
    for role in ['pilot','assessment']:
      for embryo in ['44b6','6bba','pooled']:
        selected=[r for r in rows if r['role']==role and (embryo=='pooled' or r['embryo']==embryo)]
        if not selected:continue
        ng=sum(r['gt'] for r in selected);n=sum(r['predicted'] for r in selected)
        ids_by_clip={};frame_times={}
        for r in selected:
            ids_by_clip.setdefault(r['dataset'],set()).update(r['matched_gt_at_7'])
            frame_times.setdefault(r['dataset'],set()).add(r['time'])
        eligible=covered=0
        for name,ids in ids_by_clip.items():
            clip=gt['clips'][name];times={int(r[0]):int(r[1]) for r in clip['nodes']}
            for a,b in clip['edges']:
                if times[a] in frame_times[name] and times[b] in frame_times[name]:
                    eligible+=1;covered+=a in ids and b in ids
        matched={rad:sum(r['matches'][rad] for r in selected) for rad in ['5.0','6.0','7.0']}
        gains=losses=0;paired=0
        for r in selected:
            if r['key'] in base:
                theirs=set(r['matched_gt_at_7']);ours=set(base[r['key']]['matched_gt_at_7'])
                gains+=len(theirs-ours);losses+=len(ours-theirs);paired+=1
        if paired!=len(selected):raise ValueError(f'Missing current incumbent evaluation for {len(selected)-paired} frames')
        ds=np.array([d for r in selected for d in r['assigned_distances_um_at_7']])
        out.append(dict(role=role,embryo=embryo,frames=len(selected),clips=len(ids_by_clip),gt=ng,predicted=n,
            matches=matched,recall={r:matched[r]/ng if ng else None for r in matched},
            misses_at_7=ng-matched['7.0'],gt_edge_endpoints_available=covered,gt_edges_in_panel=eligible,
            edge_endpoint_availability=covered/eligible if eligible else None,
            candidate_ratio_to_incumbent=n/sum(r['baseline_candidates'] for r in selected),
            gains_over_incumbent_at_7=gains,losses_from_incumbent_at_7=losses,paired_baseline_frames=paired,
            duplicate_centers=sum(r['duplicates_after_rounding'] for r in selected),
            gt_multiple_candidates_at_7=sum(r['gt_multiple_candidates_at_7'] for r in selected),
            nearest_coverage_at_7=sum(r['nearest_coverage_at_7'] for r in selected),
            nearest_in_5_to_7_um=sum(r['nearest_in_5_to_7_um'] for r in selected),
            p90_matched_error_um=float(np.quantile(ds,.9)) if len(ds) else None,
            budgets={b:dict(predicted=sum(r['budgets'][b]['predicted'] for r in selected),
                matched=sum(r['budgets'][b]['matched'] for r in selected),
                recall=sum(r['budgets'][b]['matched'] for r in selected)/ng if ng else None,
                **edge_availability(selected,gt,b)) for b in ['0.5','0.75','1.0','1.25','1.5']}))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--methods',nargs='*');ap.add_argument('--role',default='all',choices=['pilot','assessment','all']);args=ap.parse_args()
    logging.disable(logging.WARNING);warnings.filterwarnings('ignore',message='No matching edges')
    panel_bytes=(ROOT/'panel.json').read_bytes();gt_bytes=(ROOT/'evaluation/ground_truth.json').read_bytes()
    panel=json.loads(panel_bytes);gt=json.loads(gt_bytes)
    data_fingerprint=hashlib.sha256(panel_bytes+gt_bytes).hexdigest()
    methods=args.methods or sorted(p.name for p in (ROOT/'predictions').iterdir() if p.is_dir())
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    baseline_path=out/'incumbent.json'
    baseline_data=json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    baseline=baseline_data.get('per_frame',[]) if baseline_data.get('data_fingerprint')==data_fingerprint and baseline_data.get('evaluation_version')==EVALUATION_VERSION else []
    needed_baseline_keys={r['key'] for r in panel['frames'] if args.role=='all' or r['role']==args.role}
    if 'incumbent' in methods or not needed_baseline_keys.issubset({r['key'] for r in baseline}):
        methods=['incumbent']+[m for m in methods if m!='incumbent']
    for method in methods:
        start=time.monotonic();rows=[]
        old_path=out/(method+'.json');old_data=json.loads(old_path.read_text()) if old_path.exists() else {}
        old=old_data.get('per_frame',[]) if old_data.get('evaluation_version')==EVALUATION_VERSION and old_data.get('data_fingerprint')==data_fingerprint else []
        cached={r['key']:r for r in old}
        for row in panel['frames']:
            if args.role!='all' and row['role']!=args.role:continue
            p=ROOT/'predictions'/method/(row['key']+'.npz')
            if not p.exists():continue
            receipt=p.with_suffix('.json')
            receipt_stamp=str(receipt.stat().st_mtime_ns) if receipt.exists() else 'none'
            fingerprint=str(EVALUATION_VERSION)+':'+str(p.stat().st_mtime_ns)+':'+str(p.stat().st_size)+':'+receipt_stamp
            if row['key'] in cached and cached[row['key']].get('prediction_fingerprint')==fingerprint:
                rows.append(cached[row['key']]);continue
            item=gt['frames'][row['key']];truth=np.asarray(item['gt_nodes'],dtype=np.int64).reshape(-1,5)
            r=evaluate_frame(row,p,truth,item['baseline_candidates']);r['prediction_fingerprint']=fingerprint;rows.append(r)
        if args.role!='all':rows+= [r for r in old if r['role']!=args.role]
        if method=='incumbent':baseline=rows
        summary=aggregate(rows,gt,baseline)
        result=dict(evaluation_version=EVALUATION_VERSION,data_fingerprint=data_fingerprint,method=method,per_frame=rows,summary=summary,matching='Actual installed tracksdata DistanceMatching optimal=True, 1/(1+physical distance) weights, native integer rounding; no temporal linking.',
            interpretation='Edge endpoint availability is a detector-only upper-bound diagnostic under these per-frame matches, not tracking accuracy or a competition score. Candidate budgets use incumbent counts and confidence ranks, without GT-dependent prediction edits.')
        tmp=old_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(old_path)
        print(method,'frames',len(rows),'seconds',round(time.monotonic()-start,2),flush=True)
        for r in summary:
            if r['embryo']=='pooled':print(json.dumps({k:r[k] for k in ['role','frames','gt','predicted','recall','misses_at_7','edge_endpoint_availability','gains_over_incumbent_at_7','losses_from_incumbent_at_7']}),flush=True)

if __name__=='__main__':main()
