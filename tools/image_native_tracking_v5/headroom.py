"""Evaluation-only candidate coverage and legal, non-deployable heuristic oracles.

No production model imports this module. Local event IDs/matches remain ignored.
These interventions use truth and are feasibility diagnostics, not upper bounds.
"""
import time
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
import pandas as pd
from .common import *
from .candidates import bank,motion_config
from .temporal_decode import protected_context
from annotation_selection.metric_adapter import make_graph,match_nodes
from tracking_cellmot import division_metrics as dm


def adjacency(edges):
    successors={};predecessors={}
    for a,b in edges:
        successors.setdefault(int(a),set()).add(int(b));predecessors.setdefault(int(b),set()).add(int(a))
    return successors,predecessors


def force_legal(edges,required,locked=()):
    required=set(map(tuple,required));locked=set(map(tuple,locked));rs,rp=adjacency(required|locked)
    if any(len(v)>2 for v in rs.values()) or any(len(v)>1 for v in rp.values()):return edges,False
    owners={b:a for a,b in required}
    result={p for p in edges if p[1] not in owners or owners[p[1]]==p[0]}|required
    # Remove competing owners in one pass, then optional out-edges.
    ss,_=adjacency(result)
    keep_by_parent={}
    for a,bs in ss.items():
        if len(bs)<=2:continue
        keep={b for x,b in required|locked if x==a}
        keep.update(sorted(bs-keep)[:2-len(keep)])
        keep_by_parent[a]=keep
    result={p for p in result if p[0] not in keep_by_parent or p[1] in keep_by_parent[p[0]]}
    if not locked<=result:return edges,False
    ss,pp=adjacency(result)
    assert all(len(v)<=2 for v in ss.values()) and all(len(v)<=1 for v in pp.values())
    return result,True


def local_options(parent_ids,daughter_ids,pairs,times,old_edges,logits):
    successors,predecessors=adjacency(pairs);score=dict(zip(map(tuple,pairs),logits))
    forks=set(parent_ids)
    for p in parent_ids:forks.update(successors.get(p,()))
    options=[]
    for f in sorted(forks):
        anchors=[()] if f in parent_ids else [((p,f),) for p in sorted(predecessors.get(f,set())&parent_ids)]
        branches=[]
        for lineage in daughter_ids:
            possibilities=[]
            for c in sorted(successors.get(f,())):
                if c in lineage:possibilities.append((c,((f,c),)))
                else:
                    for gc in sorted(successors.get(c,set())&lineage):possibilities.append((c,((f,c),(c,gc))))
            branches.append(possibilities)
        if len(branches)!=2:continue
        for ca,ea in branches[0]:
            for cb,eb in branches[1]:
                if ca==cb:continue
                for anchor in anchors:
                    needed=set(anchor+ea+eb)
                    ss,pp=adjacency(needed)
                    if any(len(v)>1 for v in pp.values()) or any(len(v)>2 for v in ss.values()):continue
                    options.append((len(needed-old_edges),-sum(score.get(e,-40) for e in needed),sorted(needed)))
    options.sort()
    # A bounded heuristic intervention; coverage is computed before this cap.
    return [v[2] for v in options[:64]],len(options)


def division_diagnostics(nodes,edges,gt,pairs,logits,scale,valid,base_matches):
    import tracksdata as td
    pg,pr=make_graph(nodes,edges);gg,gr=make_graph(gt['nodes'],gt['edges'])
    official=dm.score_divisions(pg,gg,tuple(scale),7.)
    locals_=dm.match_divisions(pg,gg,tuple(scale),7.);divisions=dm.extract_divisions(gg)
    times={int(n[0]):int(n[1]) for n in nodes};old=set(map(tuple,edges));available=set(map(tuple,pairs))
    protected,_=protected_context(edges);events=[];requests=[]
    gs,gp=adjacency(gt['edges']);gt_reverse={v:k for k,v in base_matches.items()}
    validmap=dict(zip(nodes[:,0].astype(int),valid));keys=td.DEFAULT_ATTR_KEYS
    for gid,matched in locals_.items():
        attrs=dm._matched_node_attrs(matched)
        pred_to_gt={pr[int(p)]:gr[int(g)] for p,g in attrs.select(keys.NODE_ID,keys.MATCHED_NODE_ID).iter_rows()}
        raw_gid=gr[gid];rawchildren=sorted(gs.get(raw_gid,set()))
        parent_gt={raw_gid}|gp.get(raw_gid,set())
        parent_ids={p for p,g in pred_to_gt.items() if g in parent_gt}
        daughter_ids=[{p for p,g in pred_to_gt.items() if g in ({child}|gs.get(child,set()))} for child in rawchildren]
        options,option_count=local_options(parent_ids,daughter_ids,pairs,times,old,logits) if parent_ids and all(daughter_ids) else ([],0)
        exact_edges={(gt_reverse.get(raw_gid,-1),gt_reverse.get(c,-1)) for c in rawchildren}
        exact_present=len(rawchildren)==2 and exact_edges<=available
        if official.scores[gid]:bucket='C0_official_recovered'
        elif not parent_ids:bucket='missing_parent'
        elif not all(daughter_ids):bucket='missing_daughter'
        elif not options:
            possible_timing=any(all(any(times[d] in (tf+1,tf+2) for d in ds) for ds in daughter_ids)
                for p in parent_ids for tf in (times[p],times[p]+1))
            if not possible_timing:bucket='wrong_timing'
            elif any(not any(g in rawchildren for p,g in pred_to_gt.items() if p in ds) for ds in daughter_ids):bucket='missing_downstream_path'
            else:bucket='insufficient_candidates'
        elif not any(all(validmap.get(a,False) and validmap.get(b,False) for a,b in op) for op in options):bucket='unsupported_HOCT_score'
        elif not any(force_legal(protected,op,protected)[1] for op in options):bucket='conflict_or_protection_loss'
        else:bucket='available_unselected_evidence'
        event=dict(gt_divider=raw_gid,official_recovered=bool(official.scores[gid]),exact_id_candidate_pair=exact_present,
            local_legal_candidate_feasible=bool(options),candidate_timing_options=option_count,bucket=bucket,
            matched_parent_side=len(parent_ids),matched_daughter_sides=[len(s) for s in daughter_ids],
            scored_population='exact unchanged C0 nodes; fresh official local-window matching',
            options=options)
        events.append(event)
        if not official.scores[gid] and options:requests.append(options)
    return events,requests,dict(tp=sum(official.scores.values()),fp=len(official.fp_forks),fn=len(official.scores)-sum(official.scores.values()))


def association_oracle(nodes,edges,gt,pairs,matches,requests=()):
    reverse={g:p for p,g in matches.items()};available=set(map(tuple,pairs));locked=set()
    wanted={(reverse[int(a)],reverse[int(b)]) for a,b in gt['edges'] if int(a) in reverse and int(b) in reverse}&available
    result=set(map(tuple,edges));conflicts=0
    # One-to-one GT endpoint matching yields a legal supported edge set.
    proposed,ok=force_legal(result,wanted)
    assert ok,'GT support unexpectedly violates the graph contract'
    result=proposed;locked=wanted.copy()
    divisions_forced=0
    for alternatives in requests:
        for op in alternatives:
            proposed,ok=force_legal(result,op,locked)
            if ok:result=proposed;locked.update(map(tuple,op));divisions_forced+=1;break
        else:conflicts+=1
    return np.asarray(sorted(result),np.int64).reshape(-1,2),dict(supported_edges_requested=len(wanted),conflicts=conflicts,local_division_requests_forced=divisions_forced)


def augmented_selection(c,base,union_matches,base_matches):
    old_ids=set(base['nodes'][:,0].astype(int));protected,pnodes=protected_context(base['edges'])
    present=set(base_matches.values());selected=set(old_ids);owners=c['split_owner'];n=c['nodes'];paired=set();group_count=0
    for owner in sorted(set(owners)-{-1}):
        indices=np.flatnonzero(owners==owner)
        if owner in pnodes or len(indices)<2:continue
        best=sorted(indices,key=lambda i:(-c['confidence'][i],int(n[i,0])))[:2]
        ids=[int(n[i,0]) for i in best];paired.update(ids)
        mapped=[union_matches.get(i) for i in ids]
        if all(g is not None for g in mapped) and any(g not in present for g in mapped):
            selected.discard(int(owner));selected.update(ids);present.update(mapped);group_count+=1
    for p,g in sorted(union_matches.items()):
        if p not in old_ids and p not in paired and g not in present:selected.add(p);present.add(g)
    nodes=n[np.array([int(i) in selected for i in n[:,0]])]
    edges=np.asarray([e for e in base['edges'] if int(e[0]) in selected and int(e[1]) in selected],np.int64).reshape(-1,2)
    return nodes,edges,dict(added_nodes=len(selected-old_ids),removed_nodes=len(old_ids-selected),selected_split_groups=group_count)


def one(row,output_root=None):
    destination=OUT if output_root is None else Path(output_root)
    name=row['dataset'];dest=destination/'headroom'/f'{name}.json'
    if dest.exists():return read(dest)
    source='6bba' if row['embryo']=='44b6' else '44b6';start=time.monotonic()
    base=graph(name);c=arrays(OUT/'observations'/f'{name}.npz');gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    fixed=arrays(OUT/'banks'/source/'P0'/f'{name}.npz');expanded=arrays(OUT/'banks'/source/'P1'/f'{name}.npz')
    matches={int(p):int(g) for p,g in arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches']}
    reverse={g:p for p,g in matches.items()};positive={(reverse[int(a)],reverse[int(b)]) for a,b in gt['edges'] if int(a) in reverse and int(b) in reverse}
    distance=motion_config()[source]['distance_um'];spatial8=set(map(tuple,bank(base['nodes'],base['edges'],distance,8)))
    native_added=set(map(tuple,fixed['pairs']))-spatial8;caps={}
    for k in [4,8,16]:
        pp=set(map(tuple,bank(base['nodes'],base['edges'],distance,k)))|native_added
        caps[str(k)]=dict(edges=len(pp),gt_transitions_covered=len(pp&positive),gt_transitions_lost_given_endpoints=len(positive-pp))
    crowded=set()
    for t in range(100):
        nn=base['nodes'][base['nodes'][:,1]==t]
        if len(nn)>1:
            near=cKDTree(nn[:,2:]*row['physical_scale']).query_ball_point(nn[:,2:]*row['physical_scale'],6.5,return_length=True)
            crowded.update(nn[near>=3,0].astype(int))
    gt_succ,_=adjacency(gt['edges']);mitotic_parents={reverse[g] for g,cs in gt_succ.items() if len(cs)==2 and g in reverse}
    bankset=set(map(tuple,fixed['pairs']));miss=positive-bankset
    events,requests,official=division_diagnostics(base['nodes'],base['edges'],gt,fixed['pairs'],fixed['native_logits'],
        row['physical_scale'],c['valid_region'][c['oldmask']],matches)
    c0score=read(OUT/'evaluation/C0'/f'{name}.json')
    assert official==dict(tp=c0score['division_tp'],fp=c0score['division_fp'],fn=c0score['division_fn'])
    oe,oreceipt=association_oracle(base['nodes'],base['edges'],gt,fixed['pairs'],matches,requests)
    save_delta(name,'Oracle_fixed',base['nodes'],oe,output_root=destination)
    union_matches=match_nodes(c['nodes'],base['edges'],gt['nodes'],gt['edges'],row['physical_scale'])
    an,ae,selection=augmented_selection(c,base,union_matches,matches);valid_ids=set(an[:,0].astype(int))
    pmask=np.array([int(a) in valid_ids and int(b) in valid_ids for a,b in expanded['pairs']]);ap=expanded['pairs'][pmask];al=expanded['native_logits'][pmask]
    am=match_nodes(an,ae,gt['nodes'],gt['edges'],row['physical_scale'])
    lookup={int(i):j for j,i in enumerate(c['nodes'][:,0])};av=np.array([c['valid_region'][lookup[int(i)]] for i in an[:,0]])
    aevents,arequests,_=division_diagnostics(an,ae,gt,ap,al,row['physical_scale'],av,am)
    aedges,ar=association_oracle(an,ae,gt,ap,am,arequests);save_delta(name,'Oracle_augmented',an,aedges,output_root=destination)
    from strong_tracker_v3.common import validate
    validate(base['nodes'],oe,row['image_shape']);validate(an,aedges,row['image_shape'])
    new_ids=set(c['nodes'][~c['oldmask'],0].astype(int));new_matches={p:g for p,g in union_matches.items() if p in new_ids}
    r=dict(dataset=name,embryo=row['embryo'],source_motion=source,seconds=time.monotonic()-start,
        gt_nodes=len(gt['nodes']),C0_nodes=len(base['nodes']),C0_matched=len(matches),gt_edges=len(gt['edges']),
        endpoints_present=len(positive),candidate_edges=len(fixed['pairs']),candidate_gt_edges=len(positive&bankset),
        cap_coverage=caps,crowded_endpoint_edges=sum(a in crowded or b in crowded for a,b in positive),
        missed_candidate_edges_crowded=sum(a in crowded or b in crowded for a,b in miss),
        mitotic_endpoint_edges=sum(a in mitotic_parents for a,b in positive),missed_candidate_edges_mitotic=sum(a in mitotic_parents for a,b in miss),
        valid_HOCT_regions=int(c['valid_region'][c['oldmask']].sum()),novel_peaks=len(new_ids),
        union_matched=len(union_matches),new_peak_matched_nodes=len(new_matches),
        new_peak_new_GT_recovery=len(set(new_matches.values())-set(matches.values())),
        union_C0_GT_matches_lost=len(set(matches.values())-set(union_matches.values())),
        official_divisions=official,official_local_candidate_feasible=sum(e['local_legal_candidate_feasible'] for e in events),
        exact_id_candidate_divisions=sum(e['exact_id_candidate_pair'] for e in events),
        division_buckets={b:sum(e['bucket']==b for e in events) for b in sorted({e['bucket'] for e in events})},
        fixed_oracle=oreceipt,augmented_selection=selection,augmented_oracle=ar,
        oracle_scope='truth-assisted greedy supported links and local timing paths; graph-legal and bank-constrained; not a global upper bound; C0 protection relaxed for headroom',
        evaluation_only=True)
    write(destination/'oracle_events'/f'{name}.json',dict(fixed=events,augmented=aevents,local_only_GT_IDs=True))
    write(dest,r);return r


def run():
    rows=inventory()
    for r in rows:
        source='6bba' if r['embryo']=='44b6' else '44b6'
        while not (OUT/'banks'/source/'P1'/f"{r['dataset']}.json").exists():time.sleep(15)
    write(OUT/'headroom_protocol.json',dict(created=now(),oracle_names=['Oracle_fixed','Oracle_augmented'],
        protocol_sha256=sha(OUT/'execution_protocol.json'),target_results_do_not_change_training=True,
        classification='greedy bank-constrained legal feasibility, not upper bounds; annotations never enter production modules'))
    with cpu_batch(),ProcessPoolExecutor(max_workers=4) as pool:
        result=[]
        for i,r in enumerate(pool.map(one,rows)):
            result.append(r)
            if i%20==0:print('evaluation-only coverage',i+1,199,flush=True)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in result]
    pd.DataFrame(flat).to_csv(OUT/'coverage_rows.csv',index=False)
    from .evaluate import run as evaluate
    evaluate(['Oracle_fixed','Oracle_augmented'],workers=4)

if __name__=='__main__':run()
