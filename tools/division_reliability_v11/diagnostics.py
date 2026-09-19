"""Label-enabled source audits and post-freeze error attribution only."""
from collections import Counter
from .common import REPO,DATA,WORK,RESULTS,Blocked,read,write,sha,now


def bank_funnel(bank,labels,positive_events):
    """Independent local-window endpoints, existing anchors and daughter paths."""
    reached={int(e) for e in positive_events};events=[]
    for event,roles in labels.roles.items():
        key=int(labels.gt_reverse[event]);endpoint=roles is not None;anchors=[];pair=False
        if endpoint:
            ps,ds=roles
            candidates=ps|{q for p in ps for q in bank.succ[p]}
            anchors=sorted(candidates&bank.expanded)
            for p in anchors:
                for _,a,b,qa,qb in bank.events(p):
                    aa={a,qa} if qa>=0 else {a};bb={b,qb} if qb>=0 else {b}
                    if any(not aa.isdisjoint(d1) and not bb.isdisjoint(d2)
                           for i,d1 in enumerate(ds) for j,d2 in enumerate(ds) if i!=j):
                        pair=True;break
                if pair:break
        events.append(dict(event_id=key,endpoint_window_present=endpoint,anchor_present=bool(anchors),
                           daughter_paths_present=pair,legal_compatible_action=key in reached,anchors=anchors))
    counts={k:sum(bool(e[k]) for e in events) for k in ('endpoint_window_present','anchor_present','daughter_paths_present','legal_compatible_action')}
    return dict(division_events=len(events),stages=counts,events=events,
                interpretation='Overlapping diagnostic stages; local action existence is not an achievable score bound.')


def detection_audit(nodes,edges,gt,ge,labels):
    import numpy as np
    from scipy.spatial import cKDTree
    from annotation_selection.metric_adapter import make_graph
    from tracksdata.metrics import DistanceMatching
    import tracksdata as td
    import warnings
    scale=np.array([1.625,.40625,.40625]);pindex={int(n[0]):i for i,n in enumerate(nodes)}
    gindex={int(n[0]):i for i,n in enumerate(gt)}
    persistent=labels.persistent_matches
    reverse={g:p for p,g in persistent.items()}
    distance=[float(np.linalg.norm((nodes[pindex[p],2:]-gt[gindex[g],2:])*scale)) for p,g in persistent.items()]
    graph,_=make_graph(nodes,edges);truth,_=make_graph(gt,ge)
    matched3=0
    if len(nodes) and len(gt):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');graph.match(truth,matching=DistanceMatching(max_distance=3.,scale=tuple(scale),optimal=True))
        attrs=graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
        matched3=sum(x[0] is not None and x[0]!=-1 for x in attrs.iter_rows())
    predicted=set(map(tuple,edges));endpoints=available=correct=0
    for a,b in ge:
        if int(a) in reverse and int(b) in reverse:
            endpoints+=1
            if (reverse[int(a)],reverse[int(b)]) in predicted:correct+=1
    close7=close3=0
    for t in np.unique(gt[:,1]):
        pos=gt[gt[:,1]==t,2:]*scale
        if len(pos)>1:
            d,_=cKDTree(pos).query(pos,k=2);close7+=int((d[:,1]<=7).sum());close3+=int((d[:,1]<=3).sum())
    return dict(gt_nodes=len(gt),predicted_nodes=len(nodes),matched_nodes_7um=len(persistent),matched_nodes_3um=matched3,
        recall_7um=len(persistent)/len(gt) if len(gt) else None,recall_3um=matched3/len(gt) if len(gt) else None,
        matched_distance_quantiles_um=np.quantile(distance,[0,.5,.9,.99,1]).tolist() if distance else [],
        gt_edges=len(ge),gt_edges_both_endpoints_matched=endpoints,gt_edges_recovered_with_full_matching=correct,
        gt_edges_missing_endpoint=len(ge)-endpoints,gt_edges_endpoints_present_link_missing=endpoints-correct,
        annotated_nodes_with_other_annotated_node_within_7um=close7,
        annotated_nodes_with_other_annotated_node_within_3um=close3,
        annotation_scope='Sparse annotation recall and edge support; unmatched predictions are not automatic false positives.')


def source_summary(source,seed):
    """No model updates: aggregate complete source-fit bank/graph receipts."""
    from .readiness import require_production
    require_production('source-diagnostics')
    clips=read(WORK/'source_partitions.json')[source]['fit'];folder=WORK/'source_diagnostics'/source/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    counts=Counter();events=set();per_clip={}
    for clip in clips:
        path=WORK/'banks'/source/str(seed)/'fit'/clip/'receipt.json';r=read(path)
        counts.update(r['census']);events.update(r['positive_events'])
        counts['identity_groups']+=r['identity_groups']
        per_clip[clip]={k:r[k] for k in ('census','complete_group_fork_count','total_parent_count','fork_count_status')}
        for key in ('official_metrics','detection','funnel'):
            if key in r:per_clip[clip][key]=r[key]
    result=dict(status='complete',source=source,seed=seed,fit_clips=len(clips),census=dict(counts),
        distinct_positive_events=len(events),per_clip=per_clip,scope='complete source-fit C00 observation bank',finished_utc=now())
    write(folder/'summary.json',result);return result


def target_details(source,seed,arm,clip,nodes,edges,events):
    """Called only inside the globally frozen target evaluation worker."""
    import numpy as np
    from .data import labels as load_labels
    from .actions import Bank,utilities
    from .graphs import read_csv
    from pipeline_error_training.labels import SourceLabels
    baseline=WORK/'predictions/C00'/source/str(seed)/clip
    with np.load(baseline/'graph.npz') as f:bn,be,bs=(f[k] for k in ('nodes','edges','edge_scores'))
    gt,ge=load_labels(DATA/'train'/f'{clip}.geff')
    lab=SourceLabels(bn,be,gt,ge,[1.625,.40625,.40625]);bank=Bank(bn,be,bs,100)
    potential=set()
    for roles in lab.roles.values():
        if roles:
            ps,_=roles;potential.update((ps|{q for p in ps for q in bank.succ[p]})&bank.expanded)
    legal_events=set();groups={};event_groups={}
    for p in sorted(potential):
        g=bank.parent(p)
        if not g['complete']:continue
        raw=[lab.decision(d) for d in g['forks']]
        for i,r in enumerate(raw):
            for e in r['compatible_events']:
                legal_events.add(e);event_groups.setdefault(int(e),[]).append((p,i))
        groups[p]=(g,raw)
    funnel=bank_funnel(bank,lab,legal_events)
    result=dict(funnel=funnel,action_rejections=dict(bank.alternatives.rejections))
    if arm=='C00':
        result['detection']=detection_audit(nodes,edges,gt,ge,lab)
        mapping={int(n[0]):i for i,n in enumerate(bn)};reverse={g:p for p,g in lab.persistent_matches.items()}
        result['detection']['gt_edges_in_clean_candidate_union']=sum(int(a) in reverse and int(b) in reverse and
            (mapping[reverse[int(a)]],mapping[reverse[int(b)]]) in bank.logits for a,b in ge)
        return result
    prediction=WORK/'predictions'/arm/source/str(seed)/clip
    with np.load(prediction/'policy_logits.npz') as f:cache={k:f[k] for k in f.files}
    ix={int(p):i for i,p in enumerate(cache['parents'])}
    cal=read(WORK/'packages'/source/str(seed)/arm/'calibration.json')
    raw_positive=set();calibrated_positive=set();rank1=set();rank1_known=set()
    for e,pairs in event_groups.items():
        for p,i in pairs:
            if p not in ix:raise Blocked('Compatible deployment group absent from frozen raw score denominator')
            j=ix[p];a,b=cache['offset'][j:j+2];cond=cache['conditional'][a:b];occ=float(cache['occurrence'][j]);g,raw=groups[p]
            if len(cond)!=len(g['forks']):raise Blocked('Frozen conditional action order/length differs')
            if cond[i]>=cond.max()-1e-12:rank1.add(e)
            known=np.array([r['metric_fork_target']>=0 for r in raw])
            if cond[i]>=cond[known].max()-1e-12:rank1_known.add(e)
            if utilities(bank,g,occ,cond,0)[i]>0:raw_positive.add(e)
            if not cal['disabled_policy'] and utilities(bank,g,occ/cal['temperature']+cal['intercept'],cond,cal['margin'])[i]>0:
                calibrated_positive.add(e)
    trace=read(prediction/'policy_trace.json');tail=[]
    result['deployment_census']=read(prediction/'policy_progress.json')
    for gain,p,occ,n in trace['raw_top_groups']:
        g=bank.parent(int(p))
        if not g['complete']:raise Blocked('Stored score tail contains incomplete denominator')
        raw=[lab.decision(d) for d in g['forks']];risks=[r['metric_fork_target'] for r in raw]
        y=1 if 1 in risks else 0 if risks and all(r==0 for r in risks) else -1
        j=ix[int(p)];a,b=cache['offset'][j:j+2];cond=cache['conditional'][a:b]
        u=utilities(bank,g,occ,cond,0);top=int(np.argmax(u))
        tail.append(dict(parent=int(p),raw_gain=gain,raw_occurrence_logit=occ,group_target=y,
                         highest_raw_gain_action_risk=risks[top],alternatives=len(risks),
                         conditional_top_action_risk=risks[int(np.argmax(cond))]))
    recovered=set(events['recovered_gt_events'])
    for e in funnel['events']:
        eid=e['event_id'];e.update(conditional_top1=eid in rank1,conditional_top1_supported_only=eid in rank1_known,
            positive_raw_gain=eid in raw_positive,positive_calibrated_margin_gain=eid in calibrated_positive,
            actually_recovered=eid in recovered)
    result.update(raw_tail=tail,disabled_policy=cal['disabled_policy'],solver=trace['solver'],
        stage_counts=dict(legal_events=len(legal_events),conditional_top1=len(rank1),conditional_top1_supported_only=len(rank1_known),
            positive_raw_gain=len(raw_positive),positive_calibrated_margin_gain=len(calibrated_positive),actually_recovered=len(recovered)))
    return result
