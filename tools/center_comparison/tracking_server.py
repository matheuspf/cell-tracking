"""Read-only report and paginated tracking-case API; native images use the existing API."""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache

from .detection_server import clip_info, connect, snapshot
from .error_cases import Scene
from .pipeline import sha256
from .tracking_index import CATEGORIES, STAGES, adjacency


@lru_cache(maxsize=2)
def database_digest(path, modified, size):
    return sha256(path)


def connect_tracking(root):
    path = root / 'tracking-review/review.sqlite'
    if not path.is_file():
        raise FileNotFoundError('Run center_comparison index-tracking to generate the full pipeline report and tracking review.')
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    source = root / 'detection-review/review.sqlite'
    stat = source.stat()
    expected = json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])['detection_database_sha256']
    if database_digest(source, stat.st_mtime_ns, stat.st_size) != expected:
        db.close()
        raise ValueError('Detection index changed. Rebuild index-tracking before reviewing these graphs.')
    return db


def report(db):
    return json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])


def list_cases(db, params):
    model = params.get('model', 'pooled')
    category = params.get('filter', 'all')
    stage = params.get('stage', '')
    if model not in ['pooled', 'best'] or category not in ['all', *CATEGORIES] or stage not in ['', *STAGES]:
        raise ValueError('Unknown pipeline, error category or stage')
    where, values = 'model=?', [model]
    for column, value in [('category', '' if category=='all' else category), ('stage',stage),
                          ('embryo',params.get('embryo','')), ('dataset',params.get('dataset',''))]:
        if value:
            where += f' AND {column}=?'; values.append(value)
    if params.get('t', '') != '':
        where += ' AND t=?'; values.append(int(params['t']))
    total = db.execute(f'SELECT COUNT(*) FROM cases WHERE {where}',values).fetchone()[0]
    limit = max(1,min(100,int(params.get('limit',40))))
    offset = max(0,int(params.get('offset',0)))
    if params.get('key'):
        row = db.execute(f'SELECT dataset,t,key FROM cases WHERE {where} AND key=?',[*values,params['key']]).fetchone()
        if row:
            rank = db.execute(f'SELECT COUNT(*) FROM cases WHERE {where} AND (dataset,t,key)<(?,?,?)', [*values,*row]).fetchone()[0]
            offset = rank//limit*limit
    offset = min(offset,max(0,(total-1)//limit*limit))
    rows = [dict(r) for r in db.execute(f'SELECT key,model,dataset,embryo,category,stage,t,node,other FROM cases WHERE {where} ORDER BY dataset,t,key LIMIT ? OFFSET ?', [*values,limit,offset])]
    return dict(total=total,offset=offset,limit=limit,rows=rows)


def get_case(root, db, key):
    stored = db.execute('SELECT * FROM cases WHERE key=?',(key,)).fetchone()
    if stored is None:
        raise ValueError('Unknown tracking case')
    case = dict(stored); evidence = json.loads(case.pop('evidence'))
    source = connect(root)
    try:
        clip, model = clip_info(source,case['dataset'],case['model'])
        center_info = {int(r['node']):dict(r) for r in source.execute("SELECT * FROM cases WHERE model=? AND dataset=? AND entity='gt'",(case['model'],case['dataset']))}
    finally:
        source.close()
    common = snapshot(str(root / 'detection-review/snapshots' / clip['snapshot']))
    final = snapshot(str(root / 'detection-review/snapshots' / model['snapshot']))
    pn = {int(n[0]):n.tolist() for n in final['nodes']}
    gn = {int(n[0]):n.tolist() for n in common['gt']}
    full_matches = {int(p):int(g) for p,g,_ in final['matches']}
    matches = dict(evidence['local_matches']) if 'local_matches' in evidence else full_matches
    reverse = {g:p for p,g in matches.items()}
    pi, po = adjacency(final['edges']); gi, go = adjacency(common['gt_edges'])
    category = case['category']; division = category.startswith('division')
    gs, ps = set(), set()
    if category in ['edge_endpoint','edge_link']:
        gs.update([case['node'],case['other']])
        gs.update(go[case['node']])
    elif category=='division_fn':
        gs.update(evidence['local_gt']); ps.update(evidence['candidate_forks'])
    else:
        ps.add(case['node'])
        if case['other'] is not None:
            ps.add(case['other'])
    gs.update(matches[p] for p in ps if p in matches)
    # Two generations show daughter persistence without graph-wide reachability.
    if category in ['edge_fp','division_fp']:
        seeds = set(gs)
        for g in seeds:
            gs.update(gi[g]); gs.update(go[g])
            if division:
                gs.update(q for c in go[g] for q in go[c])
    ps.update(reverse[g] for g in gs if g in reverse)
    expected = sorted({(a,b) for a in gs for b in go[a] if b in gs})
    actual = {(p,c) for p in ps for c in po[p]} | {(p,c) for c in ps for p in pi[c]}
    if division:
        children = {b for _,b in actual}
        actual.update((p,c) for p in children for c in po[p])
        # Show the scoring window, not ever-expanding context past the daughters.
        # All edges for the selected flag are inside this local interval.
        context_times = [gn[g][1] for g in gs]
        if category == 'division_fp':
            context_times.extend([pn[case['node']][1]-1, pn[case['node']][1]+2])
        low, high = min(context_times), max(context_times)
        actual = {(a,b) for a,b in actual if low <= pn[a][1] <= high and low <= pn[b][1] <= high}
    actual = sorted(actual)
    status = {f'{a}:{b}': 'tp' if s==1 else 'fp' if s==-1 else 'unknown' for a,b,s in final['edge_status']}
    scene_builder = Scene(pn,gn,matches,clip['spacing'])
    for g in sorted(gs,key=lambda g:(gn[g][1],g)):
        scene_builder.annotation(g)
    scene = scene_builder.make(expected,actual,status,[])
    times = sorted({p['t'] for p in scene['points']})
    scene['times'] = times
    actual_set = set(actual)
    for edge in scene['expected']:
        a,b = int(edge['source'][2:]),int(edge['target'][2:])
        edge['present'] = (reverse.get(a),reverse.get(b)) in actual_set
        edge['focus'] = category in ['edge_endpoint','edge_link'] and (a,b)==(case['node'],case['other'])
    for edge in scene['actual']:
        a,b = int(edge['source'][2:]),int(edge['target'][2:])
        edge['focus'] = (category=='edge_fp' and (a,b)==(case['node'],case['other'])) or (category=='division_fp' and a==case['node'])
    checks = []
    for cell in scene['cells']:
        g = int(cell['annotation'][2:]); c = center_info[g]
        checks.append(dict(label=cell['label'],gt=g,t=c['t'],kind=c['kind'],raw_distance_um=c['raw_distance'],
            final_distance_um=c['distance'],final_nearest_um=c['final_nearest'],matched=bool(c['matched']),
            detection_key=f"{case['model']}:{case['dataset']}:g:{g}"))
    related = [dict(r) for r in db.execute('SELECT key,category,t,stage FROM cases WHERE model=? AND dataset=? AND key!=? AND t BETWEEN ? AND ? ORDER BY t,key LIMIT 20',
        (case['model'],case['dataset'],key,min(times),max(times))) if r['key'] != key]
    return dict(case=case,evidence=evidence,name=model['name'],scene=scene,center_checks=checks,nearby_flags=related,
        clip={k:clip[k] for k in ['dataset','shape','spacing','display_limits']},
        matching='Official division-window matching' if category=='division_fn' else 'Official whole-clip node matching',
        explanation=STAGES[case['stage']],
        provenance=dict(prediction_sha256=model['source_sha256'],gt_graph_hash=clip['gt_graph_hash'],evaluation_sha256=model['evaluation_sha256']))
