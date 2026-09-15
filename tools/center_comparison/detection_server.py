"""Read-only local API for the exhaustive detection review and native image frames."""
from __future__ import annotations

import gzip
import json
import sqlite3
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

FILTERS = {
    'all': "kind!='matched'",
    'detection': "entity='gt' AND kind!='matched'",
    'missing': "entity='gt' AND raw_count=0",
    'selection': "kind='proposal_only'",
    'association': "entity IN ('edge','pred_edge')",
    'ambiguous': 'ambiguous=1',
    'offset': "entity='gt' AND distance>=3",
    'annotations': "entity='gt'",
}


@lru_cache(maxsize=8)
def snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def connect(root):
    path = root / 'detection-review/review.sqlite'
    if not path.is_file():
        raise FileNotFoundError('The detection index is not ready. Run center_comparison index-detection, then reload.')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def clip_info(db, dataset, model):
    c = db.execute('SELECT info FROM clips WHERE dataset=?', (dataset,)).fetchone()
    m = db.execute('SELECT info FROM models WHERE dataset=? AND model=?', (dataset, model)).fetchone()
    if c is None or m is None:
        raise ValueError('Unknown clip or pipeline')
    return json.loads(c[0]), json.loads(m[0])


def catalog(db):
    meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
    clips = [json.loads(r[0]) for r in db.execute('SELECT info FROM clips ORDER BY dataset')]
    models = [json.loads(r[0]) for r in db.execute('SELECT info FROM models')]
    counts = {}
    for model in meta['models']:
        counts[model] = {name: db.execute(f'SELECT COUNT(*) FROM cases WHERE model=? AND ({where})', (model,)).fetchone()[0]
                         for name, where in FILTERS.items()}
        counts[model]['unlabeled'] = sum(r['unmatched'] for r in models if r['model'] == model)
    return dict(meta, clips=[{k: c[k] for k in ['dataset', 'embryo', 'shape', 'annotations', 'raw_predictions', 'raw_unmatched']} for c in clips],
                counts=counts, raw_unmatched=sum(c['raw_unmatched'] for c in clips),
                predictions={model: sum(r['predictions'] for r in models if r['model'] == model) for model in meta['models']})


def list_cases(root, db, params):
    model = params.get('model', 'pooled'); kind = params.get('filter', 'all')
    dataset = params.get('dataset', '')
    frame = int(params['t']) if params.get('t') else None
    if frame is not None and not 0 <= frame < 100000:
        raise ValueError('Invalid case frame')
    offset = max(0, min(100000000, int(params.get('offset', 0))))
    limit = max(1, min(100, int(params.get('limit', 40))))
    if model not in ['pooled', 'best']:
        raise ValueError('Unknown pipeline')
    if kind == 'unlabeled':
        return list_unlabeled(root, db, model, dataset, offset, limit, params.get('pool', 'final'), frame)
    if kind not in FILTERS:
        raise ValueError('Unknown review filter')
    where = f'model=? AND ({FILTERS[kind]})'; values = [model]
    if dataset:
        where += ' AND dataset=?'; values.append(dataset)
    if frame is not None:
        where += ' AND t=?'; values.append(frame)
    total = db.execute(f'SELECT COUNT(*) FROM cases WHERE {where}', values).fetchone()[0]
    if params.get('key'):
        selected = db.execute(f'SELECT priority,dataset,t,key FROM cases WHERE {where} AND key=?', [*values, params['key']]).fetchone()
        if selected:
            rank = db.execute(f'SELECT COUNT(*) FROM cases WHERE {where} AND (priority,dataset,t,key)<(?,?,?,?)', [*values, *selected]).fetchone()[0]
            offset = rank // limit * limit
    if offset >= total:
        offset = max(0, (total-1)//limit*limit)
    rows = db.execute(f'SELECT * FROM cases WHERE {where} ORDER BY priority,dataset,t,key LIMIT ? OFFSET ?', [*values, limit, offset])
    return dict(total=total, offset=offset, limit=limit, rows=[dict(row) for row in rows])


@lru_cache(maxsize=1024)
def unmatched_frame_count(path, pool, frame):
    data = snapshot(path)
    nodes, matches = (data['raw'], data['raw_matches']) if pool == 'raw' else (data['nodes'], data['matches'])
    same = nodes[nodes[:, 1] == frame]
    return int((~np.isin(same[:, 0], matches[:, 0].astype(np.int64))).sum())


def list_unlabeled(root, db, model, dataset, offset, limit, pool, frame=None):
    if pool not in ['final', 'raw']:
        raise ValueError('Unknown candidate pool')
    clips = [r[0] for r in db.execute('SELECT dataset FROM clips' + (' WHERE dataset=?' if dataset else '') + ' ORDER BY dataset', (dataset,) if dataset else ())]
    inventory = []
    for name in clips:
        c, m = clip_info(db, name, model)
        path = str(root / 'detection-review/snapshots' / (c['snapshot'] if pool == 'raw' else m['snapshot']))
        count = unmatched_frame_count(path, pool, frame) if frame is not None else c['raw_unmatched'] if pool == 'raw' else m['unmatched']
        inventory.append((name, c, m, count))
    total = sum(count for _, _, _, count in inventory)
    rows = []; skip = offset
    for name, common_info, model_info, count in inventory:
        if skip >= count:
            skip -= count; continue
        data = snapshot(str(root / 'detection-review/snapshots' / (common_info['snapshot'] if pool == 'raw' else model_info['snapshot'])))
        nodes = data['raw' if pool == 'raw' else 'nodes']
        matches = data['raw_matches' if pool == 'raw' else 'matches']
        unassigned = nodes[~np.isin(nodes[:, 0], matches[:, 0].astype(np.int64))]
        if frame is not None:
            unassigned = unassigned[unassigned[:, 1] == frame]
        if len(unassigned) != count:
            raise ValueError('Unknown-prediction count drift')
        for n in unassigned[skip:skip + limit - len(rows)]:
            code = 'r' if pool == 'raw' else 'p'
            rows.append(dict(key=f'{model}:{name}:{code}:{n[0]}', model=model, dataset=name,
                entity='raw_prediction' if pool == 'raw' else 'prediction', kind='unlabeled',
                ambiguous=1, t=int(n[1]), node=int(n[0]), other=None, distance=None,
                raw_distance=None, final_nearest=None, raw_count=None, final_count=None))
        skip = 0
        if len(rows) == limit:
            break
    return dict(total=total, offset=offset, limit=limit, rows=rows)


def get_case(root, db, key):
    stored = db.execute('SELECT * FROM cases WHERE key=?', (key,)).fetchone()
    if stored:
        case = dict(stored)
    else:
        pieces = key.split(':')
        if len(pieces) != 4 or pieces[2] not in ['p', 'r']:
            raise ValueError('Unknown review case')
        model, dataset, code, node = pieces
        clip, mi = clip_info(db, dataset, model)
        pool = snapshot(str(root / 'detection-review/snapshots' / (clip['snapshot'] if code == 'r' else mi['snapshot'])))
        nodes = pool['raw' if code == 'r' else 'nodes']; matches = pool['raw_matches' if code == 'r' else 'matches']
        row = nodes[nodes[:, 0] == int(node)]
        if len(row) != 1 or int(node) in set(matches[:, 0].astype(int)):
            raise ValueError('This is not an unmatched prediction')
        case = dict(key=key, model=model, dataset=dataset, entity='raw_prediction' if code == 'r' else 'prediction',
                    kind='unlabeled', ambiguous=1, node=int(node), other=None, t=int(row[0, 1]))
    clip, mi = clip_info(db, case['dataset'], case['model'])
    common = snapshot(str(root / 'detection-review/snapshots' / clip['snapshot']))
    final = snapshot(str(root / 'detection-review/snapshots' / mi['snapshot']))
    gt, raw, pred = common['gt'], common['raw'], final['nodes']
    raw_confidence = {int(n[0]): float(score) for n, score in zip(raw, common['confidence'], strict=True)}
    lookups = {method: {int(n[0]): n for n in nodes} for method, nodes in [('gt', gt), ('raw', raw), ('final', pred)]}
    mappings = {method: {int(p): int(g) for p, g, _ in matches} for method, matches in [('raw', common['raw_matches']), ('final', final['matches'])]}
    reverse = {method: {g: p for p, g in mapping.items()} for method, mapping in mappings.items()}
    focus = {'gt': [], 'raw': [], 'final': []}
    if case['entity'] == 'gt':
        focus['gt'] = [case['node']]
    elif case['entity'] == 'edge':
        focus['gt'] = [case['node'], case['other']]
    else:
        method = 'raw' if case['entity'] == 'raw_prediction' else 'final'
        focus[method] = [case['node']] + ([case['other']] if case['entity'] == 'pred_edge' else [])
        focus['gt'] = [mappings[method][p] for p in focus[method] if p in mappings[method]]
    target_rows = [lookups[m][i] for m in focus for i in focus[m]]
    anchor_row = next((n for n in reversed(target_rows) if n[1] == case['t']), target_rows[0])
    # Include nearby/assigned proposals as evidence, with explicit nearest status.
    for method, nodes in [('raw', raw), ('final', pred)]:
        for gid in focus['gt']:
            g = lookups['gt'][gid]
            p = reverse[method].get(gid)
            if p is not None:
                focus[method].append(p)
            same = nodes[nodes[:, 1] == g[1]]
            if len(same):
                d = np.linalg.norm((same[:, 2:] - g[2:]) * clip['spacing'], axis=1)
                focus[method].append(int(same[d.argmin(), 0]))
        focus[method] = sorted(set(focus[method]))
    expected = [list(map(int, e)) for e in common['gt_edges'] if int(e[0]) in focus['gt'] or int(e[1]) in focus['gt']]
    actual = [list(map(int, e)) for e in final['edge_status'] if int(e[0]) in focus['final'] or int(e[1]) in focus['final']]
    # Every incident endpoint is included even if the actual link skips frames.
    event_times = {case['t'], *[int(n[1]) for n in target_rows]}
    for a, b in expected:
        event_times.update([int(lookups['gt'][a][1]), int(lookups['gt'][b][1])])
    for a, b, _ in actual:
        event_times.update([int(lookups['final'][a][1]), int(lookups['final'][b][1])])
    times = sorted({t for source_t in event_times for t in range(max(0, source_t-2), min(clip['shape'][0], source_t+3))})
    status_map = {(int(a), int(b)): int(s) for a, b, s in final['edge_status']}
    expected = [dict(source=lookups['gt'][a].tolist(), target=lookups['gt'][b].tolist(),
                     recovered=status_map.get((reverse['final'].get(a), reverse['final'].get(b))) == 1) for a, b in expected]
    actual = [dict(source=lookups['final'][a].tolist(), target=lookups['final'][b].tolist(), status=status) for a, b, status in actual]
    frames = []
    for t in times:
        points = {method: nodes[nodes[:, 1] == t].tolist() for method, nodes in [('gt', gt), ('raw', raw), ('final', pred)]}
        gids = {n[0] for n in points['gt']}
        frames.append(dict(t=t, points=points, matches={method: [[int(p), int(g), float(d)] for p, g, d in matches if int(g) in gids]
            for method, matches in [('raw', common['raw_matches']), ('final', final['matches'])]}))
    neighbors = {}
    for method, nodes in [('raw', raw), ('final', pred)]:
        same = nodes[nodes[:, 1] == case['t']]
        distances = np.linalg.norm((same[:, 2:] - anchor_row[2:]) * clip['spacing'], axis=1)
        order = np.argsort(distances, kind='stable')[:max(8, int((distances <= 7).sum()))]
        neighbors[method] = [dict(id=int(same[i, 0]), zyx=same[i, 2:].tolist(), distance=float(distances[i]),
            assigned_gt=mappings[method].get(int(same[i, 0])),
            confidence=raw_confidence[int(same[i, 0])] if method == 'raw' else None) for i in order]
    if case['entity'] == 'gt':
        # Annotated rows' anchor is ground truth; prediction rows use the prediction
        # itself. Keep that difference explicit in the neighboring-center table.
        anchor_label = f"annotation {case['node']}"
    else:
        anchor_label = f"{'raw' if case['entity'] == 'raw_prediction' else 'final'} / event anchor"
    return dict(case=case, clip={k: clip[k] for k in ['dataset', 'shape', 'spacing', 'display_limits']},
                name=mi['name'], focus=focus, anchor=anchor_row[2:].tolist(), anchor_label=anchor_label,
                frames=frames, expected=expected, actual=actual, neighbors=neighbors,
                provenance=dict(prediction_sha256=mi['source_sha256'], raw_sha256=clip['raw_sha256'],
                                gt_graph_hash=clip['gt_graph_hash']))


@lru_cache(maxsize=6)
def gray_frame(image_root, t, shape, limits, metadata_sha):
    import zarr
    folder = Path(image_root)
    chunk = folder / f'0/c/{t}/0/0/0'
    if not chunk.is_file() or chunk.stat().st_size == 0:
        raise FileNotFoundError(f'Canonical image chunk missing for frame {t}')
    array = zarr.open_group(str(folder), mode='r')['0']
    if tuple(array.shape) != shape or not 0 <= t < shape[0]:
        raise ValueError('Canonical image dimensions changed')
    raw = np.asarray(array[t])
    lo, hi = limits
    gray = np.clip((raw.astype(np.float32)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)
    return gzip.compress(gray.tobytes(), compresslevel=3, mtime=0)


def make_handler(output):
    root = Path(output).resolve()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root / 'site'), **kwargs)

        def do_GET(self):
            url = urlsplit(self.path)
            if not url.path.startswith(('/api/detection/', '/api/tracking/')):
                return super().do_GET()
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            db = None
            try:
                if url.path.startswith('/api/tracking/'):
                    from . import tracking_server
                    db = tracking_server.connect_tracking(root)
                    if url.path == '/api/tracking/report':
                        result = tracking_server.report(db)
                    elif url.path == '/api/tracking/cases':
                        result = tracking_server.list_cases(db, params)
                    elif url.path == '/api/tracking/case':
                        result = tracking_server.get_case(root, db, params.get('key', ''))
                    else:
                        raise ValueError('Unknown tracking API route')
                    return self.respond(json.dumps(result, allow_nan=False, separators=(',', ':')).encode(), 'application/json')
                db = connect(root)
                if url.path == '/api/detection/catalog':
                    result = catalog(db)
                elif url.path == '/api/detection/cases':
                    result = list_cases(root, db, params)
                elif url.path == '/api/detection/case':
                    result = get_case(root, db, params.get('key', ''))
                elif url.path == '/api/detection/frame':
                    clip, _ = clip_info(db, params.get('dataset', ''), 'pooled')
                    t = int(params.get('t', '-1'))
                    if not 0 <= t < clip['shape'][0]:
                        raise ValueError('Frame outside this clip')
                    content = gray_frame(clip['image_root'], t, tuple(clip['shape']), tuple(clip['display_limits']), clip['image_metadata_sha256'])
                    return self.respond(content, 'application/octet-stream')
                else:
                    raise ValueError('Unknown detection API route')
                self.respond(json.dumps(result, allow_nan=False, separators=(',', ':')).encode(), 'application/json')
            except (ValueError, KeyError) as error:
                self.respond(json.dumps({'error': str(error)}).encode(), 'application/json', 400)
            except FileNotFoundError as error:
                self.respond(json.dumps({'error': str(error)}).encode(), 'application/json', 404)
            except Exception as error:
                self.log_error('Detection API: %s', error)
                self.respond(json.dumps({'error': 'The review could not load this data. Check the local server log.'}).encode(), 'application/json', 500)
            finally:
                if db is not None:
                    db.close()

        def respond(self, content, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A superseded frame request was canceled by the browser.
    return Handler
