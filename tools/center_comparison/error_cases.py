"""Explain scored failures using actual centers and graph edges, without new matching."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


CATEGORIES = {
    'center_missing': dict(group='centers', title='Missing center', short='No center nearby',
                           description='No predicted center lies within the matching radius.'),
    'center_conflict': dict(group='centers', title='Center assignment conflict', short='Competing assignments',
                            description='A center is nearby, but the one-to-one assignment does not give it to this cell.'),
    'center_offset': dict(group='centers', title='Center is displaced', short='Center offset ≥3 µm',
                          description='The cell is matched, but its predicted center is displaced.'),
    'link_missing': dict(group='links', title='Detected cells are not linked', short='Missing connection',
                         description='Both centers are matched. Their expected connection is absent.'),
    'link_wrong': dict(group='links', title='Incorrect connection', short='Wrong / extra connection',
                       description='The predicted connection disagrees with the annotated track.'),
    'link_detection': dict(group='links', title='Connection fails at a missing center', short='Center failure breaks link',
                           description='At least one endpoint has no matched center. This is not a pure linking failure.'),
    'division_missing': dict(group='divisions', title='Division not recovered', short='Missed division',
                             description='The prediction does not recover the annotated split into two daughters.'),
    'division_extra': dict(group='divisions', title='Incorrect division', short='Extra / incorrect division',
                           description='The official division check rejects this predicted split.'),
}


class Scene:
    """Small, explicitly paired scene. A/B/C name annotated observations, not tracks."""
    def __init__(self, nodes, gt_nodes, matches, spacing):
        self.pred = nodes if isinstance(nodes, dict) else {int(n[0]): list(map(int, n)) for n in nodes}
        self.gt = gt_nodes if isinstance(gt_nodes, dict) else {int(n[0]): list(map(int, n)) for n in gt_nodes}
        self.matches = matches
        self.reverse = {g: p for p, g in matches.items()}
        self.spacing = np.asarray(spacing)
        self.points = {}
        self.cells = []

    def prediction(self, p, label=None, role='actual'):
        key = f'p:{p}'
        if key not in self.points:
            row = self.pred[p]
            self.points[key] = dict(key=key, id=p, t=row[1], zyx=row[2:], kind='prediction',
                                    label=label or f'Pred {sum(q["kind"] == "prediction" for q in self.points.values()) + 1}', role=role)
        elif label is not None:
            self.points[key].update(label=label, role=role)
        return key

    def annotation(self, g):
        key = f'g:{g}'
        if key not in self.points:
            label = chr(ord('A') + len(self.cells))
            row = self.gt[g]
            p = self.reverse.get(g)
            distance = None if p is None else float(np.linalg.norm((np.array(self.pred[p][2:]) - row[2:]) * self.spacing))
            self.points[key] = dict(key=key, id=g, t=row[1], zyx=row[2:], kind='annotation', label=label,
                                    role='found' if p is not None else 'missing')
            pk = self.prediction(p, f'Pred {label}', 'matched') if p is not None else None
            self.cells.append(dict(label=label, t=row[1], annotation=key, prediction=pk, distance_um=distance))
        return key

    def make(self, expected, actual, edge_status, times):
        # Add annotated nodes in temporal order so labels remain A→B (→C).
        for g in sorted({g for edge in expected for g in edge}, key=lambda g: (self.gt[g][1], g)):
            self.annotation(g)
        for a, b in actual:
            for p in (a, b):
                g = self.matches.get(p)
                if g is not None:
                    self.annotation(g)
                else:
                    self.prediction(p)
        expected_rows = [dict(source=f'g:{a}', target=f'g:{b}', status='expected',
                              recovered=edge_status.get(f'{self.reverse.get(a)}:{self.reverse.get(b)}') == 'tp') for a, b in expected]
        actual_rows = [dict(source=f'p:{a}', target=f'p:{b}', status=edge_status.get(f'{a}:{b}', 'unknown')) for a, b in actual]
        return dict(points=list(self.points.values()), cells=self.cells,
                    expected=expected_rows, actual=actual_rows, times=sorted(set(times)))


def tracking_cases(diagnosis, nodes, edges, gt_nodes, gt_edges, spacing):
    """One case per missed annotated connection; attach related FP flags to it.

    The raw official flags remain unchanged. Several annotated connections may
    share a contradictory predicted edge; the full score always counts it once.
    """
    pn = {int(n[0]): n.tolist() for n in nodes}
    gn = {int(n[0]): n.tolist() for n in gt_nodes}
    pred_out, pred_in, gt_out, gt_in = (defaultdict(list) for _ in range(4))
    for a, b in edges:
        pred_out[int(a)].append(int(b)); pred_in[int(b)].append(int(a))
    for a, b in gt_edges:
        gt_out[int(a)].append(int(b)); gt_in[int(b)].append(int(a))
    reverse = {g: p for p, g in diagnosis['matches'].items()}
    fp = {(e['segments'][0]['source'][0], e['segments'][0]['target'][0]): e
          for e in diagnosis['events'] if e['kind'] == 'edge_fp'}
    covered = set()
    cases = []

    def make(event, category, expected, actual, related, explanation, times):
        scene = Scene(pn, gn, diagnosis['matches'], spacing).make(
            expected, actual, diagnosis['edge_status'], times)
        return dict(event, category=category, title=CATEGORIES[category]['title'], explanation=explanation,
                    scene=scene, related_flags=list(dict.fromkeys([event['id'], *related])),
                    radius_um=7., link_present=False if event['kind'] == 'edge_fn' else None)

    for e in diagnosis['events']:
        if e['kind'] != 'edge_fn':
            continue
        a, b = e['segments'][0]['source'][0], e['segments'][0]['target'][0]
        pa, pb = reverse.get(a), reverse.get(b)
        actual = sorted({*((pa, q) for q in pred_out[pa]), *((p, pb) for p in pred_in[pb])})
        related_edges = set(actual) & fp.keys()
        covered.update(related_edges)
        if pa is None or pb is None:
            category = 'link_detection'
            missing = [str(gn[g][1]) for g in (a, b) if g not in reverse]
            explanation = f'The expected track cannot be matched because the cell has no assigned center in frame {" and ".join(missing)}. Inspect the center first; a missing edge alone does not prove a linking-model error.'
        elif related_edges:
            category = 'link_wrong'
            explanation = 'Both annotated centers are matched. The model connects a different pair instead of the expected pair. Compare the expected A → B connection with the actual predicted connection below.'
        else:
            category = 'link_missing'
            explanation = 'Both annotated centers are matched, but there is no predicted edge between them. This isolates a frame-to-frame linking failure from a center-detection failure.'
        # Preserve a correctly linked second daughter in a division. Hiding it
        # would make a partial split look like a wholly wrong continuation.
        expected = [(a, child) for child in gt_out[a]]
        case = make(e, category, expected, actual, [fp[p]['id'] for p in sorted(related_edges)],
                    explanation, [gn[a][1], gn[b][1]])
        case['scene']['focus_connection'] = dict(source=f'g:{a}', target=f'g:{b}')
        if category == 'link_missing' and len(expected) > 1:
            case.update(title='One daughter connection is missing',
                        explanation='The parent and the missing daughter both have matched centers. The model does not connect that daughter. The other annotated branch is shown too, so a correctly recovered daughter is not mistaken for a wrong connection.')
        cases.append(case)
    for edge, e in fp.items():
        if edge in covered:
            continue
        a, b = edge
        ga, gb = diagnosis['matches'].get(a), diagnosis['matches'].get(b)
        expected = sorted({*((ga, g) for g in gt_out[ga]), *((g, gb) for g in gt_in[gb])})
        actual = sorted({edge, *((a, q) for q in pred_out[a]), *((p, b) for p in pred_in[b])})
        cases.append(make(e, 'link_wrong', expected, actual, [],
            'The model adds a connection that contradicts an evaluable annotated track. A predicted endpoint with no annotation match is shown as unknown; it is not automatically a false-positive cell.',
            [pn[a][1], pn[b][1]]))
    for e in diagnosis['events']:
        if not e['kind'].startswith('division_'):
            continue
        g = e['node'] if e['kind'] == 'division_fn' else diagnosis['matches'].get(e['node'])
        p = reverse.get(g) if e['kind'] == 'division_fn' else e['node']
        expected = [(g, b) for b in gt_out[g]]
        actual = [(p, b) for b in pred_out[p]]
        category = 'division_missing' if e['kind'] == 'division_fn' else 'division_extra'
        explanation = ('The annotated parent splits into two daughters, but the official division check does not recover that split.'
                       if category == 'division_missing' else 'The model predicts a split that the official division check rejects.')
        explanation += ' Center matches below use the full-frame 7 µm assignment; the division scorer also checks its own local matching and branch topology.'
        cases.append(make(e, category, expected, actual, [], explanation, [e['t'], e['t'] + 1]))
    return cases


def center_case(event, frame, method, radius):
    """Same-frame center diagnosis, with nearest and assigned kept distinct."""
    def native(rows):
        return [[p[0], frame['t'], *p[1:4]] for p in rows]
    matches = {p: g for p, g, _ in frame['matches'][method][str(radius)]}
    scene = Scene(native(frame['points'][method]), native(frame['points']['gt']), matches, frame['spacing'])
    g = event['node']
    scene.annotation(g)
    near = next((r for r in frame['model_nearest'][method] if r[0] == g), None)
    if event['kind'] == 'offset':
        category = 'center_offset'
        explanation = f'This cell was detected, but the assigned center is {event["distance_um"]:.2f} µm from its annotation. This is a position error in one frame; it does not by itself establish a tracking error.'
    else:
        nearby = near is not None and near[2] <= float(radius)
        category = 'center_conflict' if nearby else 'center_missing'
        explanation = ('A predicted center lies within the radius, but optimal one-to-one matching leaves this annotation unmatched. This is an assignment conflict, not proof that the detector produced no center.'
                       if nearby else f'No predicted center lies within {radius} µm of this annotated cell. Check the marked location in the image. There is no frame-to-frame decision in this view.')
        if near:
            explanation += f' The nearest prediction is {near[2]:.2f} µm away.'
            if near[1] in matches:
                scene.annotation(matches[near[1]])
            else:
                scene.prediction(near[1], 'Nearest prediction', 'nearest')
    result = scene.make([], [], {}, [frame['t']])
    target = scene.reverse.get(g) if event['kind'] == 'offset' else (near[1] if near else None)
    if target is not None:
        delta = (np.asarray(scene.pred[target][2:]) - scene.gt[g][2:]) * scene.spacing
        result['center_probe'] = dict(annotation=f'g:{g}', prediction=f'p:{target}',
                                      assigned=scene.reverse.get(g) == target, delta_um=delta.tolist(),
                                      distance_um=float(np.linalg.norm(delta)))
    return dict(event, category=category, title=CATEGORIES[category]['title'], explanation=explanation,
                scene=result, radius_um=float(radius), related_flags=[event['id']], nearest_um=None if near is None else near[2])
