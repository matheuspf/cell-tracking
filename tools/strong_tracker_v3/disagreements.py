"""Stable native-ID teacher projection; never infer identity from proximity."""
from __future__ import annotations

import numpy as np

from .common import load_graph

TEACHERS = ['incumbent', 'raw_neural', 'old_final', 'v2_E_native', 'v2_E_hgb']


def project_teacher(nodes, donor_nodes, donor_edges, raw_nodes, scale):
    """Coordinates may differ after smoothing, but inserted IDs are not portable."""
    current = {int(r[0]): r for r in nodes}
    native = {int(r[0]): r for r in raw_nodes}
    donor = {int(r[0]): r for r in donor_nodes}
    valid = {i for i in current.keys() & donor.keys() & native.keys()
             if current[i][1] == donor[i][1] == native[i][1]}
    edges = {(int(a), int(b)) for a, b in donor_edges if int(a) in valid and int(b) in valid
             and current[int(b)][1] == current[int(a)][1] + 1}
    displacement = [float(np.linalg.norm((current[i][2:] - donor[i][2:]) * scale)) for i in valid]
    return edges, dict(donor_nodes=len(donor), mapped_nodes=len(valid),
        unmapped_node_fraction=1-len(valid)/max(len(donor),1), donor_edges=len(donor_edges),
        mapped_edges=len(edges), unmapped_edge_fraction=1-len(edges)/max(len(donor_edges),1),
        different_centers=sum(x > 0 for x in displacement),
        center_displacement_um_max=max(displacement, default=0.),
        identity_rule='shared proven raw native ID and time; inserted donor IDs excluded')


def teachers(ctx, sample, nodes, edges, raw):
    name=sample['dataset']; scale=np.asarray(sample['physical_scale'])
    old=load_graph(ctx.v1/'baseline/public'/f'{name}.npz')
    with np.load(ctx.v2/'predictions'/f'{name}.npz',allow_pickle=False) as f:
        supplied=[('raw_neural',raw['nodes'],raw['edges']),('old_final',old['nodes'],old['edges']),
            ('v2_E_native',old['nodes'],f['E_native_m1.5__edges']),
            ('v2_E_hgb',old['nodes'],f['E_hgb_m1.5__edges'])]
    result={'incumbent':set(map(tuple,edges))}; audit={}
    for label,n,e in supplied:
        result[label],audit[label]=project_teacher(nodes,n,e,raw['nodes'],scale)
    return result,audit
