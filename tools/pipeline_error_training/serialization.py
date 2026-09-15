"""Integer, complete-graph CSV export and strict round-trip verification."""
import csv
from pathlib import Path

import numpy as np


def export_csv(path, name, nodes, edges):
    from .common import graph_hash
    path = Path(path)
    if not np.equal(nodes, np.rint(nodes)).all() or not np.equal(edges, np.rint(edges)).all():
        raise ValueError('Submission coordinates and identifiers must be integers')
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for k, node in enumerate(nodes):
            writer.writerow(dict(id=k, dataset=name, row_type='node', **dict(zip(columns[3:8], map(int, node))), source_id=-1, target_id=-1))
        for k, (a, b) in enumerate(edges, len(nodes)):
            writer.writerow(dict(id=k, dataset=name, row_type='edge', node_id=-1, t=-1, z=-1, y=-1, x=-1,
                                 source_id=int(a), target_id=int(b)))
    nn, ee = [], []
    with path.open(newline='') as stream:
        for k, row in enumerate(csv.DictReader(stream)):
            if int(row['id']) != k or row['dataset'] != name:
                raise ValueError('CSV row identity drift')
            if row['row_type'] == 'node':
                nn.append([int(row[c]) for c in columns[3:8]])
            elif row['row_type'] == 'edge':
                ee.append([int(row['source_id']), int(row['target_id'])])
            else:
                raise ValueError('Unknown CSV row type')
    if graph_hash(np.asarray(nn).reshape(-1, 5), np.asarray(ee).reshape(-1, 2)) != graph_hash(nodes, edges):
        raise ValueError('CSV round-trip changed the complete graph')
    return dict(nodes=len(nn), edges=len(ee), rows=len(nn)+len(ee), exact_roundtrip=True)


def export_geff(path, nodes, edges):
    """Preserve persisted IDs in an explicit attribute across library remapping."""
    import polars as pl
    import tracksdata as td
    from .common import graph_hash
    path = Path(path)
    graph = td.graph.IndexedRXGraph()
    for key in ['z', 'y', 'x']:
        graph.add_node_attr_key(key, pl.Float64, -999999.)
    graph.add_node_attr_key('study_node_id', pl.Int64, -1)
    graph.add_node_attr_key('study_row', pl.Int64, -1)
    ids = graph.bulk_add_nodes([dict(study_node_id=int(i), study_row=k, t=int(t), z=float(z), y=float(y), x=float(x))
                               for k, (i, t, z, y, x) in enumerate(nodes)])
    mapping = {int(n[0]): int(i) for n, i in zip(nodes, ids)}
    graph.add_edge_attr_key('study_edge_row', pl.Int64, -1)
    if len(edges):
        graph.bulk_add_edges([dict(source_id=mapping[int(a)], target_id=mapping[int(b)], study_edge_row=k)
                              for k, (a, b) in enumerate(edges)])
    graph.to_geff(str(path), overwrite=False)
    restored, _ = td.graph.IndexedRXGraph.from_geff(str(path))
    table = restored.node_attrs().sort('study_row')
    lookup = {int(i): int(j) for i, j in table.select('node_id', 'study_node_id').iter_rows()}
    nn = table.select('study_node_id', 't', 'z', 'y', 'x').to_numpy().astype(np.int64)
    ee = np.array([(lookup[int(a)], lookup[int(b)]) for a, b in restored.edge_attrs().sort('study_edge_row')
                   .select('source_id', 'target_id').iter_rows()], np.int64).reshape(-1, 2)
    if graph_hash(nn, ee) != graph_hash(nodes, edges):
        raise ValueError('GEFF round-trip changed persisted node identity or the complete graph')
    return dict(nodes=len(nn), edges=len(ee), exact_roundtrip=True, stable_id_attribute='study_node_id')
