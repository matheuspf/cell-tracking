import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pipeline_error_training.observations import ObservationBank
from pipeline_error_training.serialization import export_csv


def test_startup_guard_denies_ground_truth_old_arrays_exports_and_network(tmp_path):
    source = """
import json, pathlib, socket, sys
from pipeline_error_training.guard import install
root = pathlib.Path(sys.argv[1]); fresh = root/'fresh'; fresh.mkdir()
receipt = install(fresh_root=fresh)
assert receipt['installed_before_numerical']
for name in ['labels.geff/zarr.json', 'old.npz', 'old.npy', 'submission.csv']:
    try:
        (root/name).open('rb')
    except PermissionError:
        pass
    else:
        raise AssertionError(name)
try:
    socket.getaddrinfo('example.org', 443)
except PermissionError:
    pass
else:
    raise AssertionError('network')
(fresh/'new.npz').write_bytes(b'fresh')
assert receipt['blocked_reads'] == 4
assert receipt['blocked_network'] == 1
print(json.dumps(receipt))
"""
    result = subprocess.run([sys.executable, '-c', source, str(tmp_path)], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['installed_before_numerical']


def test_directory_relative_opens_use_actual_fd_and_deny_external_labels(tmp_path):
    source = """
import os, pathlib, shutil, sys, tempfile
from pipeline_error_training.guard import install
root=pathlib.Path(sys.argv[1]); fresh=root/'fresh'; fresh.mkdir()
(root/'labels.geff').mkdir();(root/'labels.geff/zarr.json').write_text('{}')
receipt=install(fresh_root=fresh)
with tempfile.TemporaryDirectory(dir=fresh) as tmp:
    (pathlib.Path(tmp)/'predictions').mkdir()
    (pathlib.Path(tmp)/'predictions/new.npz').write_bytes(b'own')
assert receipt['blocked_reads']==0
outside=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
os.chdir(fresh)
try:
    os.open('labels.geff/zarr.json',os.O_RDONLY,dir_fd=outside)
except PermissionError:
    pass
else:
    raise AssertionError('Relative descriptor bypassed the label guard')
os.close(outside)
assert receipt['blocked_reads']==1
assert receipt['blocked_paths'][0]['path']==str(root/'labels.geff/zarr.json')
"""
    result=subprocess.run([sys.executable,'-c',source,str(tmp_path)],env=os.environ.copy(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_closed_bank_keeps_close_distinct_centers_and_prediction_tracklet_group():
    old = np.array([[11, 0, 4, 4, 4], [29, 1, 4, 4, 4], [31, 2, 4, 4, 4],
                    [43, 1, 4, 7, 4]])
    raw_nodes = np.array([[0, 0, 4, 5, 4], [1, 1, 4, 5, 4], [2, 2, 4, 5, 4], [3, 1, 4, 7, 4]])
    raw = dict(nodes=raw_nodes, edges=np.array([[0, 1], [1, 2]]), node_probabilities=np.ones(4)*.8,
               original_graph_sha256='fixture')
    row = dict(physical_scale=[1., 1., 1.], image_shape=[3, 16, 16, 16],
               baselines={'P0': {'sha256': 'fixture'}}, raw={'sha256': 'fixture'})
    bank = ObservationBank(row, dict(nodes=old, edges=np.array([[11, 29], [29, 31]])), raw)
    assert bank.exact_alias[3] == 3
    assert bank.tube(1, 1) == [(0, 0), (1, 1), (2, 2)]
    assert (1, 1) in set(map(tuple, bank.pairs))
    # The existing second cell is preserved as a separate observation; proximity
    # alone creates a competing hypothesis, never a deduplication operation.
    np.testing.assert_array_equal(bank.nodes, old)


def test_csv_exports_all_integer_nodes_and_edges_with_exact_identity(tmp_path):
    nodes = np.array([[17, 0, 4, 5, 6], [42, 1, 4, 6, 6]])
    edges = np.array([[17, 42]])
    assert export_csv(tmp_path/'graph.csv', 'unfamiliar', nodes, edges)['exact_roundtrip']
    with pytest.raises(ValueError, match='integers'):
        export_csv(tmp_path/'bad.csv', 'x', nodes.astype(float)+.1, edges)
