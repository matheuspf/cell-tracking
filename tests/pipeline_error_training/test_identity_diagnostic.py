import json

import numpy as np
import pytest

from pipeline_error_training import guard,selection
from pipeline_error_training.common import sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_identity_nomination_rejects_partial_or_changed_source_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'install',lambda **kwargs:None)
    monkeypatch.setattr(selection,'WORK',tmp_path)
    clips = ['44b6_first','44b6_second']
    monkeypatch.setattr(selection,'inputs',lambda:[dict(dataset=name,embryo='44b6') for name in clips])
    native = np.zeros((2,38),dtype=np.float32)
    monkeypatch.setattr(selection,'verified_evidence',lambda row:dict(edge_features=native))
    package = tmp_path/'training/A10/44b6/20260915'
    package.mkdir(parents=True)
    (package/'model.pt').write_bytes(b'fixed checkpoint')
    weight_hash = sha(package/'model.pt')
    write(package/'calibration.json',dict(clips=clips))
    write(package/'frozen_package.json',dict(recipe=dict(source='44b6',arm='A10',seed=20260915),
        weights_sha256=weight_hash,calibration_sha256=sha(package/'calibration.json')))

    def add_clip(name):
        table = tmp_path/'calibration/A10/44b6/20260915'/f'{name}.npz'
        table.parent.mkdir(parents=True,exist_ok=True)
        labels = np.array([1,0],np.int8)
        groups = np.array([3,3])
        np.savez(table,pair_score=np.array([2.,-2.]),pair_y=labels,pair_group=groups)
        write(table.with_suffix('.json'),dict(sha256=sha(table),model_sha256=weight_hash))
        pairs = tmp_path/'source/44b6/pairs'/table.name
        pairs.parent.mkdir(parents=True,exist_ok=True)
        np.savez(pairs,index=np.array([0,1]),labels=labels,group=groups)
        write(tmp_path/'source/44b6/receipts'/f'{name}.json',dict(files={str(pairs):sha(pairs)}))
        return table

    add_clip(clips[0])
    with pytest.raises(ValueError,match='Complete registered source calibration'):
        selection.identity_loss('44b6','A10')
    assert not (package/'identity_diagnostic.json').exists()
    table = add_clip(clips[1])
    result = selection.identity_loss('44b6','A10')
    assert result['clips']==2 and result['groups']==2
    assert result['raw_source_grouped_pair_nll']<result['unchanged_native_offset_grouped_pair_nll']
    table.write_bytes(b'changed after calibration')
    with pytest.raises(ValueError,match='calibration cache changed'):
        selection.identity_loss('44b6','A10')
