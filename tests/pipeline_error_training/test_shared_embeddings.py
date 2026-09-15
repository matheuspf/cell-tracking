from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline_error_training.shared_embeddings import embeddings_many


def model():
    result = torch.nn.Module()
    result.family = 'temporal'
    result.encoder = torch.nn.Identity()
    return result.eval()


def test_shared_encoder_alias_rejected(tmp_path):
    first,second = model(),model()
    second.encoder = first.encoder
    with pytest.raises(ValueError,match='distinct encoder'):
        embeddings_many({'a':first,'b':second},{},{},None,
            {'a':tmp_path/'a.npz','b':tmp_path/'b.npz'},{'a':'a','b':'b'})


def test_forward_hook_removed_when_original_sampler_fails(tmp_path,monkeypatch):
    first,second = model(),model()
    def failed(*args,**kwargs):
        assert first.encoder._forward_hooks
        raise RuntimeError('image read failed')
    monkeypatch.setattr('pipeline_error_training.shared_embeddings.embeddings',failed)
    with pytest.raises(RuntimeError,match='image read failed'):
        embeddings_many({'a':first,'b':second},{'image_shape':[1]},
            {'nodes':np.array([[0,0,0,0,0]])},SimpleNamespace(),
            {'a':tmp_path/'a.npz','b':tmp_path/'b.npz'},{'a':'a','b':'b'})
    assert not first.encoder._forward_hooks
    assert not second.encoder._forward_hooks
