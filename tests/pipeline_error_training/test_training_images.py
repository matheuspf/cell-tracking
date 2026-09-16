import numpy as np
import pytest

from pipeline_error_training.training_images import FrameStatistics


def test_retained_quantiles_reject_changed_frame_and_bound_memory():
    cache = FrameStatistics(max_entries=2)
    frame = np.arange(1000,dtype=np.uint16).reshape(10,10,10)
    expected = tuple(np.quantile(frame,[.01,.99]))
    assert cache.quantiles(('clip',0),'first',frame)==expected
    assert cache.quantiles(('clip',0),'first',frame)==expected
    with pytest.raises(ValueError,match='image bytes changed'):
        cache.quantiles(('clip',0),'changed',frame)
    cache.quantiles(('clip',1),'second',frame)
    cache.quantiles(('clip',2),'third',frame)
    assert len(cache.entries)==2
    assert ('clip',0) not in cache.entries
    assert cache.computed==3
    assert cache.reused==1
