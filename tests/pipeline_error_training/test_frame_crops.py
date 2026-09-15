import numpy as np
import pytest

from pipeline_error_training.fast_crops import values
from pipeline_error_training.frame_crops import quantized_frames


def test_normalized_frames_match_original_boundary_and_rounding():
    rng = np.random.default_rng(12)
    frame = rng.integers(0, 65536, (11, 15, 19), dtype=np.uint16)
    frame[0, 0, :4] = [0, 1, 65534, 65535]
    lo, hi = np.quantile(frame, [.01, .99])
    half, direct = quantized_frames(frame, lo, hi)
    for position in [(0, 0, 0), (5, 7, 9), (10, 14, 18), (11, 15, 19)]:
        for scale, volume in [(1, half), (2, direct)]:
            shape = (4, 6, 8)
            reference, mask = values(frame, position, scale, shape)
            expected = np.rint(np.clip((reference-lo)/max(float(hi-lo), 1.), 0, 1)*mask*255).astype(np.uint8)
            axes = [c+scale*np.arange(n)-(n//2 if scale==1 else n-1) for c,n in zip(position,shape)]
            clipped = [np.clip(a,0,s-1) for a,s in zip(axes,volume.shape)]
            actual = volume[np.ix_(*clipped)]*mask
            np.testing.assert_array_equal(actual, expected)


def test_frame_sampler_rejects_unverified_dtype():
    with pytest.raises(ValueError, match='uint16'):
        quantized_frames(np.zeros((3,3,3), np.float32), 0., 1.)
