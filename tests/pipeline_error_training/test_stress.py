import numpy as np
import torch

from pipeline_error_training.stress import perturb


def test_fixed_perturbation_is_deterministic_and_node_batch_invariant():
    recipe = dict(sampling_scale=1.02, center_shift_um_zyx=[.4,.2,.2], contrast=.9, brightness=.9,
                  gamma=1.1, seed=20260915, read_noise_std=.01, blur_kernel=[.25,.5,.25])
    patch = torch.linspace(0, 1, 2*3*1*4*8*8).reshape(2,3,1,4,8,8)
    nodes = np.array([[19,2,4,5,6],[21,3,5,6,7]])
    both = perturb(patch, nodes, [0,1], [[1.625,.40625,.40625]], recipe, 'image')
    one = perturb(patch[1:], nodes, [1], [[1.625,.40625,.40625]], recipe, 'image')
    torch.testing.assert_close(both[1:], one, rtol=0, atol=0)
    assert torch.isfinite(both).all() and both.min()>=0 and both.max()<=1
