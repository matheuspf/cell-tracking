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


def test_compact_stress_still_uses_the_full_native_perturbation(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from pipeline_error_training import infer,organoid_inference_crops,scoring,stress
    for module,name in [(infer,'sample_gpu'),(infer,'sample_compact'),(infer,'load_model'),
                        (organoid_inference_crops,'patches'),(scoring,'load_model')]:
        monkeypatch.setattr(module,name,getattr(module,name))
    calls = []
    def native(*args):
        calls.append('native')
        return torch.full((1,7,2,16,64,64),128,dtype=torch.uint8),torch.ones((1,7,4))
    def shortcut(*args):
        raise AssertionError('Unperturbed triplane shortcut bypassed the stress recipe')
    monkeypatch.setattr(infer,'sample_gpu',native)
    monkeypatch.setattr(infer,'sample_compact',shortcut)
    recipe = dict(sampling_scale=1.02,center_shift_um_zyx=[.4,.2,.2],contrast=.9,brightness=.9,
                  gamma=1.1,seed=20260915,read_noise_std=.01,blur_kernel=[.25,.5,.25])
    stress.install(recipe)
    images = SimpleNamespace(scale=np.array([1.625,.40625,.40625]),path=Path('source_image'))
    patch,valid = infer.sample_compact(images,np.array([[19,10,8,32,32]]),[[]],[[]],[0])
    assert calls==['native'] and patch.shape==(1,3,3,12,12)
    assert np.any(patch[:,:2]) and not np.any(patch[:,2])
    assert np.all(valid[:,:2]==1) and np.all(valid[:,2]==0)
