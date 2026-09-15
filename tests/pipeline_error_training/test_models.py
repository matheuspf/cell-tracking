import numpy as np
import torch

from pipeline_error_training.crops import sample_at_coordinate, sample_native
from pipeline_error_training.models import CompactEncoder, DecisionModel, compatible_set_loss


def test_daughter_order_and_padding_invariance():
    torch.manual_seed(9)
    model = DecisionModel('compact').eval()
    z = torch.randn(4, 128)
    features = torch.randn(1, 40)
    a = model.event_scores(z, torch.tensor([[0, 1, 2]]), features)
    b = model.event_scores(z, torch.tensor([[0, 2, 1]]), features)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    encoder = CompactEncoder().eval()
    patch = torch.randn(2, 3, 3, 12, 12)
    valid = torch.ones(2, 3, 4)
    valid[:, 2, 0] = 0
    expected = encoder(patch, valid)
    patch[:, 2] = 10000
    torch.testing.assert_close(encoder(patch, valid), expected, rtol=0, atol=0)


def test_masked_compatible_set_and_group_normalization():
    values = torch.tensor([0., 1., 10., 0., 1.], requires_grad=True)
    positive = torch.tensor([True, False, False, True, False])
    known = torch.tensor([True, True, False, True, True])
    group = torch.tensor([1, 1, 1, 2, 2])
    actual = compatible_set_loss(values, positive, known, group)
    torch.testing.assert_close(actual, torch.nn.functional.softplus(torch.tensor(1.)))
    actual.backward()
    assert values.grad[2] == 0


def test_missing_future_crop_is_masked_without_frame_copy():
    class Images:
        shape = (1, 8, 8, 8)
        quantiles = {0: (0, 1)}
        def raw(self, t):
            return np.ones(self.shape[1:], np.float32) if t == 0 else None
    nodes = np.array([[101, 0, 4, 4, 4]])
    patch, valid = sample_native(Images(), nodes, [[]], [[]], 0, shape=(4, 4, 4))
    assert valid[:, 0].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert not patch[3:].any()


def test_replacement_feature_at_actual_coordinate():
    frame = np.arange(64).reshape(4, 4, 4)
    assert sample_at_coordinate(frame, [1, 1, 1]) == 21
    assert sample_at_coordinate(frame, [1, 1, 2]) == 22
