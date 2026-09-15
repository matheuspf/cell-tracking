"""Guard the physical 7 µm gate and the adapters' voxel-center conventions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .evaluate import ROOT, SPACING, matches


def main():
    gt = np.array([[101, 9, 20, 100, 100]], dtype=float)
    center = gt[:, 2:].copy()
    checks = {}
    for axis in range(3):
        for distance in (6.999, 7.001):
            point = center.copy()
            point[0, axis] += distance / SPACING[axis]
            found = len(matches(point, gt, 9, 7.0))
            assert found == int(distance < 7), (axis, distance, found)
            checks[f'axis_{axis}_distance_{distance}_um'] = found

    two_gt = np.array([[101, 9, 20, 100, 100], [202, 9, 20, 102, 100]])
    # One candidate cannot count twice even if two GT nodes are close to it.
    assert len(matches(center, two_gt, 9, 7.0)) == 1
    checks['one_candidate_two_nearby_truth_nodes'] = 1

    z, y, x = np.indices((8, 16, 16))
    ramp = 100 * z + 10 * y + x
    pooled = ramp.reshape(8, 4, 4, 4, 4).mean(axis=(2, 4))
    pz, py, px = np.indices(pooled.shape)
    expected = 100 * pz + 10 * (4 * py + 1.5) + (4 * px + 1.5)
    assert np.array_equal(pooled, expected)
    checks['spotiflow_block_mean_coordinate_origin_exact'] = True

    # scipy's grid_mode=True samples pixel centers using this exact inverse.
    a = np.arange(64, dtype=float)
    b = zoom(a, 4 / (17 / 7), order=1, grid_mode=True, mode='nearest')
    mapped = (np.arange(len(b)) + 0.5) / (len(b) / len(a)) - 0.5
    assert np.allclose(b, np.clip(mapped, 0, len(a) - 1), atol=1e-12)
    checks['xenopus_pixel_center_inverse_max_error_px'] = float(
        np.max(np.abs(b - np.clip(mapped, 0, len(a) - 1)))
    )

    result = {'checks': checks, 'spacing_um': SPACING.tolist(),
              'scope': 'Metric units, one-to-one matching, and resampling geometry; no model quality claims.'}
    path = ROOT / 'evaluation/geometry-verification.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
