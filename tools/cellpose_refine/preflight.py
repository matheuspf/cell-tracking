"""Geometry, real frozen-feature parity, sparse gradients and head resource checks."""
from __future__ import annotations

import json
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F

from .common import (REPO, RESULTS, SCREEN, SPACING, WORK, assert_source, bounded_integer,
                     config, configure, json_hash, locked_gpu, native_to_isotropic, now,
                     sha, validate_ancestry, write_json)
from .features import FrozenFeatures, normalize, patches, sample_tokens
from .model import Refiner, sparse_offset_loss


def main():
    configure()
    cfg = config()
    checks = {}
    for repo, key in (("cellpose", "cellpose_commit"), ("dinov3", "dinov3_commit")):
        actual = subprocess.check_output(["git", "-C", str(REPO.parent / repo), "rev-parse", "HEAD"], text=True).strip()
        assert actual == cfg["features"][key]
        checks[key] = actual
    shape = (64, 256, 256)
    rng = np.random.default_rng(1234)
    q = rng.uniform([0, 0, 0], np.asarray(shape) - 1, size=(1000, 3))
    emitted, changed = bounded_integer(q, q, shape)
    assert np.array_equal(emitted, np.rint(q)) and not changed.any()
    offset = rng.normal(size=(len(q), 3)); offset *= 3 / np.linalg.norm(offset, axis=1, keepdims=True)
    emitted, changed = bounded_integer(q, q + offset / SPACING, shape)
    assert (np.linalg.norm((emitted - np.rint(q)) * SPACING, axis=1) <= 3 + 1e-9).all()
    checks["integer_identity_and_3um_net_movement_bound"] = True

    from cellpose import transforms
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    ramp = zz * 3 + yy * .5 + xx * .25
    resized = transforms.resize_image_3d(ramp[..., None], (256, 256, 256), no_channels=False)[..., 0]
    from scipy.ndimage import map_coordinates
    interior = rng.uniform([2, 2, 2], [61, 253, 253], size=(25, 3))
    values = map_coordinates(resized, native_to_isotropic(interior).T, order=1, prefilter=False)
    np.testing.assert_allclose(values, interior @ np.array([3., .5, .25]), atol=5e-5)
    local = patches(ramp / 100, interior, (3, 3, 3)).astype(np.float32)
    np.testing.assert_allclose(local[:, 1, 1, 1], interior @ np.array([3., .5, .25]) / 100, atol=.0015)
    checks["resize_half_pixel_and_native_patch_ramps"] = True

    token = torch.arange(48 * 48 * 3).reshape(1, 48 * 48, 3).float()
    uv = np.array([[1.25, 13.75], [37.5, 44.5], [125.1, 153.6]])
    values = sample_tokens(token, [0, 0, 0], uv, (64, 64), (48, 48))
    coords = (uv + 64 - 3.5) / 8
    grid = torch.tensor(coords[:, ::-1].copy(), dtype=torch.float32) / 47 * 2 - 1
    reference = F.grid_sample(token.reshape(1, 48, 48, 3).permute(0, 3, 1, 2), grid.reshape(1, 1, 3, 2), align_corners=True).reshape(3, 3).T
    torch.testing.assert_close(values, reference, atol=.001, rtol=1e-6)
    checks["token_bilinear_sampling_matches_torch_grid_sample"] = True

    prediction = torch.zeros(3, 3, requires_grad=True)
    loss = sparse_offset_loss(prediction, torch.ones(3, 3), torch.tensor([1., 0., 1.]))
    loss.backward()
    assert torch.count_nonzero(prediction.grad[1]) == 0 and torch.count_nonzero(prediction.grad[[0, 2]]) == 6
    checks["unknown_query_gradient_exactly_zero"] = True
    for field in ("labels", "unlabeled_images", "calibration", "teacher_construction", "normalization_fit"):
        ancestry = {"adaptation_exposure": {k: [] for k in ("labels", "unlabeled_images", "calibration", "teacher_construction", "normalization_fit")}}
        ancestry["adaptation_exposure"][field] = ["6bba"]
        try:
            validate_ancestry("44b6", ancestry)
        except ValueError:
            pass
        else:
            raise AssertionError(field)
    try:
        assert_source("44b6", [{"embryo": "6bba", "dataset": "6bba_probe"}])
    except ValueError:
        checks["all_target_exposure_fields_and_wrong_source_rows_rejected"] = True

    frame = next(r for r in json.loads((SCREEN / "panel.json").read_text())["frames"] if r["role"] == "pilot" and r["embryo"] == "44b6")
    with np.load(SCREEN / "predictions/cellpose_cpdino_vitb" / (frame["key"] + ".npz")) as p:
        queries = p["centers_zyx"][:4].copy()
    volume = np.load(frame["image_path"])
    with locked_gpu() as queue:
        torch.cuda.reset_peak_memory_stats()
        extractor = FrozenFeatures()
        before = {k: v.detach().clone() for k, v in extractor.net.state_dict().items()}
        f, p, g = extractor.extract(volume, queries)
        first_time = extractor.last_seconds.copy()
        ff, pp, gg = extractor.extract(volume, queries)
        assert np.array_equal(f, ff) and np.array_equal(p, pp) and np.array_equal(g, gg)
        assert all(torch.equal(before[k], v) for k, v in extractor.net.state_dict().items())
        assert all(not v.requires_grad for v in extractor.net.parameters())
        checks["real_features_and_patches_repeat_bitwise"] = True
        checks["encoder_weights_unchanged_and_frozen"] = True
        # The tap must reconstruct exactly the upstream readout, not the random style output.
        normalized = normalize(volume)
        plane = np.pad(normalized[20], ((64, 64), (64, 64)))[None, None]
        tokens, logits = extractor.batch(plane)
        readout = extractor.net.out(tokens).reshape(1, 48, 48, 192).permute(0, 3, 1, 2)
        reconstructed = F.conv_transpose2d(readout, extractor.net.W2, stride=8)
        assert torch.equal(logits, reconstructed)
        checks["feature_tap_reconstructs_original_cellpose_logits_exactly"] = True
        del before, extractor, tokens, logits, readout, reconstructed
        torch.cuda.empty_cache()
        torch.manual_seed(101)
        head = Refiner(cfg).cuda()
        feature = torch.from_numpy(f).cuda().repeat(32, 1, 1)
        patch = torch.from_numpy(p).cuda().repeat(32, 1, 1, 1)
        geometry = torch.from_numpy(g).cuda().repeat(32, 1)
        assert torch.count_nonzero(head(feature, patch, geometry)) == 0
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
        target = torch.tensor([.25, -.5, .5], device="cuda").expand(128, 3)
        weight = torch.ones(128, device="cuda")
        losses = []
        started = time.perf_counter()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            out = head(feature, patch, geometry)
            loss = sparse_offset_loss(out, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        assert losses[-1] < losses[0] * .5
        assert torch.linalg.vector_norm(out, dim=1).max() <= 3 + 1e-6
        peak = torch.cuda.max_memory_reserved()
        assert peak < cfg["training"]["max_gpu_reserved_gib"] * 1024**3
        checks["head_zero_initialization_bounded_output_and_real_backward"] = True
        resources = {"batch_size": 128, "updates": 30, "seconds": time.perf_counter() - started,
                     "max_reserved_bytes_including_feature_preflight": peak,
                     "head_parameters": sum(x.numel() for x in head.parameters()),
                     "toy_loss_first_last": [losses[0], losses[-1]], "feature_probe": first_time,
                     "gpu_queue_seconds": queue}
    receipt = {"created_utc": now(), "status": "passed", "config_sha256": json_hash(cfg),
               "checks": checks, "resources": resources, "probe_frame": frame["key"],
               "labels_read": False, "training": "30 disposable constant-offset optimization steps only; no retained fit",
               "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in (REPO / "tools/cellpose_refine").glob("*.py")}}
    write_json(RESULTS / "preflight.json", receipt)
    write_json(WORK / "preflight.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
