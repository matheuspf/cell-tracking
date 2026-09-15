"""Uncached full-source parity and throughput before adopting frame crop reuse."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .bank import EventBank
    from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_graph, verified_evidence, write_json
    from .crops import Images
    from .fast_crops import sample_gpu as reference_sampler
    from .frame_crops import sample_gpu
    from .resources import Lease, Monitor
    from .scoring import load_model
    torch.set_num_threads(2)
    row = next(r for r in inputs() if r['dataset']=='44b6_5f15d135')
    package = WORK/'training_pilot/D20_temporal/44b6/20260915'
    model, spec = load_model(package)
    graph, native = verified_graph(row), verified_evidence(row)
    bank = EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    root = WORK/'frame_crop_parity'
    root.mkdir(parents=True,exist_ok=True)
    reference_path = WORK/'inference_profile/temporal'/f'{row["dataset"]}.npz'
    reference_receipt = read_json(reference_path.with_suffix('.json'))
    if sha(reference_path)!=reference_receipt['sha256'] or reference_receipt['inputs']['model_sha256']!=spec['weights_sha256']:
        raise ValueError('Original measured source embedding changed')
    with np.load(reference_path, allow_pickle=False) as f:
        expected_z, expected_v = f['embedding'], f['valid']
    images, original_images = Images(row['image_path']), Images(row['image_path'])
    nodes = graph['nodes']
    result, valid_result = np.full_like(expected_z,np.nan), np.full_like(expected_v,np.nan)
    pixel_cases, gather_seconds, encode_seconds = [], 0., 0.
    with Monitor(root/'resources.json'), Lease(required_gib=8.), torch.inference_mode():
        model.cuda()
        started = time.monotonic()
        for t in range(row['image_shape'][0]):
            ids = np.flatnonzero(nodes[:,1]==t)
            for start in range(0,len(ids),32):
                ii = ids[start:start+32]
                begin = time.monotonic()
                p, vv = sample_gpu(images,nodes,bank.pred,bank.succ,ii)
                v = vv.cpu().numpy()
                gather_seconds += time.monotonic()-begin
                if t in [0,50,99] and start==0:
                    reference_p, reference_v = reference_sampler(original_images,nodes,bank.pred,bank.succ,ii)
                    np.testing.assert_array_equal(p.cpu().numpy(),reference_p.cpu().numpy())
                    np.testing.assert_array_equal(v,reference_v.cpu().numpy())
                    pixel_cases.append(dict(frame=t,nodes=len(ii),exact_pixels=True,exact_masks=True))
                begin = time.monotonic()
                x = p.float()/255
                result[ii] = model.encoder(x,torch.from_numpy(v).cuda()).cpu().numpy()
                valid_result[ii] = v
                encode_seconds += time.monotonic()-begin
            if t%20==0:
                print(f'Exact frame-cache source parity: frame {t}/100',flush=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        model.cpu()
        torch.cuda.empty_cache()
    np.testing.assert_array_equal(result,expected_z)
    np.testing.assert_array_equal(valid_result,expected_v)
    if {str(k):v for k,v in images.read_hashes.items()}!=reference_receipt['image_frame_hashes']:
        raise ValueError('Frame-cache image provenance differs from original inference')
    save_arrays(root/'embedding.npz',embedding=result,valid=valid_result)
    original = read_json(RESULTS/'temporal_inference_profile.json')
    receipt = dict(status='measured',dataset=row['dataset'],source='44b6',frames=row['image_shape'][0],
        node_count=len(result),seconds=elapsed,nodes_per_second=len(result)/elapsed,
        original_seconds=original['seconds'],speedup=original['seconds']/elapsed,
        crop_wall_seconds=gather_seconds,encoder_wall_seconds=encode_seconds,
        full_clip=True,cached_embeddings_used=False,annotation_labels_read=False,
        source_pilot_checkpoint_sha256=spec['weights_sha256'],reference_embedding_sha256=sha(reference_path),
        checkpoint_for_timing_only_not_candidate_selection=True,
        exact_full_clip_embeddings=True,exact_full_clip_masks=True,pixel_cases=pixel_cases,
        sampler_sha256=sha(Path(__file__).with_name('frame_crops.py')),
        implementation_sha256=sha(Path(__file__)),new_target_metrics_read=False,
        preserved_precision='FP32, TF32 disabled; original NumPy quantization; identical batches of 32',
        not_a_full_pipeline_Kaggle_runtime=True)
    write_json(RESULTS/'frame_crop_parity.json',receipt,immutable=True)
    print(receipt,flush=True)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
