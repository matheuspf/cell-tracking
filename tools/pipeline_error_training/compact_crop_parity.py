"""Full source parity for compact triplanes extracted without unused 3D voxels."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .bank import EventBank
    from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_graph, verified_evidence, write_json
    from .compact_inference_crops import sample
    from .crops import Images, compact_view
    from .fast_crops import sample_native
    from .resources import Lease, Monitor
    from .scoring import load_model
    torch.set_num_threads(2)
    row = next(r for r in inputs() if r['dataset']=='44b6_5f15d135')
    package = WORK/'training/D20_compact/44b6/20260915'
    model, spec = load_model(package)
    graph, native = verified_graph(row), verified_evidence(row)
    nodes = graph['nodes']
    bank = EventBank(nodes,graph['edges'],native,row['physical_scale'])
    root = WORK/'compact_crop_parity'
    root.mkdir(parents=True,exist_ok=True)
    reference_path = WORK/'source_screen/D20_compact/44b6/20260915/predictions'/row['dataset']/'current_image_cache/embeddings'/spec['weights_sha256']/f'{row["dataset"]}.npz'
    reference_receipt = read_json(reference_path.with_suffix('.json'))
    if sha(reference_path)!=reference_receipt['sha256'] or reference_receipt['inputs']['model_sha256']!=spec['weights_sha256']:
        raise ValueError('Original compact source embedding changed')
    with np.load(reference_path,allow_pickle=False) as f:
        expected_z, expected_v = f['embedding'],f['valid']
    pixel_cases = []
    with Monitor(root/'resources.json'):
        checked_images = Images(row['image_path'])
        for t in [0,50,99]:
            ii = np.flatnonzero(nodes[:,1]==t)[:16]
            expected = [sample_native(checked_images,nodes,bank.pred,bank.succ,int(i)) for i in ii]
            actual,valid = sample(checked_images,nodes,bank.pred,bank.succ,ii)
            np.testing.assert_array_equal(actual,np.stack([compact_view(p) for p,_ in expected]))
            np.testing.assert_array_equal(valid,np.stack([v[1:4] for _,v in expected]))
            pixel_cases.append(dict(frame=t,nodes=len(ii),exact_pixels=True,exact_masks=True))
        del checked_images
        images = Images(row['image_path'])
        result,valid_result = np.full_like(expected_z,np.nan),np.full_like(expected_v,np.nan)
        with Lease(required_gib=8.),torch.inference_mode():
            model.cuda()
            started = time.monotonic()
            for t in range(row['image_shape'][0]):
                ids = np.flatnonzero(nodes[:,1]==t)
                for start in range(0,len(ids),32):
                    ii = ids[start:start+32]
                    p,v = sample(images,nodes,bank.pred,bank.succ,ii)
                    result[ii] = model.encoder(torch.from_numpy(p).cuda().float()/255,
                                              torch.from_numpy(v).cuda()).cpu().numpy()
                    valid_result[ii] = v
                if t%20==0:
                    print(f'Exact compact triplane parity: frame {t}/100',flush=True)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            model.cpu()
            torch.cuda.empty_cache()
        np.testing.assert_array_equal(result,expected_z)
        np.testing.assert_array_equal(valid_result,expected_v)
        expected_hashes = reference_receipt['image_frame_hashes']
        if any(expected_hashes[str(t)]!=h for t,h in images.read_hashes.items()):
            raise ValueError('Compact triplane image provenance differs from original native crops')
        save_arrays(root/'embedding.npz',embedding=result,valid=valid_result)
    receipt = dict(status='measured',source='44b6',dataset=row['dataset'],frames=row['image_shape'][0],
        node_count=len(nodes),seconds=elapsed,nodes_per_second=len(nodes)/elapsed,
        exact_full_clip_embeddings=True,exact_full_clip_masks=True,pixel_cases=pixel_cases,
        cached_embeddings_used=False,source_model_sha256=spec['weights_sha256'],
        reference_embedding_sha256=sha(reference_path),
        sampler_sha256=sha(Path(__file__).with_name('compact_inference_crops.py')),
        shared_quantizer_sha256=sha(Path(__file__).with_name('frame_crops.py')),
        implementation_sha256=sha(Path(__file__)),new_target_metrics_read=False,
        training_crops_unchanged=True,
        preserved_precision='Original uint8 quantization, scipy order-one resizing and float32 masks; identical batches of 32',
        not_a_full_pipeline_Kaggle_runtime=True)
    write_json(RESULTS/'compact_crop_parity.json',receipt,immutable=True)
    print(receipt,flush=True)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
    import sys
    if '--await-adoption' in sys.argv:
        from .common import WORK, read_json, sha
        marker = WORK/'maintenance/compact_sampler_adoption.json'
        print('Compact triplane parity passed; GPU released; waiting at the inference implementation boundary.',flush=True)
        while not marker.exists():
            time.sleep(5)
        accepted = read_json(marker)
        if accepted['sampler_sha256']!=sha(Path(__file__).with_name('compact_inference_crops.py')) or \
                accepted['inference_sha256']!=sha(Path(__file__).with_name('infer.py')):
            raise ValueError('Compact sampler adoption does not identify the verified implementation')
