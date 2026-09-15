"""Exact source proof for released Organoid crops with cached image quantiles."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .bank import EventBank
    from .common import RESULTS, WORK, inputs, read_json, save_arrays, sha, verified_graph, verified_evidence, write_json
    from .crops import Images, prediction_tracklet
    from .organoid_adapter import patches as reference_patches
    from .organoid_inference_crops import patches
    from .resources import Lease, Monitor
    from .scoring import load_model
    torch.set_num_threads(2)
    row = next(r for r in inputs() if r['dataset']=='44b6_5f15d135')
    package = WORK/'training/D10_frozen/44b6/20260915'
    model, spec = load_model(package)
    graph, native = verified_graph(row), verified_evidence(row)
    nodes = graph['nodes']
    bank = EventBank(nodes,graph['edges'],native,row['physical_scale'])
    root = WORK/'organoid_crop_parity'
    root.mkdir(parents=True,exist_ok=True)
    reference_path = WORK/'source_screen/D10_frozen/44b6/20260915/predictions'/row['dataset']/'current_image_cache'/spec['weights_sha256']/f'{row["dataset"]}.npz'
    reference_receipt = read_json(reference_path.with_suffix('.json'))
    if sha(reference_path)!=reference_receipt['sha256'] or reference_receipt['inputs']['model_sha256']!=spec['weights_sha256']:
        raise ValueError('Original released-model source embedding changed')
    with np.load(reference_path,allow_pickle=False) as f:
        expected_z, expected_v = f['embedding'], f['valid']
    pixel_cases = []
    with Monitor(root/'resources.json'):
        checked_images = Images(row['image_path'])
        for t in range(row['image_shape'][0]):
            raw = checked_images.raw(t)
            np.testing.assert_array_equal(checked_images.quantiles[t],
                [np.quantile(raw,.01),np.quantile(raw,.99)])
        for t in [0,50,99]:
            ii = np.flatnonzero(nodes[:,1]==t)[:32]
            expected = reference_patches(checked_images,nodes,ii)
            actual = patches(checked_images,nodes,ii)
            np.testing.assert_array_equal(actual,expected)
            pixel_cases.append(dict(frame=t,nodes=len(ii),exact_float32_pixels=True))
        del checked_images
        images = Images(row['image_path'])
        result, valid_result = np.full_like(expected_z,np.nan),np.full_like(expected_v,np.nan)
        with Lease(required_gib=8.),torch.inference_mode():
            model.cuda()
            started = time.monotonic()
            for t in range(row['image_shape'][0]):
                ids = np.flatnonzero(nodes[:,1]==t)
                for start in range(0,len(ids),32):
                    ii = ids[start:start+32]
                    p = patches(images,nodes,ii)
                    masks = []
                    for i in ii:
                        _,tracked = prediction_tracklet(nodes,bank.pred,bank.succ,int(i))
                        masks.append([[1.,float(tracked[dt]),1.,1.] if 0<=t+dt<images.shape[0] else [0.,0.,0.,0.]
                                      for dt in range(-2,5)])
                    v = np.asarray(masks,np.float32)
                    result[ii] = model.encoder(torch.from_numpy(p).cuda(),torch.from_numpy(v).cuda()).cpu().numpy()
                    valid_result[ii] = v
                if t%20==0:
                    print(f'Exact released Organoid crop parity: frame {t}/100',flush=True)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            model.cpu()
            torch.cuda.empty_cache()
        np.testing.assert_array_equal(result,expected_z)
        np.testing.assert_array_equal(valid_result,expected_v)
        if {str(k):v for k,v in images.read_hashes.items()}!=reference_receipt['image_frame_hashes']:
            raise ValueError('Cached-quantile image provenance differs from released inference')
        save_arrays(root/'embedding.npz',embedding=result,valid=valid_result)
    receipt = dict(status='measured',source='44b6',dataset=row['dataset'],frames=row['image_shape'][0],
        node_count=len(nodes),seconds=elapsed,nodes_per_second=len(nodes)/elapsed,
        exact_full_clip_embeddings=True,exact_full_clip_masks=True,pixel_cases=pixel_cases,
        exact_separate_vs_joint_quantiles_for_all_frames=True,cached_embeddings_used=False,
        source_model_sha256=spec['weights_sha256'],reference_embedding_sha256=sha(reference_path),
        sampler_sha256=sha(Path(__file__).with_name('organoid_inference_crops.py')),
        implementation_sha256=sha(Path(__file__)),new_target_metrics_read=False,
        historical_code_or_model_changed=False,training_crops_unchanged=True,
        preserved_precision='Released float32 normalization order, nearest resizing, missing-frame filling; identical batches of 32',
        not_a_full_pipeline_Kaggle_runtime=True)
    write_json(RESULTS/'organoid_crop_parity.json',receipt,immutable=True)
    print(receipt,flush=True)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
    import sys
    if '--await-adoption' in sys.argv:
        from .common import WORK, read_json, sha
        marker = WORK/'maintenance/organoid_sampler_adoption.json'
        print('Released crop parity passed; GPU released; waiting at the inference implementation boundary.',flush=True)
        while not marker.exists():
            time.sleep(5)
        accepted = read_json(marker)
        if accepted['sampler_sha256']!=sha(Path(__file__).with_name('organoid_inference_crops.py')) or \
                accepted['inference_sha256']!=sha(Path(__file__).with_name('infer.py')):
            raise ValueError('Organoid sampler adoption does not identify the verified implementation')
