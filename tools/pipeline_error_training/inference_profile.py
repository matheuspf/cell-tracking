"""Source-image-only throughput of the locked temporal architecture, without scores."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import torch
    from .bank import EventBank
    from .common import RESULTS, WORK, inputs, read_json, sha, verified_graph, verified_evidence, write_json
    from .infer import embeddings
    from .resources import Lease, Monitor
    from .scoring import load_model
    torch.set_num_threads(2)
    row = next(r for r in inputs() if r['dataset']=='44b6_5f15d135')
    package = WORK/'training_pilot/D20_temporal/44b6/20260915'
    model, spec = load_model(package)
    graph, native = verified_graph(row), verified_evidence(row)
    bank = EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    root = WORK/'inference_profile/temporal'
    path = root/f'{row["dataset"]}.npz'
    if path.exists():
        raise ValueError('Cold source throughput must not be replaced by a cached run')
    begin = time.monotonic()
    with Monitor(root/'resources.json'), Lease(required_gib=8.):
        model.cuda()
        z, _ = embeddings(model,row,graph,bank,path,spec['weights_sha256'])
        torch.cuda.synchronize()
        model.cpu()
        torch.cuda.empty_cache()
    elapsed = time.monotonic()-begin
    receipt = dict(status='measured',dataset=row['dataset'],source='44b6',frames=row['image_shape'][0],
        node_count=len(z),seconds=elapsed,nodes_per_second=len(z)/elapsed,
        full_clip=True,cached_embeddings_used=False,annotation_labels_read=False,
        source_pilot_checkpoint_sha256=spec['weights_sha256'],
        checkpoint_for_timing_only_not_candidate_selection=True,
        matched_architecture=['D20_temporal','D20_no_pretrain','A10','O10_swap'],
        new_target_metrics_read=False,implementation_sha256=sha(Path(__file__)),
        not_a_full_pipeline_Kaggle_runtime=True)
    write_json(RESULTS/'temporal_inference_profile.json',receipt,immutable=True)
    print(receipt)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
