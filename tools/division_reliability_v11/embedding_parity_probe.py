"""Diagnose trained-encoder batch roundoff without changing fit or policy output."""
from pathlib import Path
import os,sys
from .common import REPO,WORK,DATA,read,write,sha,now


def run(label='default'):
    source='44b6';seed=20260918
    clips=['44b6_0113de3b','44b6_0b24845f','44b6_0c582fdc']
    checkpoint=WORK/'fits'/source/str(seed)/'compact/mining/frozen_checkpoint.pt'
    output=WORK/'checks/trained_embedding_parity'/label;output.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha(checkpoint)
    from .resources import Resources,ROOT
    resources=Resources('event_mining',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[checkpoint,*[WORK/'predictions/C00'/source/str(seed)/n for n in clips],
        *[DATA/'train'/f'{n}.zarr' for n in clips]],outputs=[output,ROOT],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    import hashlib
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    from .actions import Bank
    from pipeline_error_training.crops import Images
    from pipeline_error_training.compact_inference_crops import sample
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
    defaults=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    rows=[]
    for clip in clips:
        with np.load(WORK/'predictions/C00'/source/str(seed)/clip/'graph.npz') as f:
            nodes,edges,scores=(f[k] for k in ('nodes','edges','edge_scores'))
        bank=Bank(nodes,edges,scores,100);images=Images(DATA/'train'/f'{clip}.zarr',max_frames=9)
        patch,valid=sample(images,bank.nodes,bank.pred,bank.succ,range(min(4096,len(nodes))))
        values=patch.astype(np.float32)/255
        with resources.lease(2*2**30) as lease:
            lease['operation']='trained_embedding_batch_parity_diagnosis'
            model.cuda().eval();x=torch.from_numpy(values).cuda();mask=torch.from_numpy(valid).cuda()
            with torch.inference_mode():
                batch=model.encoder(x,mask)
                indices=[0,1,7,31,127,511,1023,2047,4095]
                indices=[i for i in indices if i<len(x)]
                scalar=torch.cat([model.encoder(x[i:i+1],mask[i:i+1]) for i in indices])
                selected=batch[indices];diff=(selected-scalar).abs()
                torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
                fp32_batch=model.encoder(x,mask)
                fp32_scalar=torch.cat([model.encoder(x[i:i+1],mask[i:i+1]) for i in indices])
                model.double();exact=model.encoder(x[indices].double(),mask[indices].double())
                row=dict(clip=clip,batch_size=len(x),reference_indices=indices,
                    maximum_embedding_magnitude=float(batch.abs().max()),legacy_first_maximum_difference=float(diff[0].max()),
                    legacy_all_sampled_maximum_difference=float(diff.max()),
                    legacy_relative_to_global_scale=float(diff.max()/selected.abs().max()),
                    legacy_allclose_atol_1e5_rtol_1e5=bool(torch.allclose(selected,scalar,atol=1e-5,rtol=1e-5)),
                    legacy_maximum_scaled_error=float((diff/(1e-5+1e-5*scalar.abs())).max()),
                    strict_fp32_maximum_batch_scalar_difference=float((fp32_batch[indices]-fp32_scalar).abs().max()),
                    legacy_batch_equals_strict_fp32=bool(torch.equal(batch,fp32_batch)),
                    legacy_batch_maximum_difference_vs_strict_fp32=float((batch-fp32_batch).abs().max()),
                    legacy_batch_maximum_error_vs_float64=float((selected.double()-exact).abs().max()),
                    legacy_scalar_maximum_error_vs_float64=float((scalar.double()-exact).abs().max()),
                    strict_batch_maximum_error_vs_float64=float((fp32_batch[indices].double()-exact).abs().max()),
                    legacy_batch_sha256=hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest())
                rows.append(row);print(row,flush=True)
            model.float().cpu();torch.backends.cudnn.allow_tf32=defaults['cudnn_tf32'];torch.backends.cuda.matmul.allow_tf32=defaults['matmul_tf32']
            del x,mask,batch,scalar,selected,diff,fp32_batch,fp32_scalar,exact
            torch.cuda.empty_cache()
        del bank,images,patch,valid,values
    assert sha(checkpoint)==checkpoint_hash
    result=dict(status='measured',source=source,seed=seed,step=2000,checkpoint_sha256=checkpoint_hash,
        guard=guard,default_precision=defaults,NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        rows=rows,checkpoint_unchanged=True,model_selection=False,finished_utc=now())
    write(output/'receipt.json',result);resources.close();return result


if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'default')
