"""Explicit-package image-to-CSV inference; no annotation or name routing."""
from pathlib import Path
import sys
import tempfile
import time
import hashlib
from .common import REPO,WORK,Blocked,read,write,sha,now


def upstream_graph(model,image,folder,dataset,resources):
    import numpy as np
    import torch
    from .data import normalize
    from .graphs import peaks,candidate_union,continuation,export_csv,read_csv,graph_hash
    model.eval();previous=None;last_points=last_feature=last_ids=None
    nodes=[];edges=[];confidence=[];evidence=[];frame_hashes={};times=[]
    started=time.monotonic()
    for t in range(image.shape[0]):
        current=None
        if t<image.shape[0]-1:
            frames=[]
            for j in (t,t+1):
                raw=np.asarray(image[j]);frame_hashes[str(j)]=hashlib.sha256(raw.tobytes(order='C')).hexdigest()
                frames.append(normalize(raw)[0][None])
            x=np.stack(frames)[None]
            with resources.lease(2*2**30) as lease:
                model.cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,feature=model(torch.from_numpy(x).cuda())
                current=(logits[0].float().cpu(),feature[0].float().cpu())
                del logits,feature
                model.cpu();torch.cuda.empty_cache()
            times.append(lease['seconds'])
        if previous is None:logits,feature=current[0][0],current[1][0]
        elif current is None:logits,feature=previous[0][1],previous[1][1]
        else:logits,feature=(previous[0][1]+current[0][0])/2,(previous[1][1]+current[1][0])/2
        points,scores=peaks(logits.numpy());ids=np.arange(len(nodes),len(nodes)+len(points),dtype=np.int64)
        nodes.extend([[int(i),t,*map(int,p)] for i,p in zip(ids,points)])
        confidence.extend((1/(1+np.exp(-scores))).tolist())
        if last_points is not None and len(last_points) and len(points):
            pairs=candidate_union(last_points,points)
            with resources.lease(2*2**30) as lease:
                model.cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    values=model.association(last_feature.cuda(),feature.cuda(),
                        torch.tensor(last_points,dtype=torch.float32,device='cuda'),
                        torch.tensor(points,dtype=torch.float32,device='cuda'),t-1).float().cpu().numpy()
                model.cpu();torch.cuda.empty_cache()
            times.append(lease['seconds'])
            chosen=continuation(last_points,points,pairs,values[pairs[:,0],pairs[:,1]])
            edges.extend([[int(last_ids[a]),int(ids[b])] for a,b in chosen])
            extra=candidate_union(last_points,points,neighbors=6,reverse_neighbors=2,gate=16.)
            union=np.asarray(sorted(set(map(tuple,pairs))|set(map(tuple,extra))),np.int64).reshape(-1,2)
            evidence.extend([[int(last_ids[a]),int(ids[b]),float(values[a,b])] for a,b in union])
        last_points,last_feature,last_ids=points,feature,ids;previous=current
        write(folder/'progress.json',dict(status='running',frames=t+1,nodes=len(nodes),edges=len(edges),updated_utc=now()))
    n=np.asarray(nodes,np.int64).reshape(-1,5);e=np.asarray(edges,np.int64).reshape(-1,2)
    actual=export_csv(folder/'submission.csv',dataset,n,e,shape=image.shape)
    restored=read_csv(folder/'submission.csv',dataset)
    if not all(np.array_equal(a,b) for a,b in zip(actual,restored)):raise Blocked('CSV roundtrip mismatch')
    np.savez_compressed(folder/'graph.npz',nodes=actual[0],edges=actual[1],confidence=np.asarray(confidence,np.float32),
                        edge_scores=np.asarray(evidence,np.float64).reshape(-1,3))
    return dict(status='predicted_unscored',frames=image.shape[0],nodes=len(n),edges=len(e),frame_hashes=frame_hashes,
                graph_hash=graph_hash(*actual),csv_sha256=sha(folder/'submission.csv'),graph_sha256=sha(folder/'graph.npz'),
                wall_seconds=time.monotonic()-started,gpu_lease_seconds=sum(times),max_lease_seconds=max(times,default=0.),
                csv_roundtrip_exact=True)


def run(package,images,output,*,cold=False,baseline=None):
    """Only these explicit artifacts are readable; unfamiliar names are valid."""
    package=Path(package).resolve();images=Path(images).resolve();output=Path(output).resolve()
    from .provenance import unseal,validate
    spec=unseal(package)
    for name,expected in spec.get('runtime_code_sha256',{}).items():
        if sha(Path(__file__).with_name(name))!=expected:raise Blocked('Explicit package runtime code changed: '+name)
    baseline=Path(baseline).resolve() if baseline is not None else None
    if baseline is not None and (cold or spec['arm']=='C00'):raise Blocked('Cold/C00 inference must start from raw images')
    from .stage_provenance import validate_stages
    validate_stages(spec['ancestry'],spec['root_artifact'],spec['source'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str(output/'tmp')
    from .resources import Resources,ROOT
    resources=Resources('cold' if cold else 'inference',spec['source'],spec['seed'])
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[package,images,*([baseline] if baseline else [])],outputs=[output,ROOT],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
                  *[Path(p) for p in sys.path if 'site-packages' in p]])
    import torch
    import zarr
    from .upstream import Upstream
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    image=zarr.open_array(str(images/'0'),mode='r')
    if tuple(image.shape)!=(100,64,256,256):raise Blocked('Unexpected real image shape')
    name=images.name.removesuffix('.zarr')
    if (output/'receipt.json').exists():
        import numpy as np
        old=read(output/'receipt.json')
        if old.get('package_identity')!=spec['identity'] or old.get('status')!='predicted_unscored' or old.get('dataset')!=name:
            raise Blocked('Prediction output exists with a different package/image identity')
        if bool(old.get('cold'))!=cold or bool(old.get('shared_C00'))!=(baseline is not None):
            raise Blocked('Prediction resume changes the raw/cache input contract')
        if sha(output/'graph.npz')!=old['graph_sha256'] or sha(output/'submission.csv')!=old['csv_sha256']:
            raise Blocked('Prediction output bytes changed')
        for t in range(100):
            if hashlib.sha256(np.asarray(image[t]).tobytes(order='C')).hexdigest()!=old['frame_hashes'][str(t)]:
                raise Blocked('Prediction resume uses different raw image frames')
        resources.close();return old
    started=time.monotonic()
    if baseline is None:
        model=Upstream(package/'architecture')
        model.load_state_dict(torch.load(package/'upstream.pt',map_location='cpu',weights_only=True)['model'])
        result=upstream_graph(model,image,output,name,resources)
        del model
    else:
        import numpy as np
        import shutil
        result=read(baseline/'receipt.json')
        if result['arm']!='C00' or result['upstream_sha256']!=spec['files']['upstream.pt'] or result['dataset']!=name:
            raise Blocked('Shared C00 observation ancestry does not match explicit package/image')
        if sha(baseline/'graph.npz')!=result['graph_sha256'] or sha(baseline/'submission.csv')!=result['csv_sha256']:
            raise Blocked('Shared C00 prediction bytes changed')
        for t in range(100):
            if hashlib.sha256(np.asarray(image[t]).tobytes(order='C')).hexdigest()!=result['frame_hashes'][str(t)]:
                raise Blocked('Shared C00 graph belongs to different raw images')
        shutil.copy2(baseline/'graph.npz',output/'graph.npz')
        shutil.copy2(baseline/'submission.csv',output/'submission.csv')
        result['shared_C00_receipt_sha256']=sha(baseline/'receipt.json')
    if spec['arm']!='C00':
        from .policy import apply_package
        result=apply_package(spec,package,images,output,resources,result)
    result.update(package_identity=spec['identity'],source=spec['source'],seed=spec['seed'],arm=spec['arm'],
                  dataset=name,guard=guard,cold=cold,finished_utc=now(),
                  upstream_sha256=spec['files']['upstream.pt'],shared_C00=baseline is not None,
                  gpu_lease_seconds=resources.seconds,wall_seconds=time.monotonic()-started,
                  inference_implementation_sha256=sha(Path(__file__)))
    write(output/'receipt.json',result,immutable=True);resources.close();return result
