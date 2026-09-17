"""Annotation-free startup for shared-scene frozen inference matrices."""
import atexit
import json
import os
from pathlib import Path
import sys


def main():
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      ZARR_ASYNC__CONCURRENCY='2',ZARR_THREADING__MAX_WORKERS='2')
    job=json.loads(Path(sys.argv[1]).read_text());output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    from pipeline_error_training.guard import install
    guard=install(fresh_root=output,allowed_models=[job['graph_path'],job['native_path'],*[p['checkpoint'] for p in job['packages'].values()]],
                  images=[job['row']['image_path']])
    def save_guard():
        for key in job['packages']:
            root=output/key;root.mkdir(parents=True,exist_ok=True)
            (root/'guard.json').write_text(json.dumps(guard,indent=2)+'\n')
    atexit.register(save_guard)
    from pipeline_error_training.resources import cpu_budget
    cpu_budget(16)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from .common import load_graph,sha,write_json,code_hashes
    from .infer import load_checkpoint
    from .matrix_infer import predict_matrix
    from .resources import Monitor
    if sha(job['graph_path'])!=job['graph_sha256'] or sha(job['native_path'])!=job['native_sha256']:
        raise ValueError('Matrix baseline/evidence hash changed')
    packages={}
    for key,p in job['packages'].items():
        if sha(p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Matrix checkpoint hash changed')
        model,recipe=load_checkpoint(p['checkpoint'])
        if recipe['source']!=job['source']:raise ValueError('Explicit matrix source mismatch')
        packages[key]=(model,recipe,p['calibration'])
    write_json(output/'runtime_code.json',code_hashes())
    with Monitor(output/'resources.json'):
        predict_matrix(job['row'],load_graph(job['graph_path']),load_graph(job['native_path']),packages,output)


if __name__=='__main__':main()
