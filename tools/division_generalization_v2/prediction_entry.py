"""Install annotation/cache/network denial before importing numerical modules."""
import atexit
import json
import os
from pathlib import Path
import sys


def main():
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      ZARR_ASYNC__CONCURRENCY='2',ZARR_THREADING__MAX_WORKERS='2')
    job=json.loads(Path(sys.argv[1]).read_text())
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    from pipeline_error_training.guard import install
    guard=install(fresh_root=output,allowed_models=[job['checkpoint'],job['graph_path'],job['native_path']],
                  images=[job['row']['image_path']])
    atexit.register(lambda:(output/'guard.json').write_text(json.dumps(guard,indent=2)+'\n'))
    from pipeline_error_training.resources import cpu_budget
    cpu_budget(16)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from .common import sha,load_graph,write_json,code_hashes
    from .infer import load_checkpoint,predict
    from .resources import Monitor
    for key,field in [('checkpoint','checkpoint_sha256'),('graph_path','graph_sha256'),('native_path','native_sha256')]:
        if sha(job[key])!=job[field]:raise ValueError('Prediction input hash drift')
    model,recipe=load_checkpoint(job['checkpoint'])
    if recipe['source']!=job['source']:raise ValueError('Explicit model source mismatch')
    write_json(output/'runtime_code.json',code_hashes())
    with Monitor(output/'resources.json'):
        _,trace=predict(job['row'],load_graph(job['graph_path']),load_graph(job['native_path']),model,recipe,
            output,calibration=job['calibration'])
    write_json(output/'trace.json',trace)


if __name__=='__main__':main()
