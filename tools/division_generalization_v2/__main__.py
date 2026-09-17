"""Entry point installs source guard before numerical imports."""
import argparse
import importlib
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['baselines', 'prepare', 'prewarm', 'validate', 'profile', 'train', 'screen', 'source-matrix', 'queue', 'report', 'status'])
    parser.add_argument('--source', choices=['44b6', '6bba'])
    parser.add_argument('--arm', choices=['G30', 'prefix', 'J_uniform', 'J_mined'], default='G30')
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--clip')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args = parser.parse_args()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_MAX_THREADS',
                'NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS','BLOSC_NTHREADS'):
        os.environ[key] = '1'
    os.environ.update(PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      ZARR_ASYNC__CONCURRENCY='2', ZARR_THREADING__MAX_WORKERS='2')
    if args.command in ('prepare','prewarm','train','screen','profile','source-matrix'):
        if args.source is None:
            parser.error('An explicit source is required')
        from pipeline_error_training.guard import install
        install(source=args.source)
    from pipeline_error_training.resources import cpu_budget
    cpu_budget(16)
    mod = importlib.import_module('.'+args.command.replace('-','_'), __package__)
    if args.command == 'baselines':
        mod.run()
    else:
        mod.run(args)


if __name__ == '__main__':
    main()
