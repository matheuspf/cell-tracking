"""Run the isolated study stages; large outputs stay in this study's work root."""
import argparse

from .resources import cpu_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory', 'baselines', 'splits', 'profile', 'organoid-parity', 'prepare', 'feasibility', 'train', 'train-pilot', 'lock-budget', 'trace-p0', 'calibrate', 'queue'])
    parser.add_argument('--source', choices=['44b6', '6bba'])
    parser.add_argument('--arm', default='D20_temporal', choices=['D10_frozen', 'D10_adapted', 'D10_random', 'D20_compact', 'D20_temporal', 'D20_no_pretrain', 'A10', 'O10_swap'])
    parser.add_argument('--seed', type=int, default=20260915)
    args = parser.parse_args()
    cpu_budget()
    if args.stage in ['train', 'calibrate']:
        from .maintenance import drain
        drain()
    if args.stage == 'inventory':
        from .inventory import run
        run()
    elif args.stage == 'baselines':
        from .evaluate import baselines
        baselines()
    elif args.stage == 'splits':
        from .splits import run
        run()
    elif args.stage == 'profile':
        from .profile import run
        run()
    elif args.stage == 'organoid-parity':
        from .organoid_adapter import parity
        from .common import inputs, verified_graph
        import numpy as np
        row = next(r for r in inputs() if r['embryo'] == '44b6')
        g = verified_graph(row)
        print(parity(row, np.flatnonzero(g['nodes'][:, 1] == 20)[:2]))
    elif args.stage == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        from .prepare import run
        run(args.source)
    elif args.stage == 'feasibility':
        if args.source is None:
            parser.error('feasibility requires --source')
        from .feasibility import run
        run(args.source)
    elif args.stage in ['train', 'train-pilot']:
        if args.source is None:
            parser.error('training requires --source')
        if args.arm == 'O10_swap':
            if args.stage == 'train-pilot':
                parser.error('Observation training uses the already locked common update budget')
            from .observation_train import run
            run(args.source, args.seed)
        else:
            if args.arm == 'D10_random':
                from .organoid_adapter import setup
                setup().utils.set_random_seed(args.seed)
            from .train import run
            run(args.source, args.arm, args.seed, pilot=args.stage == 'train-pilot')
    elif args.stage == 'lock-budget':
        from .train import lock_budget
        lock_budget()
    elif args.stage == 'trace-p0':
        from .trace import p0
        p0()
    elif args.stage == 'calibrate':
        if args.source is None:
            parser.error('calibration requires --source')
        if args.arm == 'O10_swap':
            from .observation_calibration import run
            run(args.source, args.seed)
        else:
            from .calibration import run
            run(args.source, args.arm, args.seed)
    elif args.stage == 'queue':
        from .queue import run
        run()


if __name__ == '__main__':
    main()
