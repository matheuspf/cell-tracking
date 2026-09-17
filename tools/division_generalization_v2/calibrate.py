"""Prepare a completed checkpoint's source calibration before its full screen."""
from .common import WORK, sha


def run(args):
    from .infer import load_checkpoint
    from .dataset import SourceDataset
    from .calibration import run as calibrate
    from .resources import Monitor

    step = args.stop_at or 4096
    if args.arm == 'prefix' or step not in (3072, 4096, 8192):
        raise ValueError('Calibrate only a declared deployable checkpoint')
    checkpoint = WORK/'training'/args.arm/args.source/str(args.seed)/f'checkpoint-{step}.pt'
    root = WORK/'screens'/args.arm/args.source/str(args.seed)/str(step)
    with Monitor(root/'calibration.resources.json'):
        model, recipe = load_checkpoint(checkpoint)
        if recipe['source'] != args.source:
            raise ValueError('Explicit source does not match checkpoint')
        data = SourceDataset(args.source, 'calibration', image=model.image)
        result = calibrate(model, data, root/'calibration.json', recipe['amp'], sha(checkpoint))
    print({k: result[k] for k in ('source', 'status', 'positive_groups', 'negative_groups')}, flush=True)
    return result
