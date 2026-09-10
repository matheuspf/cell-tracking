"""Portable additive V300–V370 CLI."""
import argparse
from importlib import import_module
from .context import RunContext

def enforce_variant_budget(ctx, variants):
    """Count every complete configuration, including the historical control."""
    from .common import read_json
    existing=set()
    for path in (ctx.out/'rounds').glob('*.json'):
        existing.update(read_json(path)['variants'])
    if len(existing | set(variants)) > 32:
        raise ValueError('32-configuration budget includes historical controls')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['incumbent','evaluate','census','replay','association','events','event-decode','rescue','combinations','fresh','inference','report','preserve','resources','validation','ledger-delivery','artifacts','delivery'])
    for name in ['repo','data','v1','v2','out','work','official']:p.add_argument('--'+name)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--variant');p.add_argument('--round');p.add_argument('--steps',type=int,default=10000)
    args=p.parse_args();ctx=RunContext.default(**{k:getattr(args,k) for k in ['repo','data','v1','v2','out','work','official']}).check_outputs()
    if args.stage=='preserve':
        from .incumbent import preserve_check
        print(preserve_check(ctx));return
    if args.stage=='evaluate':
        enforce_variant_budget(ctx,args.variant.split(',') if args.variant else ['incumbent'])
    if args.stage=='event-decode':
        from .event_decode_cached import verify,run
        if args.variant=='verify':verify(ctx)
        elif args.variant in [None,'run']:run(ctx,args.workers)
        else:raise ValueError('Selective event decode accepts verify or run')
        return
    if args.stage in ['ledger-delivery','artifacts','delivery']:
        import_module('.'+args.stage.replace('-','_'),'strong_tracker_v3').run(ctx)
        return
    module={'events':'event_pipeline'}.get(args.stage,args.stage)
    import_module('.'+module,'strong_tracker_v3').run(ctx,args)

if __name__=='__main__':main()
