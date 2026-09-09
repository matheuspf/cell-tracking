from __future__ import annotations

import argparse
import importlib

from .common import OUT, now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--source',default='')
    p.add_argument('--variant',default='')
    p.add_argument('--steps',type=int,default=5000)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'commands.jsonl').open('a') as f:
        import json,sys
        f.write(json.dumps(dict(created=now(),argv=sys.argv))+'\n')
    module=importlib.import_module('.'+args.stage,__package__)
    module.run(args)


if __name__=='__main__':
    main()
