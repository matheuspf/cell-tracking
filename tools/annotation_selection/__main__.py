from __future__ import annotations

import argparse
import sys

from .common import OUT, now, write_json


def main():
    p=argparse.ArgumentParser(description='Resumable local annotation selection protocol')
    p.add_argument('stage',choices=['environment','inventory','candidates','splits','labels','train','image','freeze','infer','evaluate','controls','report','classifiers','public','public-repair-pilot','public-repair','audit','audit-viewer','lock','public-prepare','public-evaluate','export','matched-controls','resources','browser-validate','validate','finalize'])
    p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--source')
    p.add_argument('--output')
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'commands.jsonl').open('a') as f:
        import json
        f.write(json.dumps(dict(time=now(),argv=sys.argv,executable=sys.executable))+'\n')
    if a.stage=='environment':
        from .inventory import environment,provenance
        environment(); provenance()
    elif a.stage=='inventory':
        from .inventory import run
        run()
    else:
        import importlib
        name={'image':'train','freeze':'train','infer':'inference','controls':'report','classifiers':'report','public':'public_lane','public-repair-pilot':'public_repair','public-repair':'public_repair','audit':'quality_audit','audit-viewer':'audit_viewer','lock':'locking','public-prepare':'public_analysis','public-evaluate':'public_analysis','export':'exports','matched-controls':'matched_controls','browser-validate':'browser_validation','validate':'validation'}.get(a.stage,a.stage)
        mod=importlib.import_module('.'+name,__package__)
        getattr(mod,'pilot' if a.stage=='public-repair-pilot' else 'evaluate' if a.stage=='public-evaluate' else a.stage if a.stage in ('image','freeze','infer','controls','classifiers') else 'run')(a)


if __name__=='__main__':
    main()
