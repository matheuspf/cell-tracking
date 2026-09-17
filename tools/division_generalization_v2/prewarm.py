"""Read-only source scene preparation, ordered by clip/frame, on bounded CPU I/O."""
import time
from .common import WORK,write_json,now
from .dataset import SourceDataset
from .resources import Monitor


def run(args):
    if not 0<=args.shard<args.shards<=4:raise ValueError('At most four bounded CPU shards')
    data=SourceDataset(args.source,partition=None,image=True)
    clips={r['dataset'] for i,r in enumerate(data.rows) if i%args.shards==args.shard}
    keys=sorted((k for k,a in data.anchors.items() if a['dataset'] in clips),
                key=lambda k:(data.anchors[k]['dataset'],data.anchors[k]['time'],data.anchors[k]['anchor']))
    root=WORK/'prewarm'/args.source/str(args.shard);started=time.monotonic()
    with Monitor(root/'resources.json') as monitor:
        for i,key in enumerate(keys,1):
            a=data.anchors[key]
            data.scenes.get(a['dataset'],a['anchor'],a['position'],a['time'])
            # Raw scenes are atomically written using per-process temporary
            # paths. A concurrently training reader sees a complete old/new
            # file with the same image/code fingerprint and exact uint8 pixels.
            data.scenes.memory.clear()
            monitor.check()
            if i%128==0 or i==len(keys):
                receipt=dict(source=args.source,shard=args.shard,shards=args.shards,
                    completed=i,total=len(keys),seconds=time.monotonic()-started,time=now(),
                    raw_uint8_only=True,trainable_embeddings=False)
                write_json(root/'progress.json',receipt)
            if i%512==0 or i==len(keys):print(f'Source raw scenes {args.source}/{args.shard}: {i}/{len(keys)}',flush=True)
    write_json(root/'complete.json',receipt)
    return receipt
