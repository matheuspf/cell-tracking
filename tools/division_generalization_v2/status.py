"""Read actual live study workers; stale progress files do not imply activity."""
from pathlib import Path
import json
import psutil
from .common import WORK,read_json,now


def snapshot():
    workers=[]
    for p in psutil.process_iter(['cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            modules=[a for a in cmd if a.startswith('division_generalization_v2')]
            if not modules:continue
            module=modules[0]
            if module=='division_generalization_v2' and 'status' in cmd:continue
            row=dict(pid=p.pid,started=p.info['create_time'],module=module)
            if module=='division_generalization_v2' and 'train' in cmd:
                value=lambda flag:cmd[cmd.index(flag)+1]
                row.update(arm=value('--arm'),source=value('--source'),seed=int(value('--seed')))
                path=WORK/'training'/row['arm']/row['source']/str(row['seed'])/'progress.json'
                if path.exists():
                    progress=read_json(path)
                    row.update(updates=progress['joint_optimizer_updates'],status=progress['status'],
                        next_boundary=int(value('--stop-at')) if '--stop-at' in cmd else 2048 if row['arm']=='prefix' else 4096)
            elif module in ('division_generalization_v2.prediction_entry','division_generalization_v2.matrix_entry'):
                job=read_json(Path(cmd[-1]));row['dataset']=job['row']['dataset']
                path=Path(job['output'])/'progress.json'
                if path.exists():
                    progress=read_json(path)
                    row.update(anchors=progress['anchors'],total_anchors=progress['total_anchors'],seconds=progress['seconds'])
            else:row['command']=cmd[cmd.index(module)+1:]
            workers.append(row)
        except (psutil.NoSuchProcess,psutil.AccessDenied,FileNotFoundError):pass
    return dict(time=now(),workers=workers,complete=(WORK/'queue/complete.json').exists(),
        failure=read_json(WORK/'queue/failure.json') if (WORK/'queue/failure.json').exists() else None)


def run(args=None):
    result=snapshot();print(json.dumps(result,indent=2));return result
