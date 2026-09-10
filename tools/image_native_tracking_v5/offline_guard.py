"""Process-start Python audit hook, copied as package sitecustomize.py."""
import os,sys,ipaddress,json,atexit
from pathlib import Path

INSTALLED_BEFORE_NUMERICAL=not any(k in sys.modules for k in ['torch','numpy','zarr'])
EVENTS=[];READS=0;CATEGORIES={};DEPENDENCIES=set()
PACKAGE=Path(os.environ['V5_PACKAGE_ROOT']).resolve() if os.environ.get('V5_PACKAGE_ROOT') else None
MANIFEST=json.loads((PACKAGE/'base/manifest.json').read_text()) if PACKAGE else {}
ALLOW_FILES=set(MANIFEST.get('external_checkpoint_paths',{}).values())|set(MANIFEST.get('external_teacher_paths',{}).values())
if MANIFEST.get('notebook_source_path'):ALLOW_FILES.add(MANIFEST['notebook_source_path'])
if MANIFEST.get('upstream_tracking_source_root'):
    ALLOW_FILES.update(str(Path(MANIFEST['upstream_tracking_source_root'])/p) for p in MANIFEST['tracking_source_files'])

def permitted_dependency(p):
    candidates={str(p)}
    if p.suffix=='.pyc' and p.parent.name=='__pycache__':
        candidates.add(str(p.parent.parent/(p.name.split('.cpython-')[0]+'.py')))
    for item in ALLOW_FILES:
        parts=Path(item).parts
        for component,key in [('annotation-selection-v1','V5_V1'),('strong-tracker-v2','V5_V2')]:
            if component in parts and os.environ.get(key):
                item=str(Path(os.environ[key])/Path(*parts[parts.index(component)+1:]));break
        if item in candidates:return True
    return False

def audit(event,args):
    global READS
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute().resolve();text=str(p).lower();parts=set(p.parts)
        banned=('.geff' in text or bool(parts&{'evaluation','evaluation_matches','training_labels','hoct_training_features','source_calibration','oracles','oracle_events','headroom','census'})
            or p.name=='inventory.json' or any(k in text for k in ['biohub-forum-archive','biohub-data-guide','/selected_predictions/','/candidate_graphs/','/image-native-tracking-v5/deltas/','/image-native-tracking-v5/observations/','/image-native-tracking-v5/banks/','/image-native-tracking-v5/model_scores/']))
        fresh=Path(os.environ['V5_FRESH_OUTPUT']).resolve() if os.environ.get('V5_FRESH_OUTPUT') else None
        inside_fresh=fresh is not None and p.is_relative_to(fresh)
        inside_package=PACKAGE is not None and p.is_relative_to(PACKAGE)
        old_study=bool(parts&{'annotation-selection-v1','strong-tracker-v2','strong-tracker-v3','multidata-training-v4','image-native-tracking-v5'})
        if old_study and not (inside_fresh or inside_package or permitted_dependency(p)):banned=True
        if banned:EVENTS.append(dict(event=event,denied=str(p)));raise PermissionError('V5 inference cannot read training, annotations or selected graph caches: '+str(p))
        category='image' if '.zarr/' in text else 'model' if p.suffix in ['.pt','.pth','.joblib'] else 'code' if p.suffix in ['.py','.pyc','.so'] else 'other'
        CATEGORIES[category]=CATEGORIES.get(category,0)+1
        if category=='model' or (old_study and not inside_fresh):DEPENDENCIES.add(str(p))
        READS+=1;return
    host=None
    if event in ['socket.connect','socket.sendto']:
        if isinstance(args[1],tuple):host=args[1][0]
    elif event in ['socket.getaddrinfo','socket.gethostbyname']:host=args[0]
    else:return
    if host in [None,'','localhost',os.uname().nodename]:return
    try:local=ipaddress.ip_address(host).is_loopback
    except ValueError:local=False
    if not local:EVENTS.append(dict(event=event,denied=str(host)));raise PermissionError('Offline V5 inference denies nonlocal sockets')

def record():
    path=os.environ.get('V5_AUDIT_DIR')
    if path:
        p=Path(path);p.mkdir(parents=True,exist_ok=True)
        (p/f'{os.getpid()}.json').write_text(json.dumps(dict(pid=os.getpid(),installed_before_numerical=INSTALLED_BEFORE_NUMERICAL,
            allowed_read_events=READS,categories=CATEGORIES,dependency_reads=sorted(DEPENDENCIES),blocked_events=EVENTS,
            scope='Python file/socket auditing; loopback IPC allowed, not Linux namespace isolation'))+'\n')

sys.addaudithook(audit);atexit.register(record)
