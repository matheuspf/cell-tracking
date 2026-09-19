"""Semantic source ancestry, content identities, and immutable stage parents."""
import hashlib
import json
from pathlib import Path
from .common import Blocked,sha,read,write


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate(artifacts,root,source,*,calibration=False):
    """Reject unknown, cyclic, target, exposed, or pretrained fitted ancestry."""
    done=set();active=set()
    def visit(key):
        if key in active:raise Blocked('Ancestry cycle')
        if key in done:return
        if key not in artifacts:raise Blocked('Unknown artifact parent: '+key)
        item=artifacts[key]
        if item.get('identity')!=digest({k:v for k,v in item.items() if k!='identity'}):raise Blocked('Manifest identity mismatch')
        if item.get('exposed') or item.get('pretrained'):raise Blocked('Exposed or pretrained parent')
        if item['kind'] not in ('code','architecture') and item.get('source')!=source:
            raise Blocked('Artifact source identity mismatch')
        if item['kind'] in ('raw_images','raw_labels'):
            if item.get('source')!=source:raise Blocked('Opposite embryo in fitted ancestry')
            permitted=('fit','calibration') if calibration else ('fit',)
            if item.get('partition') not in permitted:raise Blocked('Stage partition violation')
        elif item['kind'] not in ('code','architecture','preprocessing','model','normalizer','linear','calibration'):
            raise Blocked('Unknown semantic artifact kind')
        if item['kind'] not in ('code','architecture','raw_images','raw_labels') and not item.get('parents'):
            raise Blocked('Fitted or cached artifact has no qualified parents')
        if item['kind']=='model' and item.get('initialization')!='random':raise Blocked('Model initialization unqualified')
        active.add(key)
        for parent in item.get('parents',[]):visit(parent)
        active.remove(key);done.add(key)
    visit(root);return sorted(done)


def artifact(kind,parents=(),**fields):
    item=dict(kind=kind,parents=list(parents),**fields)
    return dict(**item,identity=digest(item))


def check_files(root,files):
    root=Path(root).resolve()
    for name,value in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise Blocked('Artifact escaped package root')
        if not path.is_file() or sha(path)!=value:raise Blocked('Missing/changed package input: '+name)


def seal(folder,manifest):
    write(Path(folder)/'manifest.json',{**manifest,'identity':digest(manifest)},immutable=True)


def unseal(folder):
    value=read(Path(folder)/'manifest.json')
    identity=value.pop('identity')
    if digest(value)!=identity:raise Blocked('Package manifest identity differs')
    check_files(folder,value['files'])
    return {**value,'identity':identity}
