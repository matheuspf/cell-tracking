"""Copied as package sitecustomize: deny external Python socket access."""
import ipaddress
import json
import os
from pathlib import Path
import sys

NETWORK_GUARD_INSTALLED=True
BLOCKED_NETWORK_ATTEMPTS=[]
BLOCKED_EXTERNAL_DATA_READS=[]

def _local(host):
    if host in [None,'','localhost','ip6-localhost',os.uname().nodename]:return True
    try:return ipaddress.ip_address(host).is_loopback
    except ValueError:return False

def _record():
    directory=os.environ.get('V4_NETWORK_AUDIT_DIR')
    if not directory:return
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    (p/f'{os.getpid()}.json').write_text(json.dumps(dict(pid=os.getpid(),guard_installed=True,
        blocked_attempts=BLOCKED_NETWORK_ATTEMPTS,blocked_external_data_reads=BLOCKED_EXTERNAL_DATA_READS,
        scope='Python audit socket/file hooks; loopback/Unix IPC allowed'))+'\n')

def _audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=str(Path(os.fsdecode(args[0])).absolute().resolve())
        if any(x in path for x in ['biohub-forum-archive','biohub-data-guide','/cache/real/','/cache/dreal/','/cache/synthetic/','/cache/zoo/','/cache/rendered_zoo_']):
            BLOCKED_EXTERNAL_DATA_READS.append(path);_record()
            raise PermissionError('Inference: external training datasets unavailable')
        return
    host=None
    if event in ['socket.connect','socket.sendto']:
        address=args[1]
        if isinstance(address,tuple):host=address[0]
    elif event in ['socket.getaddrinfo','socket.gethostbyname']:host=args[0]
    else:return
    if not _local(host):
        BLOCKED_NETWORK_ATTEMPTS.append(dict(event=event,host=str(host)));_record()
        raise PermissionError('Offline inference: external network unavailable')

sys.addaudithook(_audit)
_record()
