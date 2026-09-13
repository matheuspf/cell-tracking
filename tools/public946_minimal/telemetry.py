"""Sample per-process GPU SM utilization for study workers (diagnostics only)."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    output=args.out/'telemetry.jsonl'
    process=subprocess.Popen(['nvidia-smi','pmon','-s','um','-d','1','-o','DT'],stdout=subprocess.PIPE,text=True)
    try:
        with output.open('a',buffering=1) as handle:
            for line in process.stdout:
                if (args.out/'stop_telemetry').exists():
                    break
                if line.lstrip().startswith('#'):
                    continue
                values=line.split()
                # -o DT prepends YYYYMMDD HH:MM:SS; pmon columns follow.
                if len(values)<13:
                    continue
                try:
                    pid=int(values[3])
                    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                except (ValueError,FileNotFoundError,PermissionError):
                    continue
                if b'public946_minimal' not in command and not any(b'worker_entry.py' in c for c in command):
                    continue
                job=None
                for i,c in enumerate(command):
                    if c==b'--job' and i+1<len(command):job=command[i+1].decode()
                    if c.endswith(b'worker_entry.py') and i+1<len(command):job=command[i+1].decode()
                handle.write(json.dumps(dict(unix_time=time.time(),pid=pid,job=job,
                    sm_percent=None if values[5]=='-' else float(values[5]),
                    memory_activity_percent=None if values[6]=='-' else float(values[6]),
                    framebuffer_mib=None if values[11]=='-' else float(values[11])))+'\n')
    finally:
        process.terminate()
        process.wait()


if __name__=='__main__':main()
