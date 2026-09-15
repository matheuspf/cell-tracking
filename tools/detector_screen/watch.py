"""Refresh detector evidence as the finite, already-launched assessment jobs finish."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'work/detector-screen-20260914'
METHODS=['incumbent','cellect','organoid','nucverse','spotiflow','spotiflow_loose',
         'xenopus','xenopus_centroid','pacmap','cellpose_cpdino_vitb']


def main():
    panel=json.loads((ROOT/'panel.json').read_text())
    keys={r['key'] for r in panel['frames'] if r['role']=='assessment'}
    previous=None;previous_complete=set()
    while True:
        counts={m:len(keys & {p.stem for p in (ROOT/'predictions'/m).glob('*.npz')}) for m in METHODS}
        complete={m for m,n in counts.items() if n==len(keys)}
        print(time.strftime('%H:%M:%S'),json.dumps(counts),flush=True)
        refresh=(previous is None or sum(counts.values())-sum(previous.values())>=80 or
                 complete!=previous_complete or len(complete)==len(METHODS))
        if refresh:
            with (ROOT/'watch-results.log').open('a') as log:
                for command in [
                    ['tools.detector_screen.evaluate'],
                    ['tools.detector_screen.analyze_failures'],
                    ['tools.detector_screen.publish','--required',*sorted(complete)],
                    ['tools.detector_screen.refresh_canvas'],
                ]:
                    subprocess.run([sys.executable,'-m',*command],cwd=REPO,
                                   env=dict(os.environ,PYTHONNOUSERSITE='1'),
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            previous=counts.copy();previous_complete=complete.copy()
            (ROOT/'watch-status.json').write_text(json.dumps(dict(
                assessment_frames=counts,complete=sorted(complete),
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
            print('EVIDENCE_REFRESHED',len(complete),'/',len(METHODS),'complete',flush=True)
        if len(complete)==len(METHODS):
            print('ALL_REQUIRED_ASSESSMENT_OUTPUTS_COMPLETE',flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
