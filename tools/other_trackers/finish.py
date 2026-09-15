"""Merge complete private clips, decode finished clips, then score OrganoidTracker."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time

from .common import ROOT, RESULTS, CONFIG, clips, read, sha, write


def complete(path):
    return all((path/f"t{t:03}.npz").exists() and (path/f"t{t:03}.json").exists() for t in range(100))


def main():
    merged=[]
    private=ROOT/"organoid-worker/organoid/frames"
    while True:
        progress={}
        for clip in clips():
            name=clip["dataset"]
            dest=ROOT/"organoid/frames"/name
            source=private/name
            if complete(source) and not dest.exists():
                for t in range(100):
                    path=source/f"t{t:03}.npz"
                    receipt=read(path.with_suffix(".json"))
                    assert receipt["sha256"]==sha(path) and receipt["inputs"]["config"]==sha(CONFIG)
                dest.parent.mkdir(parents=True,exist_ok=True)
                staging=dest.with_name(name+".incoming")
                shutil.copytree(source,staging)
                staging.rename(dest)
                merged.append(name)
                write(ROOT/"private-merge.json",dict(clips=merged,source=str(private),complete_clips_only=True))
                print("MERGED",name,flush=True)
            full=complete(dest)
            graph_paths=[ROOT/"graphs"/arm/f"{name}.json" for arm in ("organoid","organoid_no_division")]
            if full and not all(p.exists() for p in graph_paths):
                print("DECODING",name,flush=True)
                subprocess.run([sys.executable,"-m","tools.other_trackers.decode","organoid","--datasets",name],check=True)
            progress[name]=dict(primary_frames=len(list(dest.glob("t*.json"))),
                                private_frames=len(list(source.glob("t*.json"))),graphs_complete=all(p.exists() for p in graph_paths))
        write(ROOT/"progress.json",progress)
        if all(v["graphs_complete"] for v in progress.values()):break
        time.sleep(10)
    subprocess.run([sys.executable,"-m","tools.other_trackers.evaluate","--families","organoid"],check=True)
    subprocess.run([sys.executable,"-m","tools.other_trackers.report"],check=True)
    print("BOTH TRACKER EXPERIMENTS SCORED",flush=True)


if __name__=="__main__":main()
