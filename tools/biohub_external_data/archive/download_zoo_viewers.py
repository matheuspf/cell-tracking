"""Archive the five remaining enriched viewer stores listed on the Zoo site."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urljoin
import httpx

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"
NAMES = {"Zebrafish": "zebrafish", "Drosophila": "drosophila", "Mouse": "mouse", "Ascidian": "ascidian", "C_elegans": "elegans"}


def main():
    files = []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for species, slug in NAMES.items():
            directory = f"tracks_{slug}_attributes_bundle.zarr"
            base = f"https://public.czbiohub.org/royerlab/zoo/{species}/{directory}/"
            pending, seen = [base], set()
            start = len(files)
            while pending:
                url = pending.pop(0)
                if url in seen:
                    continue
                seen.add(url)
                response = client.get(url, headers={"Accept": "application/json"})
                response.raise_for_status()
                for item in response.json():
                    child = urljoin(url, item["url"])
                    if not child.startswith(base) or item.get("is_symlink"):
                        raise ValueError("Unexpected listing entry")
                    if item["is_dir"]:
                        pending.append(child)
                    else:
                        rel = "downloads/virtual-embryo-zoo/" + directory + "/" + child.removeprefix(base)
                        files.append({"url": child, "path": rel, "bytes": item["size"]})
                time.sleep(1)
            print(f"{species}: {len(files)-start} files", flush=True)
    inventory = ROOT / "zoo-viewer-files.json"
    inventory.write_text(json.dumps(files, indent=2))
    lines = []
    for item in files:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Unsafe filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines.extend([item["url"], " dir=" + str(path.parent), " out=" + path.name])
    jobs = ROOT / "zoo-viewers.aria2"
    jobs.write_text("\n".join(lines) + "\n")
    print(f"Downloading {len(files)} files, {sum(i['bytes'] for i in files)} bytes", flush=True)
    with (ROOT / "zoo-viewer-download.log").open("w") as log:
        subprocess.run([
            "aria2c", "--input-file=" + str(jobs), "--continue=true", "--max-concurrent-downloads=4",
            "--max-connection-per-server=1", "--split=1", "--max-tries=3", "--retry-wait=10",
            "--auto-file-renaming=false", "--allow-overwrite=false", "--console-log-level=warn",
            "--download-result=hide", "--summary-interval=30",
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    for item in files:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or path.with_name(path.name + ".aria2").exists():
            raise ValueError("Incomplete file: " + item["path"])
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory.write_text(json.dumps(files, indent=2))
    print("All five enriched viewer stores downloaded and checked", flush=True)


if __name__ == "__main__":
    main()
