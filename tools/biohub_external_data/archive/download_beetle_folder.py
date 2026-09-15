"""Recover the public enriched Zarr folder behind a broken ZIP link."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urljoin
import zipfile
import httpx

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"
BASE = "https://public.czbiohub.org/royerlab/zoo/Tribolium/tracks_tribolium_attributes_bundle.zarr/"
DEST = ROOT / "downloads/virtual-embryo-zoo/tracks_tribolium_attributes_bundle.zarr"


def main():
    pending = [BASE]
    seen = set()
    files = []
    with httpx.Client(timeout=45, follow_redirects=True) as c:
        while pending:
            url = pending.pop(0)
            if url in seen:
                continue
            seen.add(url)
            response = c.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            items = response.json()
            for item in items:
                child = urljoin(url, item["url"])
                if not child.startswith(BASE) or item.get("is_symlink"):
                    raise ValueError("Unexpected directory entry")
                if item["is_dir"]:
                    pending.append(child)
                else:
                    files.append({"url": child, "path": child.removeprefix(BASE), "bytes": item["size"]})
            print(f"Listed {len(seen)} directories, {len(files)} files", flush=True)
            time.sleep(1)
    (ROOT / "beetle-folder-files.json").write_text(json.dumps(files, indent=2))
    lines = []
    for item in files:
        path = (DEST / item["path"]).resolve()
        if not path.is_relative_to(DEST):
            raise ValueError("Unsafe filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines.extend([item["url"], " dir=" + str(path.parent), " out=" + path.name])
    jobs = ROOT / "beetle-folder.aria2"
    jobs.write_text("\n".join(lines) + "\n")
    with (ROOT / "beetle-download.log").open("w") as log:
        subprocess.run([
            "aria2c", "--input-file=" + str(jobs), "--continue=true", "--max-concurrent-downloads=2",
            "--max-connection-per-server=1", "--split=1", "--max-tries=3", "--retry-wait=10",
            "--auto-file-renaming=false", "--allow-overwrite=false", "--console-log-level=warn",
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    for item in files:
        path = DEST / item["path"]
        if path.stat().st_size != item["bytes"] or path.with_name(path.name + ".aria2").exists():
            raise ValueError("Incomplete file: " + item["path"])
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "beetle-folder-files.json").write_text(json.dumps(files, indent=2))
    archive = DEST.with_suffix(DEST.suffix + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in files:
            path = DEST / item["path"]
            z.write(path, arcname=path.relative_to(DEST.parent))
    print(f"Recovered {len(files)} files, {sum(i['bytes'] for i in files)} bytes; rebuilt ZIP {archive.name}", flush=True)


if __name__ == "__main__":
    main()
