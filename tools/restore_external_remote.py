#!/usr/bin/env python3
"""Restore large external inputs directly, checking the saved local SHA-256s."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp-root", type=Path, default=Path.home() / "tmp/external-restore")
    parser.add_argument("--receipt", type=Path, default=Path.home() / "setup/external-direct-restore.json")
    parser.add_argument("--metadata-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "work/biohub-forum-archive"
    metadata_root = args.metadata_root or root
    verified = json.loads((metadata_root / "verification.json").read_text())["verified"]
    expected = {k: v for k, v in verified.items()
                if k.startswith(("downloads/kaggle/", "downloads/ssbd/"))}
    urls = {item["path"]: item["url"]
            for item in json.loads((metadata_root / "external-downloads.json").read_text())
            if item["path"] in expected}
    api = KaggleApi()
    api.authenticate()
    token = None
    with api.build_kaggle_client() as client:
        for page in range(1000):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = "josefreitasalvesneto"
            request.kernel_slug = "biohub-synthetic-dataset"
            request.page_size = 100
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = "downloads/kaggle/" + item.file_name
                if name in expected:
                    urls[name] = item.url
            token = response.next_page_token
            if not token:
                break
            time.sleep(1)  # Preserve the original source-listing throttle.
        else:
            raise RuntimeError("Kaggle output pagination did not finish")
    missing_urls = set(expected) - set(urls)
    if missing_urls:
        raise RuntimeError(f"No current source URL for {len(missing_urls)} saved files")
    print(f"Located original URLs for {len(expected):,} files; retaining the saved checksums.", flush=True)
    tmp = args.tmp_root
    tmp.mkdir(parents=True, exist_ok=True)
    progress = {"files": 0, "bytes": 0, "last": time.monotonic()}
    lock = threading.Lock()

    def download(name):
        metadata = expected[name]
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe source path")
        if target.exists():
            with target.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if target.stat().st_size != metadata["bytes"] or actual != metadata["sha256"]:
                raise ValueError(f"Existing input differs from the saved source: {name}")
        else:
            partial = tmp / (hashlib.sha256(name.encode()).hexdigest() + ".partial")
            for attempt in range(4):
                try:
                    digest = hashlib.sha256()
                    with requests.get(urls[name], stream=True, timeout=(30, 90)) as response:
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP {response.status_code}")
                        with partial.open("wb") as output:
                            for chunk in response.iter_content(1024**2):
                                output.write(chunk)
                                digest.update(chunk)
                    if partial.stat().st_size != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
                        raise ValueError(f"Upstream input differs from the local snapshot: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(f"Input appeared during restore: {name}")
                    os.utime(partial, ns=(metadata["mtime_ns"], metadata["mtime_ns"]))
                    partial.replace(target)
                    break
                except (requests.RequestException, RuntimeError):
                    if attempt == 3:
                        raise RuntimeError(f"Could not retrieve {name}") from None
                    time.sleep(10)
        with lock:
            progress["files"] += 1
            progress["bytes"] += metadata["bytes"]
            if time.monotonic() - progress["last"] >= 30:
                print(f"Verified {progress['files']:,}/{len(expected):,} restored files; {progress['bytes'] / 1024**3:.2f} GiB.", flush=True)
                progress["last"] = time.monotonic()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download, name) for name in expected]):
            future.result()
    receipt = {"status": "passed", "files": progress["files"], "bytes": progress["bytes"],
               "all_source_sha256_match": True}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
