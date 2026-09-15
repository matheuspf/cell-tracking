#!/usr/bin/env python3
"""Restore the exact local competition archive directly from Kaggle by HTTP range.

The input manifest is a copy of work/data-manifest.json with source file mtimes.
This avoids uploading 82 GiB from the local machine or storing a second 82 GiB
ZIP on the destination. Archive membership, sizes and CRCs must match the local
snapshot before extraction starts. Existing correct files are retained; existing
conflicting files are never overwritten. The reference sync client is unchanged.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import zipfile
import zlib

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFilesRequest


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, etag, block_size=16 * 1024**2):
        super().__init__()
        self.url, self.size, self.etag = url, size, etag
        self.block_size = block_size
        self.position = 0
        self.block_start, self.block = -1, b""
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else min(length, self.size - self.position)
        result = bytearray()
        while length > 0:
            if not self.block_start <= self.position < self.block_start + len(self.block):
                start = self.position // self.block_size * self.block_size
                end = min(self.size - 1, start + self.block_size - 1)
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                if self.etag:
                    headers["If-Match"] = self.etag
                for attempt in range(6):
                    try:
                        response = self.session.get(self.url, headers=headers, timeout=(30, 90))
                        if response.status_code != 206:
                            raise RuntimeError(f"Archive range returned HTTP {response.status_code}")
                        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                            raise RuntimeError("Unexpected archive range identity")
                        block = response.content
                        if len(block) != end - start + 1:
                            raise RuntimeError("Short archive range response")
                        self.block_start, self.block = start, block
                        break
                    except (requests.RequestException, RuntimeError):
                        if attempt == 5:
                            raise RuntimeError(f"Failed to read archive range at {start}") from None
                        time.sleep(min(2**attempt, 30))
            offset = self.position - self.block_start
            chunk = self.block[offset:offset + length]
            result.extend(chunk)
            self.position += len(chunk)
            length -= len(chunk)
        return bytes(result)

    def close(self):
        self.session.close()
        super().close()


def crc(path):
    value = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            value = zlib.crc32(chunk, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tmp-root", type=Path, default=Path.home() / "tmp/competition-restore")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    destination = Path(manifest["data_root"])
    expected = {row["path"]: row for row in manifest["files"]}
    for name in expected:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = manifest["competition"]
        response = client.competitions.competition_api_client.download_data_files(request)
        url = response.url
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        for part in response.headers.get("x-goog-hash", "").split(","):
            if part.strip().startswith("md5="):
                actual = base64.b64decode(part.strip()[4:]).hex()
                if actual != manifest["archive_md5"]:
                    raise ValueError("Kaggle archive MD5 differs from the local snapshot")
        response.close()
    if size != manifest["archive_bytes"]:
        raise ValueError("Kaggle archive size differs from the local snapshot")
    with RangeReader(url, size, etag) as reader, zipfile.ZipFile(reader) as bundle:
        files = [entry for entry in bundle.infolist() if not entry.is_dir()]
        actual = {entry.filename: (entry.file_size, entry.CRC) for entry in files}
        wanted = {name: (row["bytes"], int(row["crc32"], 16)) for name, row in expected.items()}
        if len(actual) != len(files) or actual != wanted:
            raise ValueError("Archive file inventory differs from the local snapshot")
        files.sort(key=lambda entry: entry.header_offset)
    print(f"Exact archive confirmed: {len(files):,} files, {size:,} archive bytes.", flush=True)
    if args.check_only:
        return
    destination.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(destination)
    missing = []
    for entry in files:
        target = destination / entry.filename
        if target.exists():
            if target.stat().st_size != entry.file_size or crc(target) != entry.CRC:
                raise ValueError(f"Existing input conflicts with the source: {entry.filename}")
        else:
            missing.append(entry)
    needed = sum(entry.file_size for entry in missing)
    if needed + 8 * 1024**3 > disk.free:
        raise RuntimeError("Insufficient free space for the missing inputs")
    print(f"Restoring {len(missing):,} missing files ({needed / 1024**3:.2f} GiB).", flush=True)
    lock = threading.Lock()
    progress = {"files": 0, "bytes": 0, "last": time.monotonic()}
    tmp_root = args.tmp_root
    tmp_root.mkdir(parents=True, exist_ok=True)
    if tmp_root.stat().st_dev != destination.stat().st_dev:
        raise RuntimeError("Temporary directory must share the destination filesystem")

    def worker(entries, worker_id):
        partial = tmp_root / f"worker-{worker_id}.partial"
        with RangeReader(url, size, etag) as reader, zipfile.ZipFile(reader) as bundle:
            for entry in entries:
                target = destination / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry.filename) as stream, partial.open("wb") as output:
                    shutil.copyfileobj(stream, output, 1024**2)
                # ZipExtFile validates the CRC when EOF is reached.
                if partial.stat().st_size != entry.file_size:
                    raise ValueError(f"Extracted size mismatch: {entry.filename}")
                row = expected[entry.filename]
                os.chmod(partial, row.get("mode", 0o644))
                if "mtime_ns" in row:
                    os.utime(partial, ns=(row["mtime_ns"], row["mtime_ns"]))
                if target.exists():
                    raise FileExistsError(f"Input appeared during restore: {entry.filename}")
                partial.replace(target)
                with lock:
                    progress["files"] += 1
                    progress["bytes"] += entry.file_size
                    if time.monotonic() - progress["last"] >= 30:
                        print(f"Restored {progress['files']:,}/{len(missing):,} files; {progress['bytes'] / 1024**3:.2f}/{needed / 1024**3:.2f} GiB.", flush=True)
                        progress["last"] = time.monotonic()

    count = max(1, min(args.workers, 12))
    # Contiguous archive ranges minimize redundant HTTP reads between workers.
    groups = [[] for _ in range(count)]
    for entry in missing:
        groups[min(count - 1, entry.header_offset * count // size)].append(entry)
    with ThreadPoolExecutor(max_workers=count) as pool:
        for future in as_completed([pool.submit(worker, group, i) for i, group in enumerate(groups)]):
            future.result()
    print(f"Restoration complete: {progress['files']:,} files, each checked against its source CRC.", flush=True)


if __name__ == "__main__":
    main()
