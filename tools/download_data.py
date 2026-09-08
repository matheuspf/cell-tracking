#!/usr/bin/env python3
"""Download, safely extract, and inventory the official competition archive.

The Kaggle client resumes interrupted transfers. Extraction reads every member to
EOF, which verifies its ZIP CRC. A completed manifest makes reruns inexpensive.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import zlib

from competition_paths import COMPETITION, DATA_ROOT, REPO_ROOT
from setup_paths import main as setup_paths

MANIFEST = REPO_ROOT / "work/data-manifest.json"


def verify(manifest_path: Path, *, checksums: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["competition"] != COMPETITION or Path(manifest["data_root"]).resolve() != DATA_ROOT.resolve():
        raise ValueError("Manifest belongs to a different competition or data root")
    for entry in manifest["files"]:
        path = DATA_ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing file or size mismatch: {path}")
        if checksums:
            crc = 0
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            if f"{crc:08x}" != entry["crc32"]:
                raise RuntimeError(f"CRC mismatch: {path}")
    print(f"Verified {len(manifest['files']):,} files, {manifest['extracted_bytes']:,} bytes"
          f" ({'CRC32' if checksums else 'paths and sizes'}).", flush=True)
    return manifest


def extract(archive: Path, expected_md5: str | None = None) -> dict:
    if expected_md5:
        print("Verifying archive MD5 against the download host...", flush=True)
        with archive.open("rb") as stream:
            actual_md5 = hashlib.file_digest(stream, "md5").hexdigest()
        if actual_md5 != expected_md5.lower():
            raise RuntimeError("Archive MD5 mismatch; extraction aborted")
    else:
        actual_md5 = None
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        seen = set()
        for member in members:
            name = PurePosixPath(member.filename)
            dest = (DATA_ROOT / member.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or not dest.is_relative_to(DATA_ROOT.resolve())
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.filename in seen:
                raise ValueError(f"Duplicate ZIP member: {member.filename}")
            seen.add(member.filename)
        total = sum(m.file_size for m in members if not m.is_dir())
        # Existing members can be replaced during recovery without doubling disk use.
        additional = sum(max(0, m.file_size - ((DATA_ROOT / m.filename).stat().st_size
                         if (DATA_ROOT / m.filename).is_file() else 0))
                         for m in members if not m.is_dir())
        if shutil.disk_usage(DATA_ROOT).free < additional + 1024**3:
            raise RuntimeError(f"Insufficient free space to extract {total:,} bytes")
        print(f"Extracting {len(members):,} archive members ({total / 1024**3:.2f} GiB)...", flush=True)
        files = []
        for i, member in enumerate(members, 1):
            bundle.extract(member, DATA_ROOT)
            if not member.is_dir():
                files.append({"path": member.filename, "bytes": member.file_size,
                              "crc32": f"{member.CRC:08x}"})
            if i % 1000 == 0 or i == len(members):
                print(f"Extracted {i:,}/{len(members):,} members; ZIP CRCs checked.", flush=True)
    manifest = {
        "schema": 1,
        "competition": COMPETITION,
        "source_url": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "data_root": str(DATA_ROOT.resolve()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": actual_md5,
        "zip_crc_verified": True,
        "extracted_bytes": total,
        "files": files,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(MANIFEST)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true", help="Use an already downloaded ZIP")
    mode.add_argument("--verify-only", action="store_true", help="Check the completed local manifest")
    parser.add_argument("--checksums", action="store_true", help="Re-read extracted files to verify CRCs")
    parser.add_argument("--expected-md5", help="Optional MD5 from the official download host")
    parser.add_argument("--keep-archive", action="store_true", help="Retain the ZIP after successful extraction")
    args = parser.parse_args()
    if args.verify_only or (MANIFEST.exists() and not args.extract_only):
        verify(MANIFEST, checksums=args.checksums)
        setup_paths()
        return 0
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DATA_ROOT / f"{COMPETITION}.zip"
    if not args.extract_only:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {COMPETITION} to {DATA_ROOT}", flush=True)
        api.competition_download_files(COMPETITION, path=str(DATA_ROOT), quiet=False)
    extract(archive, args.expected_md5)
    verify(MANIFEST)
    setup_paths()
    if not args.keep_archive:
        archive.unlink()
        print("Removed the successfully extracted ZIP to reclaim space.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
