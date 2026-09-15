"""Verify completed downloads without loading or executing third-party code."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"


def digest(stream):
    h = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def main():
    report_path = ROOT / "verification.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    verified = previous.get("verified", {})
    expected = {}
    manifest = json.loads((ROOT / "downloads/kaggle/biohub_synthetic/manifest.json").read_text())
    for group in ("static", "sequences"):
        for item in manifest[group]:
            expected["downloads/kaggle/biohub_synthetic/" + item["file"]] = {"bytes": item["bytes"]}
    listed = json.loads((ROOT / "kaggle-output-files.json").read_text())
    for item in listed:
        expected.setdefault("downloads/kaggle/" + item["path"], {})
    jobs = json.loads((ROOT / "external-downloads.json").read_text())
    recovery = json.loads((ROOT / "tribolium-recovery.json").read_text())
    for item in jobs:
        path = item["path"]
        if "tracks_tribolium_attributes" in path:
            path = path.replace("tracks_tribolium_attributes", "tracks_tribolium")
        expected[path] = {k: item[k] for k in ("sha256",) if k in item}
    expected["downloads/virtual-embryo-zoo/tracks_tribolium_bundle.zarr.zip"].update(
        {k: recovery[k] for k in ("sha256", "bytes")}
    )
    expected["attachments/Selection_4780.png"] = {}
    viewers_path = ROOT / "zoo-viewer-files.json"
    if viewers_path.exists():
        for item in json.loads(viewers_path.read_text()):
            expected[item["path"]] = {k: item[k] for k in ("bytes", "sha256") if k in item}
    extras_path = ROOT / "zoo-additional-files.json"
    if extras_path.exists():
        for item in json.loads(extras_path.read_text()):
            expected[item["path"]] = {}
    beetle_path = ROOT / "beetle-folder-files.json"
    if beetle_path.exists():
        for item in json.loads(beetle_path.read_text()):
            expected["downloads/virtual-embryo-zoo/tracks_tribolium_attributes_bundle.zarr/" + item["path"]] = {
                k: item[k] for k in ("bytes", "sha256") if k in item
            }
        expected["downloads/virtual-embryo-zoo/tracks_tribolium_attributes_bundle.zarr.zip"] = {}
    incomplete, errors = [], []
    for rel, constraints in expected.items():
        path = ROOT / rel
        if not path.is_file() or path.with_name(path.name + ".aria2").exists():
            incomplete.append(rel)
            continue
        stat = path.stat()
        # aria2 briefly creates the payload before its progress sidecar.
        # Leave freshly written files to the next pass to avoid that race.
        if time.time() - stat.st_mtime < 10:
            incomplete.append(rel)
            continue
        cached = verified.get(rel, {})
        if cached.get("bytes") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            continue
        try:
            if "bytes" in constraints and stat.st_size != constraints["bytes"]:
                raise ValueError(f"Size {stat.st_size} != expected {constraints['bytes']}")
            if not stat.st_size:
                raise ValueError("Empty download")
            with path.open("rb") as f:
                sha = digest(f)
            if "sha256" in constraints and sha != constraints["sha256"]:
                raise ValueError("Published SHA-256 mismatch")
            record = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha}
            if path.suffix in (".zip", ".npz"):
                with zipfile.ZipFile(path) as z:
                    bad = z.testzip()
                    if bad:
                        raise ValueError(f"ZIP CRC failed: {bad}")
                    record["archive_members"] = len(z.infolist())
                    record["zip_crc"] = "passed"
                    if "/ssbd/" in rel:
                        record["contents"] = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()]
            verified[rel] = record
        except Exception as exc:
            verified.pop(rel, None)
            errors.append({"path": rel, "error": str(exc)})
    result = {
        "expected_files": len(expected), "verified_files": len(verified),
        "verified_bytes": sum(x["bytes"] for x in verified.values()),
        "incomplete": incomplete, "errors": errors, "verified": verified,
    }
    report_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "verified" and k != "incomplete"}))
    print(f"Incomplete: {len(incomplete)}")


if __name__ == "__main__":
    main()
