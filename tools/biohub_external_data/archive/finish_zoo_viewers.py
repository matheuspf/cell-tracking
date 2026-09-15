"""Resume small viewer files over a shared HTTP/2 connection (eight requests)."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"


async def main():
    manifest = ROOT / "zoo-viewer-files.json"
    files = json.loads(manifest.read_text())
    queue = asyncio.Queue()
    done = 0
    for item in files:
        path = ROOT / item["path"]
        if path.exists() and path.stat().st_size == item["bytes"] and not path.with_name(path.name + ".aria2").exists():
            done += 1
        else:
            queue.put_nowait(item)
    print(f"Reusing {done} files; downloading {queue.qsize()} files with eight concurrent requests", flush=True)
    errors = []
    start = time.monotonic()
    async with httpx.AsyncClient(http2=True, timeout=60, limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
        async def worker():
            nonlocal done
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                path = ROOT / item["path"]
                for attempt in range(3):
                    try:
                        response = await client.get(item["url"])
                        response.raise_for_status()
                        if len(response.content) != item["bytes"]:
                            raise ValueError("Remote file size mismatch")
                        temp = path.with_name(path.name + ".part")
                        temp.write_bytes(response.content)
                        temp.replace(path)
                        path.with_name(path.name + ".aria2").unlink(missing_ok=True)
                        done += 1
                        if done % 500 == 0:
                            print(f"Downloaded {done}/{len(files)} files; elapsed {time.monotonic()-start:.0f}s", flush=True)
                        break
                    except Exception as exc:
                        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                        if status == 429:
                            raise RuntimeError("Server requested rate limiting; pause downloads") from exc
                        if attempt == 2 or status in (403, 404):
                            errors.append({"path": item["path"], "error": str(exc)})
                            break
                        await asyncio.sleep(5 * (attempt + 1))
                queue.task_done()
        await asyncio.gather(*(worker() for _ in range(8)))
    (ROOT / "zoo-viewer-errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} failed downloads; see zoo-viewer-errors.json")
    for item in files:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError("Incomplete file: " + item["path"])
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(files, indent=2))
    print(f"Complete: {len(files)} files, all sizes verified and SHA-256 recorded", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
