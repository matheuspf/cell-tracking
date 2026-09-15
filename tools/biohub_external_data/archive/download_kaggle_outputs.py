"""Mirror the published synthetic outputs; never execute the notebook."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"
DEST = ROOT / "downloads/kaggle"
REFERENCE = "josefreitasalvesneto/biohub-synthetic-dataset"


def main():
    k = importlib.import_module("kaggle.api.kaggle_api_extended")
    api = k.KaggleApi()
    api.authenticate()
    entries = []
    lines = []
    token = None
    with api.build_kaggle_client() as client:
        for page in range(1, 1000):
            req = k.ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = REFERENCE.split("/")
            req.page_size = 100
            if token:
                req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in response.files or []:
                path = (DEST / item.file_name).resolve()
                if not path.is_relative_to(DEST.resolve()):
                    raise ValueError("Unsafe output path")
                entries.append({"path": item.file_name})
                lines.extend([item.url, " dir=" + str(path.parent), " out=" + path.name])
            if page == 1 and response.log:
                (ROOT / "raw/biohub-synthetic-dataset.log").write_text(response.log)
            print(f"Listed page {page}: {len(entries)} output files", flush=True)
            token = response.next_page_token
            if not token:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Pagination did not finish")
    (ROOT / "kaggle-output-files.json").write_text(json.dumps(entries, indent=2))
    # Signed download URLs are short-lived; keep them local with restricted permissions.
    request_file = ROOT / ".kaggle-downloads.aria2"
    fd = os.open(request_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    with (ROOT / "kaggle-download.log").open("w") as log:
        result = subprocess.run([
            "aria2c", "--input-file=" + str(request_file), "--continue=true",
            "--max-concurrent-downloads=4", "--max-connection-per-server=1", "--split=1",
            "--max-tries=3", "--retry-wait=10", "--connect-timeout=30", "--timeout=60",
            "--auto-file-renaming=false", "--allow-overwrite=false",
            "--summary-interval=30", "--console-log-level=warn", "--download-result=hide",
        ], stdout=log, stderr=subprocess.STDOUT)
    print(f"Download process exited {result.returncode}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
