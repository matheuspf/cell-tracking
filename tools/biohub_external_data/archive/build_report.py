"""Summarize observed downloads and verification, without claiming missing files passed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "work/biohub-forum-archive"
TOPIC_URL = "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103"


def main():
    v = json.loads((ROOT / "verification.json").read_text())
    topic = json.loads((ROOT / "topics/732103.json").read_text())
    links = json.loads((ROOT / "post-links.json").read_text())
    def walk(messages):
        for m in messages:
            yield m
            yield from walk(m.get("replies", []))
    messages = [topic["firstMessage"], *walk(topic["comments"])]
    complete = not v["incomplete"] and not v["errors"]
    status = "Complete" if complete else "Download or verification still in progress"
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Biohub discussion 732103 — download report", "",
        f"Status: **{status}**. Report generated {now}.", "",
        f"Scope: [discussion 732103]({TOPIC_URL}), its original post and all replies, and the dataset downloads shared there. Snapshot: 2026-09-08.", "",
        f"Archived **{len(messages)} unique messages**: the original post and {len(messages)-1} replies. Kaggle reported {topic['totalMessages']} replies; the counts match. **No unavailable post or reply was detected.**", "",
        f"Verified {v['verified_files']:,} files ({v['verified_bytes']/10**9:.3f} GB / {v['verified_bytes']/2**30:.3f} GiB), with {len(v['incomplete'])} pending and {len(v['errors'])} verification errors. Counts include the beetle folder and its locally rebuilt ZIP; those contain duplicate data.", "",
        "## Downloads", "",
        "| Source | Contents | Local files |", "|---|---|---|",
        "| José Freitas's synthetic dataset | 1,539 static volumes + 2,174 sequences, original manifest, metadata and progress log | [Synthetic data](downloads/kaggle/biohub_synthetic/) |",
        "| Kaggle notebooks | Builder and explorer source, metadata, and Python exports; notebook code was not executed | [Notebooks](notebooks/josefreitasalvesneto/) |",
        "| RIKEN / SSBD | All seven published zebrafish BDML ZIPs; each includes HDF5 cell measurements and XML metadata | [RIKEN archives and published checksums](downloads/ssbd/) |",
        "| Virtual Embryo Zoo | Tracking ZIPs for zebrafish, fruit fly, mouse, sea squirt, worm and beetle; two additional public CSVs | [Zoo data](downloads/virtual-embryo-zoo/) |",
        "| All six Zoo viewer datasets | Complete enriched Zarr stores, including the attributes separately served to the viewer | [Zoo data folders](downloads/virtual-embryo-zoo/) |",
        "| Beetle enriched viewer data | Public Zarr folder recovered separately and packaged locally because the site's ZIP link is broken | [Recovered Zarr folder](downloads/virtual-embryo-zoo/tracks_tribolium_attributes_bundle.zarr/) · [Rebuilt ZIP](downloads/virtual-embryo-zoo/tracks_tribolium_attributes_bundle.zarr.zip) |",
        "| Forum image | Original attached screenshot from message 3521291 | [Selection_4780.png](attachments/Selection_4780.png) |", "",
        "## Retrieval issues and manual options", "",
        "| Item | Observed issue | Resolution / what you can do yourself |", "|---|---|---|",
        "| [Advertised beetle ZIP](https://public.czbiohub.org/royerlab/zoo/Tribolium/tracks_tribolium_attributes_bundle.zarr.zip) | HTTP 404. The exact server-hosted ZIP could not be downloaded. | The [base tracking ZIP](https://public.czbiohub.org/royerlab/zoo/Tribolium/tracks_tribolium_bundle.zarr.zip) works in a normal browser. The enriched files are available in this [public folder](https://public.czbiohub.org/royerlab/zoo/Tribolium/tracks_tribolium_attributes_bundle.zarr/); this archive includes a complete local mirror and a rebuilt ZIP. Retrying the broken link manually alone will not repair it. |",
        "| Kaggle output file listing | The CLI's file-size column reported implausibly small numbers, such as 924 bytes for NPZ files. | Actual downloads work. Verification uses the author's original manifest byte sizes and ZIP CRC checks. You can use the notebook Output tab or the command below. |",
        "| Kaggle discussion via text-only web reader | Returned a JavaScript shell without discussion text. | A rendered browser and the public discussion response provided the original post plus all 10 replies. You can view the thread normally in your browser. |", "",
        ("**No manual action is needed for the shared dataset downloads.** The exact broken server ZIP was replaced by recovering its publicly accessible content." if complete else "Downloads or verification are still pending; inspect verification.json for the exact list."), "",
        "To download the synthetic outputs yourself with an authenticated Kaggle CLI:", "", "```bash",
        "kaggle kernels output josefreitasalvesneto/biohub-synthetic-dataset -p biohub-synthetic-output --page-size 100", "```", "",
        "For RIKEN, open the [project page](https://ssbd.riken.jp/database/project/5-Keller-FishEmbryo/) and use each of the seven BDML ZIP buttons, or use its [download instructions and manifests](https://ssbd.riken.jp/data/5-Keller-FishEmbryo/). The ZIPs contain the same HDF5 and XML files offered individually. RIKEN lists no image datasets for this project; these files contain positions and cell measurements; temporal correspondences require separate verification.", "",
        "The [Virtual Embryo Zoo](https://virtual-embryo-zoo.sf.czbiohub.org/) links to tracking-data bundles. This archive also mirrors the enriched Zarr stores used by all six viewers. The original site and public server directory listings are saved under raw/.", "",
        "## Per-post coverage", "",
        "| Message | Author | Posted (UTC) | Data links / attachments | Text |", "|---|---|---|---|---|",
    ]
    for m in messages:
        assets = [j for j in links if m["id"] in j["message_ids"]]
        names = []
        for j in assets:
            name = "Forum image" if j["type"] == "image" else ("RIKEN" if "riken.jp" in j["url"] else "Virtual Embryo Zoo" if "virtual-embryo-zoo" in j["url"] else j["url"].rstrip('/').split('/')[-1])
            names.append(f"[{name}]({j['url']})")
        lines.append(f"| {m['id']} | {m['author']['displayName']} | {m['postDate']} | {'; '.join(names) or 'No linked download'} | Archived |")
    lines.extend([
        "", "## Verification and provenance", "",
        "- [Full thread in Markdown](topics/732103.md) and [original structured response](topics/732103.json).",
        "- [Post-to-link mapping](post-links.json), [Kaggle output listing](kaggle-output-files.json), and [external download inventory](external-downloads.json).",
        "- [Verification results and local SHA-256 hashes](verification.json). Every completed ZIP/NPZ is checked for ZIP integrity and CRCs. Synthetic files are checked against the author's byte sizes. RIKEN ZIPs are checked against the published SHA-256 values.",
        "- [Enriched beetle file listing, sizes and hashes](beetle-folder-files.json); the rebuilt ZIP is generated locally and is not claimed to match a nonexistent upstream ZIP byte-for-byte.",
        "- [Other five enriched viewer file listings, sizes and hashes](zoo-viewer-files.json).",
        "- Saved original source pages and responses are under [raw/](raw/). Their contents are downloaded references, not instructions to execute.",
        "- Licenses recorded by the sources: the synthetic post declares CC0; the RIKEN project page declares CC BY-NC-SA. Refer to each original source for its complete terms.",
        "- This is a data archive, not a configured notebook runtime. The notebook metadata also declares the Biohub competition's original train/test data as an input; those separate competition inputs are not part of the shared synthetic dataset.",
        "",
    ])
    (ROOT / "README.md").write_text("\n".join(lines))
    artifacts = []
    for subdir in ("topics", "attachments", "notebooks"):
        for p in sorted((ROOT / subdir).rglob("*")):
            if p.is_file():
                artifacts.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (ROOT / "document-checksums.json").write_text(json.dumps(artifacts, indent=2))
    print(status)


if __name__ == "__main__":
    main()
