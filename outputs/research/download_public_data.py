"""Download the public TacBench GelSight force subset; no credentials required.

Original dataset: https://huggingface.co/datasets/facebook/gelsight-force-estimation
License: CC-BY-NC-4.0. Preserve attribution; do not redistribute as newly collected data.
Downloads compressed tactile frames and force metadata, not large org_* duplicates.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "facebook/gelsight-force-estimation"

def fetch(item, root, revision):
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"
    expected = item.get("lfs", {}).get("oid")
    for attempt in range(3):
        try:
            if not path.exists() or path.stat().st_size != item["size"]:
                request = urllib.request.Request(url, headers={"User-Agent": "public-tactile-research/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, path.with_suffix(path.suffix + ".partial").open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                path.with_suffix(path.suffix + ".partial").replace(path)
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            if path.stat().st_size != item["size"] or (expected and expected != digest):
                raise ValueError(f"Integrity mismatch: {path}")
            print(f"OK {item['path']} {path.stat().st_size}", flush=True)
            return {"path": item["path"], "url": url, "bytes": path.stat().st_size, "sha256": digest}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("work/data/gelsight_force"))
    parser.add_argument("--tree", type=Path, default=Path(__file__).with_name("data_download_manifest.json"))
    parser.add_argument("--revision", default="136db55485b6501605b1d2648ce0fe44740e302e")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", nargs="*", help="Optional shape/batch_N prefixes")
    args = parser.parse_args()
    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    if isinstance(tree, dict) and "files" in tree:
        tree = [{"type": "file", "path": row["path"], "size": row["bytes"],
                 "lfs": {"oid": row["sha256"]}} for row in tree["files"]]
    items = [x for x in tree if x["type"] == "file" and
             (x["path"].endswith("README.md") or Path(x["path"]).name.startswith("dataset_")) and
             (not args.batches or x["path"].endswith("README.md") or any(x["path"].startswith(b + "/") for b in args.batches))]
    print(f"Downloading {len(items)} files, {sum(i['size'] for i in items)/1e9:.3f} GB", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifest = list(pool.map(lambda item: fetch(item, args.root, args.revision), items))
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "download_manifest.json").write_text(json.dumps({"repository": REPO, "revision": args.revision,
        "license": "CC-BY-NC-4.0", "files": manifest}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
