"""Download catalog-pinned public MLX checkpoints and retain file-level provenance."""

import argparse
import json
from pathlib import Path

from .runner import digest, now, write_json


def provision(catalog, repo_id, model_root, output):
    from huggingface_hub import snapshot_download
    document = json.loads(Path(catalog).read_text())
    entry = next(item for item in document["huggingface"] if item["repo_id"] == repo_id)
    revision = entry["metadata"]["sha"]
    slug = repo_id.split("/")[-1].lower()
    target = Path(model_root) / f"{slug}-{revision[:8]}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provision.json", {"repo_id": repo_id, "revision": revision,
               "started_at": now(), "status": "downloading", "catalog_sha256": digest(catalog)})
    snapshot_download(repo_id, revision=revision, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.md"],
                      max_workers=3)
    files = [{"path": str(path.relative_to(target)), "bytes": path.stat().st_size,
              "sha256": digest(path)} for path in sorted(target.rglob("*"))
             if path.is_file() and ".cache" not in path.parts]
    config = {"system_id": slug + "-mlx", "track": "direct-generation", "type": "mlx",
              "model": str(target), "model_id": repo_id, "revision": revision,
              "quantization": json.loads((target / "config.json").read_text()).get("quantization"),
              "thinking": False, "max_tokens": 256, "timeout_s": 120, "paid": False}
    write_json(output / "config.json", config)
    write_json(output / "provision.json", {"repo_id": repo_id, "revision": revision,
               "completed_at": now(), "status": "downloaded", "catalog_sha256": digest(catalog),
               "license_metadata": (entry["metadata"].get("cardData") or {}).get("license"),
               "files": files, "config_sha256": digest(output / "config.json")})
    print(json.dumps({"repo_id": repo_id, "revision": revision, "model_path": str(target),
                      "config_path": str(output / "config.json"), "status": "downloaded"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", default="models/benchmark-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    provision(args.catalog, args.model, args.model_root, args.output)


if __name__ == "__main__":
    main()
