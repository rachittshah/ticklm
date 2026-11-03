"""Package an explicit allowlist of benchmark evidence, excluding weights and private state."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from .runner import digest, now, write_json

_SECRET_KEYS = {"apikey", "authorization", "password", "accesstoken", "refreshtoken", "clientsecret"}
_SECRET_VALUE = re.compile(r"(?:sk-(?:ant-)?[A-Za-z0-9_-]{24,}|Bearer [A-Za-z0-9._-]{24,})")


def _audit(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z]", "", key.lower())
            if normalized in _SECRET_KEYS and child not in (None, "", False):
                raise ValueError(f"credential-like field in {path.name}; review before publication")
            _audit(child, path)
    elif isinstance(value, list):
        for item in value:
            _audit(item, path)
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError(f"credential-like value in {path.name}; review before publication")


def export(root, destination, *, preflights=None):
    root, destination = Path(root), Path(destination)
    selected = []
    for name in ("frozen", "frozen-micro", "runs", "live", "provision", "source-recovery",
                 "preflights", "local-preflight", "verification"):
        directory = root / name
        if directory.exists():
            selected.extend((p, str(p.relative_to(root))) for p in directory.rglob("*")
                            if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".py", ".log", ".txt"})
    catalog = root / "catalog/catalog.json"
    if catalog.exists():
        selected.append((catalog, "catalog/catalog.json"))
    for name in ("budget-summary.json", "native-micro-plan.json", "local-runtime-plan.json",
                 "llamacpp-runtime-plan.json", "llamacpp-models.json", "llamacpp-direct-dev-check.json",
                 "ollama-direct-dev-check.json", "ollama-unload.json", "llamacpp-server.log"):
        path = root / name
        if path.exists():
            selected.append((path, name))
    external = root / "external/sdb"
    allowlist = external / "release-allowlist.txt"
    if allowlist.exists():
        for name in allowlist.read_text().splitlines():
            if not name.strip():
                continue
            path = external / name
            if not path.resolve().is_relative_to(external.resolve()) or not path.is_file():
                raise ValueError("invalid external evidence allowlist")
            selected.append((path, "external/sdb/" + name))
    for path in sorted(root.glob("execution-*.tar")):
        selected.append((path, path.name))
    if preflights:
        selected.extend((p, "preflights/" + p.name) for p in Path(preflights).iterdir()
                        if p.is_file() and p.suffix in {".json", ".txt"})
    selected.sort(key=lambda item: item[1])
    if len({name for _, name in selected}) != len(selected):
        raise ValueError("duplicate archive member names")
    for path, _ in selected:
        if path.is_symlink():
            raise ValueError("publication sources must not be symbolic links")
        if path.suffix == ".json":
            _audit(json.loads(path.read_text()), path)
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    _audit(json.loads(line), path)
        elif path.suffix in {".txt", ".log"} and _SECRET_VALUE.search(path.read_text()):
            raise ValueError(f"credential-like log content in {path.name}")
    for kind in ("runs", "live"):
        for path in (root / kind).glob("*/manifest.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("status") in (None, "running"):
                raise ValueError(f"unfinished run cannot be published: {path.parent.name}")
            hashes = dict(manifest.get("artifact_sha256", {}))
            if manifest.get("records_sha256"):
                hashes["records.jsonl"] = manifest["records_sha256"]
            for name, expected in hashes.items():
                artifact = path.parent / name
                if not artifact.resolve().is_relative_to(path.parent.resolve()) or digest(artifact) != expected:
                    raise ValueError(f"run evidence hash mismatch: {path.parent.name}")
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "ticklm-bench-v0.1-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path, name in selected:
            bundle.add(path, arcname=name, recursive=False)
    manifest = {"created_at": now(), "archive": archive.name, "archive_sha256": digest(archive),
                "files": [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
                          for path, name in selected],
                "selection": "Explicit evidence allowlist; weights, authentication profiles, caches, quarantine and arbitrary logs excluded."}
    write_json(destination / "manifest.json", manifest)
    with tarfile.open(archive) as bundle:
        actual = bundle.getnames()
        if actual != [name for _, name in selected]:
            raise ValueError("archive member verification failed")
        for member, (path, _) in zip(bundle.getmembers(), selected, strict=True):
            stream = bundle.extractfile(member)
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != digest(path):
                raise ValueError("archive content verification failed")
    return {"archive": str(archive), "files": len(selected), "bytes": archive.stat().st_size,
            "sha256": manifest["archive_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--preflights")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.root, args.destination, preflights=args.preflights)))


if __name__ == "__main__":
    main()
