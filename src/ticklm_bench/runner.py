"""Freeze, execute and summarize independently auditable temporal benchmark runs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
import urllib.parse
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .adapters import create_adapter
from .budget import BudgetLedger, Pricing, estimate_max_cost
from .catalog import load_openrouter_key
from .scoring import parse_answer, summarize
from .tasks import audit_oracle, build_suite, deterministic_oracle

INSTRUCTION = (
    "Solve the following temporal reasoning question using only the stated facts and rules. "
    "Return exactly one option letter (A, B, C, or D), without explanation.\n\n"
)


def now():
    return datetime.now(UTC).isoformat(timespec="microseconds")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_row(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        # Keep the request identity durable before crossing the provider boundary.
        import os
        os.fsync(stream.fileno())


def _source_state():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "source_root": str(root), "package_path": str(Path(__file__).resolve().parent),
            "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}



# With an immutable checkout this captures the source files as the process starts.
_STARTUP_SOURCE = _source_state()
_EXECUTION_SOURCES = ('runner.py', 'adapters.py', 'scoring.py', 'tasks.py', 'budget.py', 'catalog.py')


def _execution_source():
    current = _source_state()
    for name in _EXECUTION_SOURCES:
        if current['source_sha256'].get(name) != _STARTUP_SOURCE['source_sha256'].get(name):
            raise ValueError(f'execution source changed since import: {name}')
    return _STARTUP_SOURCE


def _dataset_registration(dataset, dataset_hash, case_count, source, allow_unregistered):
    dataset = Path(dataset)
    manifest_path = dataset.parent / 'manifest.json'
    if manifest_path.is_file():
        frozen = json.loads(manifest_path.read_text())
        for tier, entry in frozen.get('tiers', {}).items():
            if (manifest_path.parent / entry['path']).resolve() != dataset.resolve():
                continue
            if entry['sha256'] != dataset_hash or entry['requests'] != case_count:
                raise ValueError('frozen dataset hash or count does not match')
            if frozen.get('instruction') != INSTRUCTION:
                raise ValueError('frozen dataset instruction does not match')
            frozen_sources = frozen.get('source', {}).get('source_sha256', {})
            for name in ('tasks.py', 'scoring.py'):
                if frozen_sources.get(name) != source['source_sha256'].get(name):
                    raise ValueError(f'frozen {name} source does not match')
            return {'registered': True, 'tier': tier,
                    'freeze_manifest_sha256': digest(manifest_path),
                    'freeze_manifest_path': str(manifest_path.resolve())}
    if not allow_unregistered:
        raise ValueError('unregistered dataset requires explicit opt-in')
    return {'registered': False, 'reason': 'no matching sibling freeze tier'}

def freeze(output, *, seed=11032026, pairs_per_family=16, screen_pairs=4):
    """Write test tiers sharing exact cases, plus development cases and source hashes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not 0 < screen_pairs <= pairs_per_family:
        raise ValueError("screen pairs must be positive and no larger than full tier")
    full = build_suite("test", pairs_per_family, seed)
    for case in full:
        assert deterministic_oracle(case) == audit_oracle(case)
    episodes = list(dict.fromkeys(case["episode_id"] for case in full))
    random.Random(seed).shuffle(episodes)
    order = {episode: index for index, episode in enumerate(episodes)}
    full.sort(key=lambda case: (order[case["episode_id"]], case["variant"], case["view"]))
    screen = [case for case in full if int(case["episode_id"].rsplit("-", 1)[1]) < screen_pairs]
    datasets = {"full": full, "screen": screen, "dev": build_suite("dev", 2, seed)}
    manifest = {"created_at": now(), "seed": seed, "instruction": INSTRUCTION,
                "parser": "strict single A-D, entire Answer:/Final: marker or exact answer JSON; no prose",
                "tiers": {}, "source": _source_state()}
    for tier, cases in datasets.items():
        target = output / f"{tier}.jsonl"
        target.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
        manifest["tiers"][tier] = {"path": target.name, "sha256": digest(target),
                                    "requests": len(cases), "episodes": len(cases) // 4}
    write_json(output / "manifest.json", manifest)
    return manifest



def subset(dataset, output, *, pairs_per_family=1):
    """Register a deterministic subset of a previously frozen dataset in a new directory."""
    if not isinstance(pairs_per_family, int) or isinstance(pairs_per_family, bool) or pairs_per_family <= 0:
        raise ValueError('pairs_per_family must be positive')
    dataset = Path(dataset)
    source = _execution_source()
    cases = read_rows(dataset)
    parent_hash = digest(dataset)
    registration = _dataset_registration(dataset, parent_hash, len(cases), source, False)
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('duplicate case identity')
    by_family = {}
    for case in cases:
        by_family.setdefault(case['family'], {}).setdefault(case['episode_id'], []).append(case)
    expected_families = {'duration', 'policy', 'interval', 'revision', 'event_window', 'calendar', 'chronology'}
    if set(by_family) != expected_families:
        raise ValueError('parent suite must contain all seven families')
    selected = set()
    for episodes in by_family.values():
        try:
            ordered = sorted(episodes, key=lambda episode: int(episode.rsplit('-', 1)[1]))
        except (ValueError, IndexError) as exc:
            raise ValueError('episode identity must have a numeric suffix') from exc
        if len(ordered) < pairs_per_family:
            raise ValueError('parent suite does not contain enough pairs per family')
        for episode in ordered[:pairs_per_family]:
            rows = episodes[episode]
            if len(rows) != 4 or {(row['variant'], row['view']) for row in rows} != {
                    (0, 'raw'), (0, 'derived'), (1, 'raw'), (1, 'derived')}:
                raise ValueError('episode must include both variants and both views')
            selected.add(episode)
    result = [case for case in cases if case['episode_id'] in selected]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'micro.jsonl'
    target.write_text(''.join(json.dumps(case, sort_keys=True) + '\n' for case in result))
    manifest = {'created_at': now(), 'instruction': INSTRUCTION, 'source': source,
                'parent_dataset_sha256': parent_hash,
                'parent_freeze_manifest_sha256': registration['freeze_manifest_sha256'],
                'parent_tier': registration['tier'],
                'selection': {'policy': 'lowest numeric episode indices per family; preserve parent order',
                              'pairs_per_family': pairs_per_family},
                'tiers': {'micro': {'path': target.name, 'sha256': digest(target),
                                    'requests': len(result), 'episodes': len(selected)}}}
    write_json(output / 'manifest.json', manifest)
    return manifest

def _config_check(config):
    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.lower().replace('_', '').replace('-', '')
                if normalized in {"apikey", "authorization", "password", "accesstoken", "secret",
                                  "token", "clientsecret"}:
                    raise ValueError("credentials must be loaded in memory, never from run configuration")
                if normalized == 'endpoint' and isinstance(child, str):
                    endpoint = urllib.parse.urlsplit(child)
                    if (endpoint.scheme not in ('http', 'https') or not endpoint.hostname
                            or endpoint.username is not None or endpoint.password is not None
                            or endpoint.query or endpoint.fragment):
                        raise ValueError('endpoint must not contain credentials, query, or fragment')
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(config)
    if not config.get("system_id") or not config.get("track"):
        raise ValueError("system_id and track are required")
    if config.get("paid") and (config["type"] != "http" or not config.get("hard_token_cap")):
        raise ValueError("paid requests require an HTTP adapter with an enforced billable token cap")
    return config


def runtime_metadata():
    versions = {}
    for package in ("numpy", "mlx", "mlx-lm", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hardware = {}
    if platform.system() == "Darwin":
        for key, name in (("cpu", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            try:
                hardware[key] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                hardware[key] = None
    return {"python": platform.python_version(), "os": platform.system(),
            "os_release": platform.release(), "machine": platform.machine(),
            "hardware": hardware, "packages": versions}


def local_identity(config):
    model = Path(config.get("model", ""))
    if not model.is_dir():
        return None
    names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors.index.json"}
    return {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(model.iterdir()) if p.is_file()
            and (p.name in names or p.suffix in {".safetensors", ".jinja", ".model"})}


def run(dataset, config_file, output, *, ledger_path=None, resume=False, allow_unregistered=False):
    """Persist each attempt before execution; never silently retry an uncertain request."""
    config = _config_check(json.loads(Path(config_file).read_text()))
    cases = read_rows(dataset)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case identity")
    output = Path(output)
    dataset_hash = digest(dataset)
    config_hash = digest(config_file)
    source = _execution_source()
    registration = _dataset_registration(dataset, dataset_hash, len(cases), source, allow_unregistered)
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["dataset_sha256"] != dataset_hash or manifest["config_sha256"] != config_hash:
            raise ValueError("resume requires identical dataset and configuration")
        old_sources = manifest.get('source', {}).get('source_sha256', {})
        if any(old_sources.get(name) != source['source_sha256'].get(name) for name in _EXECUTION_SOURCES):
            raise ValueError('resume requires identical execution source')
        if manifest.get('dataset_registration') != registration:
            raise ValueError('resume requires identical dataset registration')
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": str(uuid.uuid4()), "started_at": now(), "config": config,
                    "config_sha256": config_hash, "dataset_sha256": dataset_hash,
                    "expected_requests": len(cases), "source": source,
                    "dataset_registration": registration,
                    "runtime": runtime_metadata(), "status": "running"}
        write_json(output / "manifest.json", manifest)
        (output / "cases.jsonl").write_bytes(Path(dataset).read_bytes())
    record_path, journal_path = output / "records.jsonl", output / "journal.jsonl"
    records = read_rows(record_path) if record_path.exists() else []
    attempted = {row["case_id"] for row in read_rows(journal_path)} if journal_path.exists() else set()
    adapter_config = dict(config)
    ledger = None
    if config.get("paid"):
        if not ledger_path:
            raise ValueError("paid run requires a shared budget ledger")
        ledger = BudgetLedger(ledger_path, limit_usd=25)
        if config.get("credential_source") != "openrouter":
            raise ValueError("unsupported credential source")
        adapter_config["api_key"] = load_openrouter_key()
        if not adapter_config["api_key"]:
            manifest["status"] = "authentication_unavailable"
            write_json(output / "manifest.json", manifest)
            return manifest
    adapter = create_adapter(adapter_config)
    manifest["adapter"] = adapter.metadata
    manifest["local_model_files"] = local_identity(config)
    write_json(output / "manifest.json", manifest)
    max_tokens, timeout = config.get("max_tokens", 256), config.get("timeout_s", 120)
    stopping_reason = None
    try:
        for case in cases:
            if case["id"] in attempted:
                continue
            messages = [{"role": "user", "content": INSTRUCTION + case["prompt"]}]
            request_id = manifest["run_id"] + "." + case["id"]
            reserve = None
            if ledger is not None:
                pricing = Pricing(**config["pricing"])
                estimate = estimate_max_cost(
                    len(json.dumps(messages, ensure_ascii=False).encode()), max_tokens, pricing,
                    provider_max_tokens_confirmed=config.get("billable_cap_confirmed", False))
                try:
                    reserve = ledger.reserve(request_id, estimate.maximum_usd,
                                             {"system_id": config["system_id"], "case_id": case["id"]})
                except RuntimeError as exc:
                    stopping_reason = type(exc).__name__
                    break
                if not reserve.can_execute:
                    continue
            append_row(journal_path, {"request_id": request_id, "case_id": case["id"],
                                     "started_at": now(), "started_monotonic_s": time.monotonic(),
                                     "reserved_usd": str(reserve.max_usd) if reserve else None})
            result = adapter.generate(messages, max_tokens=max_tokens, timeout_s=timeout,
                                      request_id=request_id)
            accounting = None
            if ledger is not None:
                settlement = ledger.settle(request_id, result.get("usage", {}).get("cost_usd"))
                accounting = {"reserved_usd": str(reserve.max_usd),
                              "charged_usd": str(settlement.charged_usd),
                              "unknown": settlement.unknown, "blocked": settlement.blocked}
            row = {"request_id": request_id, "case": case, "result": result,
                   "predicted": parse_answer(result.get("text", "")) if result["status"] == "ok" else None,
                   "completed_at": now(), "accounting": accounting}
            append_row(record_path, row)
            records.append(row)
            print(json.dumps({"system": config["system_id"], "completed": len(records),
                              "total": len(cases), "status": result["status"]}), flush=True)
            if accounting and accounting["blocked"]:
                stopping_reason = "accounting_overrun"
                break
            if len(records) >= 3 and all(r["result"]["status"] != "ok" for r in records[-3:]):
                stopping_reason = "three_consecutive_transport_failures"
                break
    finally:
        adapter.close()
        summary = summarize(records, expected_cases=cases)
        write_json(output / "summary.json", summary)
        manifest.update({"finished_at": now(), "status": "complete" if len(records) == len(cases)
                         else "incomplete", "stopping_reason": stopping_reason,
                         "record_count": len(records), "records_sha256": digest(record_path)
                         if record_path.exists() else None})
        if ledger:
            manifest["budget"] = ledger.summary()
        write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--output", required=True)
    frozen.add_argument("--seed", type=int, default=11032026)
    micro = commands.add_parser("subset")
    micro.add_argument("--dataset", required=True)
    micro.add_argument("--output", required=True)
    micro.add_argument("--pairs-per-family", type=int, default=1)
    executed = commands.add_parser("run")
    executed.add_argument("--dataset", required=True)
    executed.add_argument("--config", required=True)
    executed.add_argument("--output", required=True)
    executed.add_argument("--ledger")
    executed.add_argument("--resume", action="store_true")
    executed.add_argument("--allow-unregistered", action="store_true",
                          help="Allow a dataset absent from a sibling freeze manifest")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.output, seed=args.seed)
    elif args.command == "subset":
        result = subset(args.dataset, args.output, pairs_per_family=args.pairs_per_family)
    else:
        result = run(args.dataset, args.config, args.output, ledger_path=args.ledger, resume=args.resume,
                     allow_unregistered=args.allow_unregistered)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "complete") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
