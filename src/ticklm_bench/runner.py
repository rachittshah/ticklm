"""Freeze, execute and summarize independently auditable temporal benchmark runs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
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
            "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


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


def _config_check(config):
    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in {"api_key", "authorization", "password", "access_token", "secret"}:
                    raise ValueError("credentials must be loaded in memory, never from run configuration")
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
    return {"python": platform.python_version(), "os": platform.system(),
            "os_release": platform.release(), "machine": platform.machine(), "packages": versions}


def run(dataset, config_file, output, *, ledger_path=None, resume=False):
    """Persist each attempt before execution; never silently retry an uncertain request."""
    config = _config_check(json.loads(Path(config_file).read_text()))
    cases = read_rows(dataset)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case identity")
    output = Path(output)
    dataset_hash = digest(dataset)
    config_hash = digest(config_file)
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["dataset_sha256"] != dataset_hash or manifest["config_sha256"] != config_hash:
            raise ValueError("resume requires identical dataset and configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": str(uuid.uuid4()), "started_at": now(), "config": config,
                    "config_sha256": config_hash, "dataset_sha256": dataset_hash,
                    "expected_requests": len(cases), "source": _source_state(),
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
                   "predicted": parse_answer(result.get("text", "")),
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
    executed = commands.add_parser("run")
    executed.add_argument("--dataset", required=True)
    executed.add_argument("--config", required=True)
    executed.add_argument("--output", required=True)
    executed.add_argument("--ledger")
    executed.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.output, seed=args.seed)
    else:
        result = run(args.dataset, args.config, args.output, ledger_path=args.ledger, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "complete") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
