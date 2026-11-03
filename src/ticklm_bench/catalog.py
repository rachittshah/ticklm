"""Read-only public model catalog discovery and safe OpenRouter credential lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_HF_MODELS_URL = "https://huggingface.co/api/models/"
_MAX_HTTP_TIMEOUT_S = 10.0
_HF_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class CatalogFetchError(RuntimeError):
    """A sanitized failure while fetching a public model catalog."""


def _load_openrouter_credential() -> tuple[str | None, str]:
    environment_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if environment_key:
        return environment_key, "environment"

    home = Path.home()
    paths = (
        (home / ".local/share/opencode/auth.json", "opencode_local_share"),
        (home / ".config/opencode/auth.json", "opencode_config"),
    )
    for path, source_kind in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            continue
        provider = document.get("openrouter") if isinstance(document, dict) else None
        key = provider.get("key") if isinstance(provider, dict) else None
        if isinstance(key, str) and key.strip():
            return key.strip(), source_kind
    return None, "none"


def load_openrouter_key() -> str | None:
    """Load a key in memory from one explicit source; never print or persist it."""
    return _load_openrouter_credential()[0]


def openrouter_auth_status() -> dict[str, bool | str]:
    """Return only availability and source kind, never credential contents."""
    key, source_kind = _load_openrouter_credential()
    return {"available": key is not None, "source_kind": source_kind}


def _http_timeout(timeout_s: float) -> float:
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > _MAX_HTTP_TIMEOUT_S):
        raise ValueError("HTTP timeout must be positive and at most 10 seconds")
    return float(timeout_s)


def _read_public_json(url: str, timeout_s: float) -> tuple[bytes, dict]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "ticklm-bench/0.1"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise CatalogFetchError(f"public catalog request failed with HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = "timeout" if isinstance(exc, TimeoutError) else "network error"
        raise CatalogFetchError(f"public catalog request failed ({reason})") from None
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise CatalogFetchError("public catalog returned invalid JSON") from None
    if not isinstance(document, dict):
        raise CatalogFetchError("public catalog JSON must be an object")
    return raw, document


def _positive_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _decimal_string(value) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return format(amount, "f")


def normalize_openrouter_catalog(document: dict) -> list[dict]:
    """Extract comparable metadata without treating catalog claims as verified access."""
    if not isinstance(document, dict):
        raise TypeError("OpenRouter catalog must be a JSON object")
    if not isinstance(document.get("data"), list):
        raise TypeError("OpenRouter catalog must contain a data list")
    normalized = []
    for model in document["data"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
        top_provider = model.get("top_provider") if isinstance(model.get("top_provider"), dict) else {}
        supported = model.get("supported_parameters")
        supported = sorted({item for item in supported if isinstance(item, str)}) if isinstance(supported, list) else []
        normalized.append({
            "id": model["id"],
            "name": model.get("name") if isinstance(model.get("name"), str) else None,
            "context_length": _positive_int(model.get("context_length")),
            "max_completion_tokens": _positive_int(top_provider.get("max_completion_tokens")),
            "supported_parameters": supported,
            "supports_max_tokens_parameter": bool({"max_tokens", "max_completion_tokens"} & set(supported)),
            "supports_temperature_parameter": "temperature" in supported,
            "supports_seed_parameter": "seed" in supported,
            "text_pricing_usd_per_token": {
                "input": _decimal_string(pricing.get("prompt")),
                "output": _decimal_string(pricing.get("completion")),
                "request": _decimal_string(pricing.get("request")),
            },
        })
    return normalized


def _validate_hf_repositories(hf_models) -> list[str]:
    if hf_models is None:
        return []
    if isinstance(hf_models, str):
        hf_models = [hf_models]
    if not isinstance(hf_models, (list, tuple)):
        raise TypeError("Hugging Face models must be a list of repo IDs")
    repositories = []
    for repo_id in hf_models:
        if not isinstance(repo_id, str) or not _HF_REPO_ID.fullmatch(repo_id):
            raise ValueError("Hugging Face model must be a public namespace/model repo ID")
        if repo_id not in repositories:
            repositories.append(repo_id)
    return repositories


def discover_catalog(output_dir: str | Path, *, hf_models=(), timeout_s: float = 10.0) -> dict:
    """Fetch public model metadata and write one immutable catalog to a new directory."""
    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    timeout = _http_timeout(timeout_s)
    repositories = _validate_hf_repositories(hf_models)

    openrouter_raw, openrouter_document = _read_public_json(_OPENROUTER_MODELS_URL, timeout)
    openrouter_entry = {
        "url": _OPENROUTER_MODELS_URL,
        "fetched_at_utc": _now(),
        "response_sha256": hashlib.sha256(openrouter_raw).hexdigest(),
        "catalog": openrouter_document,
        "models": normalize_openrouter_catalog(openrouter_document),
        "access_note": "Public catalog metadata does not prove account access or price enforcement.",
    }

    hf_entries = []
    for repo_id in repositories:
        url = _HF_MODELS_URL + urllib.parse.quote(repo_id, safe="/")
        raw, metadata = _read_public_json(url, timeout)
        hf_entries.append({
            "repo_id": repo_id,
            "url": url,
            "fetched_at_utc": _now(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": metadata,
            "access_note": "Public metadata lookup only; no model files were downloaded.",
        })

    auth_status = openrouter_auth_status()
    artifact = {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "auth_status": auth_status,
        "openrouter": openrouter_entry,
        "huggingface": hf_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    destination = output / "catalog.json"
    created_file = False
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created_file = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        if created_file:
            destination.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    return {
        "output_dir": str(output),
        "catalog_path": str(destination),
        "openrouter_model_count": len(openrouter_entry["models"]),
        "hf_model_count": len(hf_entries),
        "openrouter_response_sha256": openrouter_entry["response_sha256"],
        "auth_status": auth_status,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover public model catalog metadata without inference")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="fetch public OpenRouter and selected Hugging Face metadata")
    discover.add_argument("--output", required=True, help="new output directory; existing paths are refused")
    discover.add_argument("--hf-model", action="append", default=[], metavar="NAMESPACE/MODEL",
                          help="public Hugging Face repo metadata to include; may be repeated")
    discover.add_argument("--timeout-s", type=float, default=10.0,
                          help="per-request timeout, at most 10 seconds")
    args = parser.parse_args(argv)
    try:
        result = discover_catalog(args.output, hf_models=args.hf_model, timeout_s=args.timeout_s)
    except (CatalogFetchError, FileExistsError, ValueError, OSError) as exc:
        parser.exit(2, f"catalog discovery failed: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
