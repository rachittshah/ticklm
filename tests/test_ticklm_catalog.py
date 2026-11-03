import hashlib
import json
from urllib.parse import urlsplit

import pytest

from ticklm_bench import catalog


def test_openrouter_key_prefers_environment_and_status_never_exposes_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-secret-value")

    assert catalog.load_openrouter_key() == "env-secret-value"
    assert catalog.openrouter_auth_status() == {
        "available": True,
        "source_kind": "environment",
    }


def test_openrouter_key_reads_only_exact_provider_key_from_standard_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(catalog.Path, "home", lambda: tmp_path)
    auth = tmp_path / ".local" / "share" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({
        "openrouter": {"type": "api", "key": "or-secret"},
        "anthropic": {"type": "api", "key": "other-secret"},
    }))

    assert catalog.load_openrouter_key() == "or-secret"
    assert catalog.openrouter_auth_status() == {
        "available": True,
        "source_kind": "opencode_local_share",
    }


def test_openrouter_key_falls_back_to_config_auth_and_ignores_similar_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(catalog.Path, "home", lambda: tmp_path)
    auth = tmp_path / ".config" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"OpenRouter": {"key": "wrong-case"},
                                "openrouter": {"api_key": "wrong-field"}}))

    assert catalog.load_openrouter_key() is None
    assert catalog.openrouter_auth_status() == {
        "available": False,
        "source_kind": "none",
    }

    auth.write_text(json.dumps({"openrouter": {"key": "config-secret"}}))
    assert catalog.load_openrouter_key() == "config-secret"
    assert catalog.openrouter_auth_status()["source_kind"] == "opencode_config"


def test_normalize_openrouter_catalog_keeps_prices_as_exact_strings():
    normalized = catalog.normalize_openrouter_catalog({"data": [{
        "id": "vendor/model",
        "name": "Model",
        "context_length": 32768,
        "pricing": {"prompt": "0.000001", "completion": "0.000002", "request": "0.001"},
        "top_provider": {"max_completion_tokens": 4096},
        "supported_parameters": ["max_tokens", "temperature", "seed"],
    }]})

    assert normalized == [{
        "id": "vendor/model",
        "name": "Model",
        "context_length": 32768,
        "max_completion_tokens": 4096,
        "supported_parameters": ["max_tokens", "seed", "temperature"],
        "supports_max_tokens_parameter": True,
        "supports_temperature_parameter": True,
        "supports_seed_parameter": True,
        "text_pricing_usd_per_token": {
            "input": "0.000001",
            "output": "0.000002",
            "request": "0.001",
        },
    }]


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def test_discover_writes_public_payload_hashes_and_selected_hf_metadata(
        monkeypatch, tmp_path):
    api_key = "test-secret-never-persist"
    monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    openrouter_body = json.dumps({"data": [{
        "id": "vendor/model",
        "name": "Model",
        "context_length": 1000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": ["max_tokens"],
    }]}).encode()
    hf_body = json.dumps({"modelId": "org/model", "sha": "abc123", "private": False,
                          "cardData": {"license": "apache-2.0"}}).encode()
    requests = []

    def open_public(request, timeout):
        requests.append((request, timeout))
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") is None
        if urlsplit(request.full_url).hostname == "openrouter.ai":
            return FakeResponse(openrouter_body)
        assert request.full_url == "https://huggingface.co/api/models/org/model"
        return FakeResponse(hf_body)

    monkeypatch.setattr(catalog.urllib.request, "urlopen", open_public)
    output = tmp_path / "catalog-run"

    result = catalog.discover_catalog(output, hf_models=["org/model"])

    saved = json.loads((output / "catalog.json").read_text())
    assert result["output_dir"] == str(output)
    assert saved["schema_version"] == 1
    assert saved["openrouter"]["response_sha256"] == hashlib.sha256(openrouter_body).hexdigest()
    assert saved["openrouter"]["catalog"] == json.loads(openrouter_body)
    assert saved["openrouter"]["models"][0]["max_completion_tokens"] is None
    assert saved["huggingface"][0]["repo_id"] == "org/model"
    assert saved["huggingface"][0]["response_sha256"] == hashlib.sha256(hf_body).hexdigest()
    assert saved["huggingface"][0]["metadata"] == json.loads(hf_body)
    assert saved["auth_status"] == {"available": True, "source_kind": "environment"}
    assert api_key not in (output / "catalog.json").read_text()
    assert len(requests) == 2
    assert all(0 < timeout <= 10 for _, timeout in requests)
    assert all(item["fetched_at_utc"].endswith("+00:00") for item in
               [saved["openrouter"], *saved["huggingface"]])


def test_discover_refuses_existing_directory_before_network_access(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def forbidden_network(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not run for an existing output directory")

    monkeypatch.setattr(catalog.urllib.request, "urlopen", forbidden_network)
    with pytest.raises(FileExistsError):
        catalog.discover_catalog(output)
    assert called is False


@pytest.mark.parametrize("repo", ["", "missing-slash", "org/a/b", "../model", "org/model?x=1"])
def test_discover_rejects_invalid_huggingface_repo_names(tmp_path, repo):
    with pytest.raises(ValueError, match="Hugging Face"):
        catalog.discover_catalog(tmp_path / "catalog", hf_models=[repo])


def test_cli_requires_new_output_directory_and_prints_safe_summary(monkeypatch, capsys, tmp_path):
    output = tmp_path / "catalog"
    monkeypatch.setattr(catalog, "discover_catalog", lambda *a, **k: {
        "output_dir": str(output), "openrouter_model_count": 3, "hf_model_count": 1,
        "auth_status": {"available": True, "source_kind": "environment"},
    })

    code = catalog.main(["discover", "--output", str(output), "--hf-model", "org/model"])

    captured = capsys.readouterr().out
    printed = json.loads(captured)
    assert code == 0
    assert printed["auth_status"] == {"available": True, "source_kind": "environment"}
    assert "secret" not in captured.lower()
