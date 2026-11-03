import json
import tarfile

import pytest

from ticklm_bench.export import export


def test_export_allowlist_excludes_private_state_and_weights(tmp_path):
    root = tmp_path / "raw"
    (root / "frozen").mkdir(parents=True)
    (root / "frozen/screen.jsonl").write_text('{"id":"one","prompt":"question"}\n')
    (root / "accidental-claude-help-profile").mkdir()
    (root / "accidental-claude-help-profile/private.json").write_text('{"api_key":"private"}')
    (root / "model.safetensors").write_bytes(b"not model evidence")
    result = export(root, tmp_path / "published")
    with tarfile.open(result["archive"]) as bundle:
        assert bundle.getnames() == ["frozen/screen.jsonl"]
    manifest = json.loads((tmp_path / "published/manifest.json").read_text())
    assert manifest["files"][0]["path"] == "frozen/screen.jsonl"


def test_export_rejects_secret_before_creating_publication(tmp_path):
    root = tmp_path / "raw"
    (root / "runs/one").mkdir(parents=True)
    (root / "runs/one/manifest.json").write_text('{"apiKey":"private"}')
    with pytest.raises(ValueError, match="credential-like"):
        export(root, tmp_path / "published")
    assert not (tmp_path / "published").exists()
