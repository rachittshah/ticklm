import json
from typing import ClassVar

import pytest

from ticklm_bench import runner


class FakeAdapter:
    metadata: ClassVar[dict] = {"type": "fake"}
    calls: ClassVar[list] = []

    def generate(self, messages, **kwargs):
        self.calls.append(kwargs["request_id"])
        return {"status": "ok", "text": "A", "usage": {"cost_usd": None}}

    def close(self):
        pass


def test_freeze_screen_is_exact_subset_and_oracles_agree(tmp_path):
    manifest = runner.freeze(tmp_path / "data")
    full = runner.read_rows(tmp_path / "data/full.jsonl")
    screen = runner.read_rows(tmp_path / "data/screen.jsonl")
    assert len(full) == 448 and len(screen) == 112
    assert all(case in full for case in screen)
    assert manifest["tiers"]["screen"]["sha256"] == runner.digest(tmp_path / "data/screen.jsonl")
    with pytest.raises(FileExistsError):
        runner.freeze(tmp_path / "data")


def test_resume_preserves_attempt_identity_and_never_retries_uncertain_request(tmp_path, monkeypatch):
    runner.freeze(tmp_path / "data", pairs_per_family=1, screen_pairs=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"type": "mlx", "model": "fake", "system_id": "fake", "track": "test"}))
    monkeypatch.setattr(runner, "create_adapter", lambda _: FakeAdapter())
    FakeAdapter.calls = []
    out = tmp_path / "run"
    dataset = tmp_path / "data/screen.jsonl"
    first = runner.run(dataset, config, out)
    assert first["status"] == "complete" and len(FakeAdapter.calls) == 28
    # Simulate loss of a response after the durable request-start boundary.
    rows = (out / "records.jsonl").read_text().splitlines()
    (out / "records.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    resumed = runner.run(dataset, config, out, resume=True)
    assert len(FakeAdapter.calls) == 28
    assert resumed["status"] == "incomplete"
    summary = json.loads((out / "summary.json").read_text())
    assert not all(view["comparable"] for view in summary["views"].values())


def test_secrets_cannot_be_persisted_in_run_config(tmp_path):
    with pytest.raises(ValueError, match="credentials"):
        runner._config_check({"system_id": "x", "track": "test", "extra_body": {"api_key": "do-not-save"}})


def test_paid_uncapped_harness_rejected_before_adapter():
    with pytest.raises(ValueError, match="billable token cap"):
        runner._config_check({"system_id": "x", "track": "native", "type": "codex", "paid": True})
