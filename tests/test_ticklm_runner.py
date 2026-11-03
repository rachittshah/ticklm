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


def test_failed_partial_answer_retained_but_never_scored(tmp_path, monkeypatch):
    class PartialAdapter(FakeAdapter):
        def generate(self, messages, **kwargs):
            return {"status": "timeout", "text": "A", "usage": {"cost_usd": None}}
    runner.freeze(tmp_path / "data", pairs_per_family=1, screen_pairs=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"type": "mlx", "model": "fake", "system_id": "fake", "track": "test"}))
    monkeypatch.setattr(runner, "create_adapter", lambda _: PartialAdapter())
    out = tmp_path / "run"
    result = runner.run(tmp_path / "data/screen.jsonl", config, out)
    assert result["stopping_reason"] == "three_consecutive_transport_failures"
    assert all(row["predicted"] is None for row in runner.read_rows(out / "records.jsonl"))


@pytest.mark.parametrize('key', ['apiKey', 'token', 'client_secret', 'access-token', 'Authorization'])
def test_credential_aliases_rejected_before_output(tmp_path, key):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'http', 'model': 'fake', 'system_id': 'x', 'track': 'test',
                                  'extra_body': {key: 'sensitive'}}))
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='credentials'):
        runner.run(tmp_path / 'unused.jsonl', config, out)
    assert not out.exists()


@pytest.mark.parametrize('endpoint', ['https://user:password@example.com/v1',
                                     'https://example.com/v1?api_key=sensitive',
                                     'https://example.com/v1#sensitive'])
def test_credential_urls_rejected_before_output(tmp_path, endpoint):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'http', 'model': 'fake', 'system_id': 'x', 'track': 'test',
                                  'endpoint': endpoint}))
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='endpoint'):
        runner.run(tmp_path / 'unused.jsonl', config, out)
    assert not out.exists()


def test_unregistered_dataset_requires_explicit_opt_in(tmp_path, monkeypatch):
    runner.freeze(tmp_path / 'frozen', pairs_per_family=1, screen_pairs=1)
    dataset = tmp_path / 'custom.jsonl'
    dataset.write_bytes((tmp_path / 'frozen/screen.jsonl').read_bytes())
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'mlx', 'model': 'fake', 'system_id': 'x', 'track': 'test'}))
    monkeypatch.setattr(runner, 'create_adapter', lambda _: FakeAdapter())
    with pytest.raises(ValueError, match='unregistered'):
        runner.run(dataset, config, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
    manifest = runner.run(dataset, config, tmp_path / 'allowed', allow_unregistered=True)
    assert manifest['dataset_registration']['registered'] is False


def test_registered_data_and_scorer_hashes_are_verified(tmp_path, monkeypatch):
    runner.freeze(tmp_path / 'frozen', pairs_per_family=1, screen_pairs=1)
    dataset = tmp_path / 'frozen/screen.jsonl'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'mlx', 'model': 'fake', 'system_id': 'x', 'track': 'test'}))
    monkeypatch.setattr(runner, 'create_adapter', lambda _: FakeAdapter())
    path = tmp_path / 'frozen/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['source']['source_sha256']['scoring.py'] = 'incorrect'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='scoring'):
        runner.run(dataset, config, tmp_path / 'bad-scorer')
    assert not (tmp_path / 'bad-scorer').exists()
    manifest['source']['source_sha256']['scoring.py'] = runner._source_state()['source_sha256']['scoring.py']
    manifest['tiers']['screen']['sha256'] = 'incorrect'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dataset'):
        runner.run(dataset, config, tmp_path / 'bad-data', allow_unregistered=True)
    assert not (tmp_path / 'bad-data').exists()


def test_resume_rejects_changed_loaded_runner_or_adapter(tmp_path, monkeypatch):
    runner.freeze(tmp_path / 'frozen', pairs_per_family=1, screen_pairs=1)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'mlx', 'model': 'fake', 'system_id': 'x', 'track': 'test'}))
    monkeypatch.setattr(runner, 'create_adapter', lambda _: FakeAdapter())
    out = tmp_path / 'out'
    runner.run(tmp_path / 'frozen/screen.jsonl', config, out)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source']['source_sha256']['adapters.py'] = 'changed'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source'):
        runner.run(tmp_path / 'frozen/screen.jsonl', config, out, resume=True)


def test_micro_subset_preserves_exact_pairs_and_registers_separate_manifest(tmp_path):
    parent_dir = tmp_path / 'parent'
    runner.freeze(parent_dir, pairs_per_family=4, screen_pairs=4)
    before = (parent_dir / 'manifest.json').read_bytes()
    parent = runner.read_rows(parent_dir / 'screen.jsonl')
    child_dir = tmp_path / 'micro'
    manifest = runner.subset(parent_dir / 'screen.jsonl', child_dir, pairs_per_family=1)
    micro = runner.read_rows(child_dir / 'micro.jsonl')
    expected = [case for case in parent if int(case['episode_id'].rsplit('-', 1)[1]) == 0]
    assert micro == expected
    assert len(micro) == 28 and len({case['episode_id'] for case in micro}) == 7
    assert manifest['tiers']['micro']['sha256'] == runner.digest(child_dir / 'micro.jsonl')
    assert manifest['parent_dataset_sha256'] == runner.digest(parent_dir / 'screen.jsonl')
    assert manifest['selection']['pairs_per_family'] == 1
    assert (parent_dir / 'manifest.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        runner.subset(parent_dir / 'screen.jsonl', child_dir, pairs_per_family=1)


def test_micro_subset_is_usable_as_registered_dataset(tmp_path, monkeypatch):
    runner.freeze(tmp_path / 'parent', pairs_per_family=1, screen_pairs=1)
    runner.subset(tmp_path / 'parent/screen.jsonl', tmp_path / 'micro', pairs_per_family=1)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'type': 'mlx', 'model': 'fake', 'system_id': 'x', 'track': 'test'}))
    monkeypatch.setattr(runner, 'create_adapter', lambda _: FakeAdapter())
    manifest = runner.run(tmp_path / 'micro/micro.jsonl', config, tmp_path / 'run')
    assert manifest['dataset_registration']['registered'] is True
    assert manifest['dataset_registration']['tier'] == 'micro'
    assert manifest['expected_requests'] == 28


def test_execution_refuses_source_changed_after_import(monkeypatch):
    source = runner._source_state()
    source['source_sha256'] = {**source['source_sha256'], 'runner.py': 'changed'}
    monkeypatch.setattr(runner, '_source_state', lambda: source)
    with pytest.raises(ValueError, match='changed since import'):
        runner._execution_source()
