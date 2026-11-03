import json
import re

import pytest

from ticklm_bench import live


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        assert 0 <= seconds <= 60
        self.now += seconds


class FakeAdapter:
    def __init__(self, clock):
        self.metadata = {'type': 'fake', 'capabilities': {'hard_token_cap': False}}
        self.clock = clock
        self.messages = []
        self.closed = False
    def generate(self, messages, *, max_tokens, timeout_s, request_id):
        self.messages.append(messages)
        self.clock.now += 2.5
        letter = ('B', 'A', 'B', 'A')[(len(self.messages) - 1) % 4]
        return {'status': 'ok', 'text': letter, 'request_id': request_id}
    def close(self):
        self.closed = True


def test_in_force_integral_accounts_for_unguarded_staleness_and_guard_expiry():
    schedule = live.make_schedule(10)
    publications = [{'at': 1, 'action': 'WAIT', 'oracle_eligible': True, 'freshness_eligible': True},
                    {'at': 3, 'action': 'ALARM', 'oracle_eligible': True, 'freshness_eligible': True},
                    {'at': 5, 'action': 'WAIT', 'oracle_eligible': True, 'freshness_eligible': True},
                    {'at': 7, 'action': 'ALARM', 'oracle_eligible': True, 'freshness_eligible': True}]
    result = live.integrate_in_force(10, schedule, publications)
    assert result['unguarded']['correct_s'] == pytest.approx(6.5)
    assert result['unguarded']['wrong_s'] == pytest.approx(2.5)
    assert result['unguarded']['undecided_s'] == pytest.approx(1)
    assert result['oracle_guarded']['correct_s'] == pytest.approx(4.5)
    assert result['oracle_guarded']['wrong_s'] == 0
    assert result['oracle_guarded']['undecided_s'] == pytest.approx(5.5)


def test_real_clock_snapshots_separate_correctness_and_freshness():
    clock = FakeClock()
    adapter = FakeAdapter(clock)
    result = live.run_trajectory(adapter, duration=10, trajectory_id='example', clock=clock,
                                 max_tokens=4, timeout_s=10)
    first = result['requests'][0]
    assert first['dispatch_s'] == 0 and first['completion_s'] == 2.5
    assert first['snapshot_correct'] is True
    assert first['completion_correct'] is False
    assert first['same_action'] is False and first['same_state_version'] is False
    assert first['oracle_eligible'] is False
    assert first['dispatch_snapshot']['event_version'] == 0
    assert first['completion_snapshot']['event_version'] == 1
    third = result['requests'][2]
    assert third['deadline_crossed'] is True and third['oracle_eligible'] is False
    assert result['actual_elapsed_s'] == 10
    assert result['metrics']['attempts'] == 4
    assert result['metrics']['snapshot_accuracy_on_valid'] == 1
    assert result['metrics']['completion_accuracy_on_valid'] == .25
    assert result['metrics']['same_state_fraction'] == 0
    assert all(msg[0]['role'] == 'user' for msg in adapter.messages)
    assert all('ALARM' in msg[0]['content'] and 'WAIT' in msg[0]['content'] for msg in adapter.messages)
    assert len({r['request_id'] for r in result['requests']}) == len(result['requests'])
    assert all(re.fullmatch(r'[A-Za-z0-9._-]{1,128}', r['request_id']) for r in result['requests'])


def test_exact_version_distinguishes_same_action_after_a_change():
    schedule = live.make_schedule(10)
    before = live.snapshot(schedule, 7.5)
    after = live.snapshot(schedule, 8.5)
    assert before['action'] == after['action'] == 'ALARM'
    assert before['state_version'] != after['state_version']
    assert before['event_version'] != after['event_version']


def test_failed_or_unparseable_answer_never_gets_published():
    clock = FakeClock()
    class Failed(FakeAdapter):
        def generate(self, messages, *, max_tokens, timeout_s, request_id):
            self.clock.now += 5
            return {'status': 'timeout', 'text': 'A'}
    result = live.run_trajectory(Failed(clock), duration=10, trajectory_id='x', clock=clock,
                                 max_tokens=4, timeout_s=10)
    assert all(not request['oracle_eligible'] and request['prediction'] is None for request in result['requests'])
    assert result['in_force']['unguarded']['undecided_s'] == 10
    assert result['in_force']['oracle_guarded']['undecided_s'] == 10


def test_run_saves_three_trajectories_and_refuses_paid_without_ledger(tmp_path, monkeypatch):
    clock = FakeClock()
    adapter = FakeAdapter(clock)
    monkeypatch.setattr(live, 'create_adapter', lambda config: adapter)
    config = {'type': 'fake', 'model': 'dummy', 'system_id': 'sample', 'track': 'native', 'paid': False}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    out = tmp_path / 'out'
    live.run_live(config_path, out, duration=10, trajectories=3, clock=clock)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['trajectory_count'] == 3 and manifest['status'] == 'complete'
    assert manifest['adapter']['capabilities']['hard_token_cap'] is False
    records = [json.loads(line) for line in (out / 'trajectories.jsonl').read_text().splitlines()]
    assert len(records) == 3 and all(r['requests'] for r in records)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['observed_trajectories'] == 3
    assert summary['total_requests'] == 12
    assert 0 <= summary['oracle_guarded']['accuracy_in_force'] <= 1
    assert summary['actual_elapsed_s'] == 30
    assert adapter.closed
    config['paid'] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='ledger'):
        live.run_live(config_path, tmp_path / 'paid', duration=10, trajectories=3, clock=clock)
    assert not (tmp_path / 'paid').exists()


def test_publication_guard_rechecks_after_accounting_latency():
    clock = FakeClock()
    class Fast(FakeAdapter):
        def generate(self, messages, *, max_tokens, timeout_s, request_id):
            self.clock.now += 1.5
            return {'status': 'ok', 'text': 'B'}
    def accounting(request_id, result, reservation):
        clock.now += 1
        return {'blocked': False}
    result = live.run_trajectory(Fast(clock), duration=10, trajectory_id='x', clock=clock,
                                 max_tokens=4, timeout_s=10, after_call=accounting)
    first = result['requests'][0]
    assert first['snapshot_correct'] and first['completion_correct']
    assert first['same_state_version'] is True
    assert first['publication_s'] == 2.5
    assert first['same_state_at_publication'] is False
    assert first['oracle_eligible'] is False


def test_already_expired_snapshot_remains_eligible_until_next_change():
    clock = FakeClock()
    class Delays(FakeAdapter):
        def generate(self, messages, *, max_tokens, timeout_s, request_id):
            self.clock.now += (6.75 if self.clock.now == 0 else .5)
            return {'status': 'ok', 'text': 'A'}
    result = live.run_trajectory(Delays(clock), duration=10, trajectory_id='x', clock=clock,
                                 max_tokens=4, timeout_s=10)
    after_deadline = result['requests'][1]
    assert after_deadline['dispatch_snapshot']['deadline_reached']
    assert not after_deadline['deadline_crossed_before_publication']
    assert after_deadline['same_state_at_publication']
    assert after_deadline['oracle_eligible']


def test_run_records_error_manifest_and_closes_adapter(tmp_path, monkeypatch):
    clock = FakeClock()
    adapter = FakeAdapter(clock)
    monkeypatch.setattr(live, 'create_adapter', lambda config: adapter)
    def fail(*args, **kwargs):
        raise ValueError('synthetic error')
    monkeypatch.setattr(live, 'run_trajectory', fail)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'type': 'fake', 'model': 'dummy', 'system_id': 'x', 'track': 'native'}))
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='synthetic error'):
        live.run_live(path, out, duration=1, trajectories=1, clock=clock)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'error'
    assert manifest['error_type'] == 'ValueError'
    assert adapter.closed


def test_never_dispatch_after_trajectory_expired_during_reservation():
    clock = FakeClock()
    adapter = FakeAdapter(clock)
    def before(request_id, messages):
        clock.now += 11
    events = []
    result = live.run_trajectory(adapter, duration=10, trajectory_id='x', clock=clock,
                                 max_tokens=4, timeout_s=10, before_call=before,
                                 record_call=events.append)
    assert adapter.messages == []
    assert result['stopping_reason'] == 'trajectory_elapsed_before_send'
    assert events[-1]['kind'] == 'not_sent'


def test_paid_fake_requests_use_ledger_and_preserve_unknown_cost(tmp_path, monkeypatch):
    clock = FakeClock()
    adapter = FakeAdapter(clock)
    monkeypatch.setattr(live, 'load_openrouter_key', lambda: 'test-only-secret')
    def factory(config):
        assert config['api_key'] == 'test-only-secret'
        return adapter
    monkeypatch.setattr(live, 'create_adapter', factory)
    config = {'type': 'http', 'model': 'dummy', 'system_id': 'x', 'track': 'api',
              'paid': True, 'hard_token_cap': True, 'billable_cap_confirmed': True,
              'credential_source': 'openrouter', 'max_tokens': 4,
              'pricing': {'input_usd_per_token': '0.000001',
                          'output_usd_per_token': '0.000001', 'request_fee_usd': '0'}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    out = tmp_path / 'paid'
    live.run_live(path, out, duration=10, trajectories=1, ledger_path=tmp_path / 'ledger.db', clock=clock)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['budget']['request_counts']['unknown'] == 4
    assert float(manifest['budget']['unknown_usd']) > 0
    for artifact in ('manifest.json', 'requests.jsonl', 'trajectories.jsonl', 'summary.json'):
        assert 'test-only-secret' not in (out / artifact).read_text()


def test_wrong_but_fresh_output_is_published_only_by_freshness_guard():
    schedule = live.make_schedule(10)
    publications = [{'at': 1, 'action': 'ALARM', 'oracle_eligible': False,
                     'freshness_eligible': True}]
    result = live.integrate_in_force(10, schedule, publications)
    assert result['freshness_only']['wrong_s'] == pytest.approx(1)
    assert result['freshness_only']['undecided_s'] == pytest.approx(9)
    assert result['oracle_guarded']['undecided_s'] == 10
    assert result['oracle_guarded']['wrong_s'] == 0


def test_freshness_eligibility_does_not_use_oracle_answer():
    clock = FakeClock()
    class Wrong(FakeAdapter):
        def generate(self, messages, *, max_tokens, timeout_s, request_id):
            self.clock.now += .5
            return {'status': 'ok', 'text': 'A'}
    result = live.run_trajectory(Wrong(clock), duration=10, trajectory_id='x', clock=clock,
                                 max_tokens=4, timeout_s=10)
    first = result['requests'][0]
    assert first['snapshot_correct'] is False
    assert first['same_state_at_publication'] is True
    assert first['freshness_eligible'] is True
    assert first['oracle_eligible'] is False
    assert result['in_force']['freshness_only']['wrong_s'] > 0
