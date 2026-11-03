import copy
import hashlib
import json

import pytest

from ticklm_bench import live, replay


def test_alternative_parser_only_accepts_exact_option_line_or_frozen_parser():
    assert replay.parse_live_answer('A') == 'A'
    assert replay.parse_live_answer('Answer: B') == 'B'
    assert replay.parse_live_answer('A. ALARM') == 'A'
    assert replay.parse_live_answer('B. WAIT') == 'B'
    assert replay.parse_live_answer('C. STOP') == 'C'
    assert replay.parse_live_answer('D. RECHECK') == 'D'
    for text in ('A. WAIT', 'A. ALARM\nexplanation', '**A. ALARM**', 'B.  WAIT',
                 'a. ALARM', 'A. ALARM and B. WAIT', 'A. alarm'):
        assert replay.parse_live_answer(text) is None


def make_trajectory(text, *, status='ok', dispatch=1.0, completion=1.0, publication=1.0):
    schedule = live.make_schedule(10)
    ds, cs, ps = (live.snapshot(schedule, t) for t in (dispatch, completion, publication))
    from ticklm_bench.scoring import parse_answer
    prediction = parse_answer(text) if status == 'ok' else None
    action = live._ACTIONS.get(prediction)
    fresh = bool(action is not None and ds['state_version'] == ps['state_version']
                 and not dispatch < schedule['deadline_s'] <= publication and publication < 10)
    oracle = fresh and action == ds['action']
    row = {'request_id': 'fixture', 'result': {'status': status, 'text': text},
           'dispatch_s': dispatch, 'completion_s': completion, 'publication_s': publication,
           'dispatch_snapshot': ds, 'completion_snapshot': cs, 'publication_snapshot': ps,
           'prediction': prediction, 'predicted_action': action,
           'snapshot_correct': action == ds['action'] if action else False,
           'completion_correct': action == cs['action'] if action else False,
           'freshness_eligible': fresh, 'oracle_eligible': oracle}
    publications = [{'at': publication, 'action': action, 'freshness_eligible': fresh,
                     'oracle_eligible': oracle}] if action else []
    return {'trajectory_id': 'fixture', 'schedule': schedule, 'requests': [row],
            'observed_horizon_s': 10.0, 'in_force': live.integrate_in_force(10, schedule, publications)}


def test_replay_preserves_time_and_separates_wrong_and_undecided():
    original = make_trajectory('A. ALARM')
    result = replay.replay_trajectory(original)
    assert result['strict']['in_force']['unguarded']['undecided_s'] == 10
    assert result['alternative']['in_force']['unguarded']['undecided_s'] == 1
    assert result['alternative']['in_force']['unguarded']['wrong_s'] == pytest.approx(3.5)
    assert result['alternative']['in_force']['freshness_only']['wrong_s'] == 1
    assert result['alternative']['requests']['snapshot_wrong'] == 1
    assert result['alternative']['requests']['invalid_or_failed'] == 0


def test_stale_and_failed_responses_do_not_become_guarded_publications():
    stale = replay.replay_trajectory(make_trajectory('A. ALARM', dispatch=1, completion=3, publication=3))
    assert stale['alternative']['requests']['freshness_eligible'] == 0
    assert stale['alternative']['in_force']['freshness_only']['undecided_s'] == 10
    assert stale['alternative']['in_force']['unguarded']['undecided_s'] == 3
    failed = replay.replay_trajectory(make_trajectory('A. ALARM', status='timeout'))
    assert failed['alternative']['requests']['invalid_or_failed'] == 1
    assert failed['alternative']['in_force']['unguarded']['undecided_s'] == 10


def test_strict_replay_rejects_tampered_original_metrics():
    trajectory = make_trajectory('B')
    trajectory['in_force']['unguarded']['correct_s'] += .5
    with pytest.raises(ValueError, match='strict replay'):
        replay.replay_trajectory(trajectory)


def test_strict_replay_rejects_changed_recorded_answer():
    trajectory = make_trajectory('B')
    altered = copy.deepcopy(trajectory)
    altered['requests'][0]['result']['text'] = 'A'
    with pytest.raises(ValueError, match='strict replay'):
        replay.replay_trajectory(altered)


def test_crossing_deadline_before_publication_rejects_freshness():
    result = replay.replay_trajectory(make_trajectory('A. ALARM', dispatch=6.4,
                                                       completion=6.45, publication=6.6))
    assert result['alternative']['requests']['freshness_eligible'] == 0
    assert result['alternative']['in_force']['freshness_only']['undecided_s'] == 10


def test_incomplete_or_mismatched_summary_is_refused(tmp_path):
    trajectory = make_trajectory('B')
    data = tmp_path / 'trajectories.jsonl'
    data.write_text(json.dumps(trajectory) + '\n')
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps({'observed_horizon_s': 10, **trajectory['in_force']}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'status': 'incomplete', 'trajectory_count': 1, 'completed_trajectories': 1,
                'run_id': 'test', 'config': {'system_id': 'test'},
                'source_sha256': {'live.py': digest(replay.Path(live.__file__)),
                                  'scoring.py': digest(replay.Path(replay.scoring.__file__))},
                'artifact_sha256': {'trajectories.jsonl': digest(data), 'summary.json': digest(summary)}}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        replay.load_run(tmp_path)
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest))
    assert replay.load_run(tmp_path)['strict']['requests']['valid'] == 1
    altered = json.loads(summary.read_text())
    altered['unguarded']['correct_s'] += 1
    summary.write_text(json.dumps(altered))
    manifest['artifact_sha256']['summary.json'] = digest(summary)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='summary'):
        replay.load_run(tmp_path)
