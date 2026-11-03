import json
from pathlib import Path

import pytest

from ticklm_bench import report, runner


def _cases(split='heldout', answers=('A', 'B')):
    choices = {'A': 'ALARM', 'B': 'WAIT', 'C': 'STOP', 'D': 'RECHECK'}
    options = '\n'.join(f'{label}. {text}' for label, text in choices.items())
    return [
        {'id': f'episode-{split}-raw-{variant}', 'episode_id': f'episode-{split}',
         'pair_id': f'pair-{split}', 'family': 'duration', 'view': 'raw', 'variant': variant,
         'split': split, 'answer': answer, 'choices': choices.copy(),
         'prompt': f'{split} case {variant}\n{options}'}
        for variant, answer in enumerate(answers)
    ]


def _write_run(root, name, *, cases=None, track='local', model='weights-a', adapter_type='mlx',
               max_tokens=256, effort=None, outputs=None, scorer_hash=None, status='ok'):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    cases = _cases() if cases is None else cases
    case_path = run_dir / 'cases.jsonl'
    case_path.write_text(''.join(json.dumps(case, sort_keys=True) + '\n' for case in cases))
    if outputs is None:
        outputs = ['A', 'B']
    records = []
    for case, text in zip(cases, outputs, strict=True):
        records.append({'case_id': case['id'], 'case': case,
                        'result': {'status': status, 'text': text,
                                   'resolved_model': model if adapter_type == 'mlx' else None,
                                   'elapsed_s': 1.0, 'first_visible_s': 0.1,
                                   'usage': {'input_tokens': 10, 'output_tokens': 2}}})
    (run_dir / 'records.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
    current_scorer = runner.digest(Path(report.__file__).with_name('scoring.py'))
    manifest = {
        'run_id': name,
        'dataset_sha256': runner.digest(case_path),
        'expected_requests': len(cases),
        'record_count': len(records),
        'status': 'complete',
        'config': {'system_id': name, 'track': track, 'type': adapter_type,
                   'model': model, 'model_id': model, 'max_tokens': max_tokens,
                   'effort': effort, 'thinking': False, 'seed': 7},
        'adapter': {'type': adapter_type, 'provider': track,
                    'requested_model': model,
                    'capabilities': {'hard_token_cap': adapter_type == 'mlx',
                                     'sampling_seed': adapter_type == 'http',
                                     'thinking': False}},
        'source': {'source_sha256': {'scoring.py': scorer_hash or current_scorer}},
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    return run_dir


def _run_rows(report_json):
    return {row['directory']: row for row in report_json['runs']}


def test_same_request_count_from_different_datasets_is_not_comparable(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    _write_run(runs, 'split-a', cases=_cases('split-a'))
    _write_run(runs, 'split-b', cases=_cases('split-b'))

    output = tmp_path / 'report'
    report.build_report(runs, output)
    result = json.loads((output / 'results.json').read_text())
    rows = _run_rows(result)

    first = rows['split-a']['comparisons']['raw']
    second = rows['split-b']['comparisons']['raw']
    assert rows['split-a']['manifest']['expected_requests'] == rows['split-b']['manifest']['expected_requests']
    assert first['quality_group_id'] != second['quality_group_id']
    assert first['quality_comparable'] is False
    assert second['quality_comparable'] is False
    assert first['policy_group_id'] != second['policy_group_id']


def test_quality_comparison_uses_shared_complete_cases_across_tracks_and_weights(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    cases = _cases()
    _write_run(runs, 'local-base', cases=cases, model='base-weights')
    _write_run(runs, 'local-adapted', cases=cases, model='adapted-weights')
    _write_run(runs, 'native', cases=cases, track='native-client', model='gpt-client',
               adapter_type='codex', effort='xhigh')
    _write_run(runs, 'local-longer', cases=cases, model='longer-cap', max_tokens=512)

    output = tmp_path / 'report'
    report.build_report(runs, output)
    result = json.loads((output / 'results.json').read_text())
    rows = _run_rows(result)
    base = rows['local-base']['comparisons']['raw']
    adapted = rows['local-adapted']['comparisons']['raw']
    native = rows['native']['comparisons']['raw']
    longer = rows['local-longer']['comparisons']['raw']

    assert base['quality_comparable'] is True
    assert base['quality_group_id'] == adapted['quality_group_id'] == native['quality_group_id']
    assert base['policy_group_id'] == adapted['policy_group_id']
    assert native['policy_group_id'] != base['policy_group_id']
    assert longer['quality_group_id'] == base['quality_group_id']
    assert longer['policy_group_id'] != base['policy_group_id']
    assert rows['native']['resolved_model_display'].startswith('UNKNOWN')
    assert rows['local-base']['resolved_model_display'] == 'base-weights'
    markdown = (output / 'results.md').read_text()
    assert '| Policy group |' in markdown
    assert 'Thinking requested / supported' in markdown
    assert 'Streaming / supported' in markdown


def test_complete_option_line_gap_is_posthoc_and_strict_score_stays_invalid(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    choices = {'A': 'ALARM', 'B': 'WAIT', 'C': 'STOP', 'D': 'RECHECK'}
    options = '\n'.join(f'{label}. {text}' for label, text in choices.items())
    cases = [
        {'id': f'posthoc-{variant}', 'episode_id': 'posthoc-episode', 'pair_id': 'posthoc-pair',
         'family': 'duration', 'view': 'raw', 'variant': variant, 'answer': answer,
         'choices': choices.copy(), 'prompt': f'Choose.\n{options}'}
        for variant, answer in enumerate(('D', 'C'))
    ]
    _write_run(runs, 'qwen', cases=cases,
               outputs=['D. RECHECK', 'C. STOP'])

    output = tmp_path / 'report'
    report.build_report(runs, output)
    result = json.loads((output / 'results.json').read_text())
    row = result['runs'][0]
    strict = row['summary']['views']['raw']
    diagnostic = row['posthoc_option_line']['raw']

    assert strict['overall']['correct'] == 0
    assert strict['overall']['invalid'] == 2
    assert diagnostic == {'correct': 2, 'total': 2, 'invalid': 0}
    markdown = (output / 'results.md').read_text()
    assert 'Post-hoc exact-option diagnostic' in markdown
    assert 'not preregistered' in markdown.lower()
    assert '| qwen | raw | 2/2 | 0 |' in markdown


def test_scorer_source_mismatch_requires_explicit_rescore(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    _write_run(runs, 'old-scorer', scorer_hash='0' * 64)

    with pytest.raises(ValueError, match='scoring source hash'):
        report.build_report(runs, tmp_path / 'reject')

    report.build_report(runs, tmp_path / 'rescored', rescore=True)
    result = json.loads((tmp_path / 'rescored/results.json').read_text())
    assert result['rescore_mode'] is True
    assert result['runs'][0]['scoring_source']['rescored'] is True


def test_optional_live_report_separates_snapshot_staleness_and_guarded_reference(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    _write_run(runs, 'static', cases=_cases())
    live_root = tmp_path / 'live'
    live_root.mkdir()
    live_dir = live_root / 'trajectory-run'
    live_dir.mkdir()
    modes = ('oracle_guarded', 'freshness_only', 'unguarded')
    in_force = {mode: {'correct_s': 5.0, 'wrong_s': 2.0, 'undecided_s': 3.0,
                       'accuracy_in_force': .5, 'decided_fraction': .7} for mode in modes}
    trajectory = {'trajectory_id': 't0', 'observed_horizon_s': 10.0, 'actual_elapsed_s': 10.0,
                  'in_force': in_force,
                  'requests': [{'result': {'status': 'ok', 'text': 'A'}, 'prediction': 'A',
                                'snapshot_correct': True, 'completion_correct': False,
                                'same_state_at_publication': False,
                                'deadline_crossed_before_publication': True}]}
    trajectory_path = live_dir / 'trajectories.jsonl'
    trajectory_path.write_text(json.dumps(trajectory) + '\n')
    source_dir = Path(report.__file__).parent
    manifest = {'run_id': 'live-1', 'status': 'complete', 'track': 'local',
                'trajectory_count': 1, 'completed_trajectories': 1,
                'config': {'system_id': 'live-model', 'model': 'unresolved'},
                'adapter': {'type': 'codex', 'capabilities': {'hard_token_cap': False}},
                'source_sha256': {'scoring.py': runner.digest(source_dir / 'scoring.py'),
                                  'live.py': runner.digest(source_dir / 'live.py')},
                'artifact_sha256': {'trajectories.jsonl': runner.digest(trajectory_path)}}
    (live_dir / 'manifest.json').write_text(json.dumps(manifest))

    output = tmp_path / 'report'
    report.build_report(runs, output, live_root=live_root)
    result = json.loads((output / 'results.json').read_text())
    live = result['live_runs'][0]

    assert live['snapshot']['correct'] == 1
    assert live['snapshot']['valid'] == 1
    assert live['stale_before_publication']['count'] == 1
    assert live['in_force']['oracle_guarded']['correct_s'] == 5.0
    markdown = (output / 'results.md').read_text()
    assert 'Oracle-guarded reference (not deployable)' in markdown
    assert 'freshness-only guard does not check semantic correctness' in markdown.lower()
    assert 'Trajectories completed / expected; requests' in markdown
    assert '1/1; 1' in markdown


def test_live_report_rejects_missing_in_force_mode_instead_of_reporting_zero(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    _write_run(runs, 'static', cases=_cases())
    live_root = tmp_path / 'live'
    live_root.mkdir()
    live_dir = live_root / 'partial-live'
    live_dir.mkdir()
    trajectory_path = live_dir / 'trajectories.jsonl'
    trajectory_path.write_text(json.dumps({
        'observed_horizon_s': 10.0,
        'requests': [],
        'in_force': {'unguarded': {'correct_s': 0, 'wrong_s': 0, 'undecided_s': 10}},
    }) + '\n')
    source_dir = Path(report.__file__).parent
    manifest = {
        'trajectory_count': 1,
        'adapter': {'type': 'codex'},
        'source_sha256': {'scoring.py': runner.digest(source_dir / 'scoring.py'),
                          'live.py': runner.digest(source_dir / 'live.py')},
        'artifact_sha256': {'trajectories.jsonl': runner.digest(trajectory_path)},
    }
    (live_dir / 'manifest.json').write_text(json.dumps(manifest))

    with pytest.raises(TypeError, match='missing in-force policy data'):
        report.build_report(runs, tmp_path / 'report', live_root=live_root)
