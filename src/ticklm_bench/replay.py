"""Post-hoc, offline replay of live records with a narrow option-line parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from . import live, scoring

_OPTION = re.compile(r'([ABCD])\. (ALARM|WAIT|STOP|RECHECK)')
_MODES = ('oracle_guarded', 'freshness_only', 'unguarded')
_TIMES = ('correct_s', 'wrong_s', 'undecided_s', 'accuracy_in_force', 'decided_fraction')


def parse_live_answer(text):
    """Use the frozen parser or accept only an exact matching action-option line."""
    parsed = scoring.parse_answer(text)
    if parsed is not None or not isinstance(text, str):
        return parsed
    match = _OPTION.fullmatch(text.strip())
    if match and live._ACTIONS[match.group(1)] == match.group(2):
        return match.group(1)
    return None


def _same(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and math.isclose(left, right, abs_tol=1e-9, rel_tol=0)
    return left == right


def _replay(trajectory, parser, *, verify_original):
    schedule = trajectory['schedule']
    horizon = trajectory['observed_horizon_s']
    publications = []
    metrics = {'attempts': 0, 'valid': 0, 'invalid': 0, 'failed': 0,
               'invalid_or_failed': 0, 'snapshot_correct': 0, 'snapshot_wrong': 0,
               'completion_correct': 0, 'completion_wrong': 0,
               'freshness_eligible': 0, 'oracle_eligible': 0}
    details = []
    for row in trajectory['requests']:
        dispatch, completion, publication = (row[key] for key in ('dispatch_s', 'completion_s', 'publication_s'))
        if not (0 <= dispatch <= completion <= publication):
            raise ValueError('invalid recorded live timing')
        ds, cs, ps = (live.snapshot(schedule, t) for t in (dispatch, completion, publication))
        for key, derived in (('dispatch_snapshot', ds), ('completion_snapshot', cs), ('publication_snapshot', ps)):
            if any(not _same(derived[k], row[key].get(k)) for k in derived):
                raise ValueError(f'strict replay rejected inconsistent {key}')
        result = row['result']
        status = result.get('status')
        prediction = parser(result.get('text')) if status == 'ok' else None
        action = live._ACTIONS.get(prediction)
        snapshot_correct = bool(action is not None and action == ds['action'])
        completion_correct = bool(action is not None and action == cs['action'])
        fresh = bool(action is not None and ds['state_version'] == ps['state_version']
                     and not dispatch < schedule['deadline_s'] <= publication
                     and publication < schedule['duration_s'])
        oracle = bool(fresh and snapshot_correct)
        expected = {'prediction': prediction, 'predicted_action': action,
                    'snapshot_correct': snapshot_correct, 'completion_correct': completion_correct,
                    'freshness_eligible': fresh, 'oracle_eligible': oracle}
        if verify_original and any(not _same(row.get(key), value) for key, value in expected.items()):
            raise ValueError('strict replay disagrees with original request record')
        metrics['attempts'] += 1
        if status != 'ok':
            metrics['failed'] += 1
        elif prediction is None:
            metrics['invalid'] += 1
        else:
            metrics['valid'] += 1
            metrics['snapshot_correct' if snapshot_correct else 'snapshot_wrong'] += 1
            metrics['completion_correct' if completion_correct else 'completion_wrong'] += 1
        metrics['freshness_eligible'] += int(fresh)
        metrics['oracle_eligible'] += int(oracle)
        if action is not None:
            publications.append({'at': publication, 'action': action, 'freshness_eligible': fresh,
                                 'oracle_eligible': oracle})
        details.append({'request_id': row['request_id'], 'status': status, 'dispatch_s': dispatch,
                        'completion_s': completion, 'publication_s': publication, **expected})
    metrics['invalid_or_failed'] = metrics['invalid'] + metrics['failed']
    in_force = live.integrate_in_force(horizon, schedule, publications)
    if verify_original:
        for mode in _MODES:
            for key in _TIMES:
                if not _same(in_force[mode][key], trajectory['in_force'][mode][key]):
                    raise ValueError(f'strict replay disagrees with original in-force {mode}.{key}')
    return {'requests': metrics, 'in_force': in_force, 'decisions': details}


def replay_trajectory(trajectory):
    """Verify the original strict integral before computing the alternative."""
    strict = _replay(trajectory, scoring.parse_answer, verify_original=True)
    alternative = _replay(trajectory, parse_live_answer, verify_original=False)
    return {'trajectory_id': trajectory['trajectory_id'], 'observed_horizon_s': trajectory['observed_horizon_s'],
            'strict': strict, 'alternative': alternative}


def _aggregate(rows, arm):
    horizon = sum(row['observed_horizon_s'] for row in rows)
    request_keys = ('attempts', 'valid', 'invalid', 'failed', 'invalid_or_failed',
                    'snapshot_correct', 'snapshot_wrong', 'completion_correct', 'completion_wrong',
                    'freshness_eligible', 'oracle_eligible')
    counts = {key: sum(row[arm]['requests'][key] for row in rows) for key in request_keys}
    modes = {}
    for mode in _MODES:
        totals = {key: sum(row[arm]['in_force'][mode][key] for row in rows)
                  for key in ('correct_s', 'wrong_s', 'undecided_s')}
        totals['accuracy_in_force'] = totals['correct_s'] / horizon if horizon else None
        totals['decided_fraction'] = 1 - totals['undecided_s'] / horizon if horizon else None
        modes[mode] = totals
    return {'requests': counts, 'in_force': modes}


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_run(path):
    """Read only a complete run whose recorded artifact and execution hashes match."""
    path = Path(path)
    manifest_path, data_path, summary_path = path / 'manifest.json', path / 'trajectories.jsonl', path / 'summary.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'complete' or not manifest.get('trajectory_count') or
            manifest.get('completed_trajectories') != manifest.get('trajectory_count')):
        raise ValueError(f'live run is incomplete: {path}')
    if manifest.get('artifact_sha256', {}).get('trajectories.jsonl') != _hash(data_path):
        raise ValueError(f'live trajectory hash mismatch: {path}')
    if manifest.get('artifact_sha256', {}).get('summary.json') != _hash(summary_path):
        raise ValueError(f'live summary hash mismatch: {path}')
    for name, module in (('live.py', live), ('scoring.py', scoring)):
        if manifest.get('source_sha256', {}).get(name) != _hash(module.__file__):
            raise ValueError(f'live replay source hash mismatch: {name}')
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if len(rows) != manifest['trajectory_count']:
        raise ValueError('live trajectory count mismatch')
    replayed = [replay_trajectory(row) for row in rows]
    original_summary = json.loads(summary_path.read_text())
    strict_aggregate = _aggregate(replayed, 'strict')
    if not _same(original_summary['observed_horizon_s'], sum(r['observed_horizon_s'] for r in replayed)):
        raise ValueError('strict replay disagrees with original summary horizon')
    for mode in _MODES:
        for key in _TIMES:
            if not _same(original_summary[mode][key], strict_aggregate['in_force'][mode][key]):
                raise ValueError(f'strict replay disagrees with original summary {mode}.{key}')
    return {'system_id': manifest['config']['system_id'], 'run_id': manifest['run_id'],
            'input': {'manifest_sha256': _hash(manifest_path), 'trajectories_sha256': _hash(data_path),
                      'summary_sha256': _hash(summary_path),
                      'source_live_sha256': _hash(live.__file__), 'source_scoring_sha256': _hash(scoring.__file__)},
            'trajectory_count': len(rows), 'observed_horizon_s': sum(r['observed_horizon_s'] for r in replayed),
            'strict': strict_aggregate, 'alternative': _aggregate(replayed, 'alternative'),
            'trajectories': replayed}


def _write_atomic(path, text):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp-replay-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _markdown(payload):
    lines = ['# Post-hoc live parser replay', '', payload['scope'], '',
             '| System | Parser | Guard | Correct s | Wrong s | Undecided s | Invalid | Failed | Snapshot correct | Snapshot wrong | Valid |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in payload['runs']:
        for arm in ('strict', 'alternative'):
            counts = run[arm]['requests']
            for mode in _MODES:
                times = run[arm]['in_force'][mode]
                label = 'oracle-assisted reference' if mode == 'oracle_guarded' else mode
                lines.append(f"| {run['system_id']} | {arm} | {label} | {times['correct_s']:.3f} | "
                             f"{times['wrong_s']:.3f} | {times['undecided_s']:.3f} | {counts['invalid']} | "
                             f"{counts['failed']} | {counts['snapshot_correct']} | {counts['snapshot_wrong']} | {counts['valid']} |")
    lines.extend(['', ('The alternative accepts only the frozen parser or an exact matching option line. '
                  'Timing, dispatches, completions, state changes and publication timestamps remain recorded values.'),
                  '', 'Oracle-assisted decisions use ground truth and are a reference, not a deployable guard.', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Completed live-run directory; repeat per model')
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args(argv)
    runs = [load_run(path) for path in args.run]
    if len({run['run_id'] for run in runs}) != len(runs):
        raise ValueError('duplicate live run')
    payload = {'scope': ('Post-hoc after Qwen format failures: offline parser counterfactual at original recorded '
                         'publication times, assuming zero extra parser time. No new inference or live intervention.'),
               'strict_reproduction': 'Each original trajectory integral and request decision verified before alternative.',
               'alternative': 'Frozen parser or exact one-line A. ALARM, B. WAIT, C. STOP, or D. RECHECK.',
               'oracle_guarded': 'Ground-truth-assisted reference, not a deployable semantic guard.',
               'replay_source_sha256': _hash(__file__), 'runs': runs}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_atomic(output / 'replay.json', json.dumps(payload, indent=2, allow_nan=False) + '\n')
    _write_atomic(output / 'replay.md', _markdown(payload))


if __name__ == '__main__':
    main()
