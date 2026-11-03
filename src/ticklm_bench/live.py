"""Live temporal decision benchmark driven by an external monotonic clock.

Each trajectory has a fixed exogenous event schedule. ALARM is correct when the
latest event is ALARM or the deadline has been reached; otherwise WAIT is correct.
The freshness-only guard publishes any parseable answer whose state version is
current at publication; it never checks correctness. The oracle-guarded track
also requires ground-truth correctness at dispatch and is an oracle-assisted
reference, not a deployable semantic guard. Both guarded tracks clear output at
every state change. Unguarded output remains in force until replaced. A snapshot
already past the deadline has no extra time-to-live; it is valid until another
state change or the trajectory ends. The externally supplied current time is
elapsed time from trajectory start; the model compares it with a supplied
deadline and does not measure elapsed time itself.

In-force accuracy is correct seconds divided by observed trajectory seconds;
initial or invalidated undecided time is reported separately, not counted correct.
This simulation does not provide an atomic external publication guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
import uuid
from pathlib import Path

from .adapters import create_adapter
from .budget import BudgetLedger, Pricing, estimate_max_cost
from .catalog import load_openrouter_key
from .scoring import parse_answer

_ACTIONS = {'A': 'ALARM', 'B': 'WAIT', 'C': 'STOP', 'D': 'RECHECK'}
_SCHEDULES = (
    (0.65, ((0.2, 'ALARM'), (0.4, 'CLEAR'), (0.8, 'ALARM'), (0.9, 'CLEAR'))),
    (0.55, ((0.15, 'ALARM'), (0.35, 'CLEAR'), (0.7, 'ALARM'), (0.9, 'CLEAR'))),
    (0.65, ((0.1, 'ALARM'), (0.5, 'CLEAR'), (0.75, 'ALARM'), (0.85, 'CLEAR'))),
)


def make_schedule(duration, trajectory_index=0):
    """Return an exogenous schedule in seconds relative to a trajectory's start."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive')
    deadline, events = _SCHEDULES[trajectory_index % len(_SCHEDULES)]
    return {'duration_s': duration, 'deadline_s': duration * deadline,
            'events': [{'at_s': duration * fraction, 'event': event, 'event_version': index + 1}
                       for index, (fraction, event) in enumerate(events)]}


def snapshot(schedule, elapsed):
    """Derive state from the complete schedule at the supplied monotonic time."""
    events = [event for event in schedule['events'] if event['at_s'] <= elapsed]
    event = events[-1]['event'] if events else 'CLEAR'
    event_version = len(events)
    expired = elapsed >= schedule['deadline_s']
    return {'elapsed_s': elapsed, 'deadline_s': schedule['deadline_s'], 'latest_event': event,
            'event_version': event_version, 'state_version': event_version + int(expired),
            'deadline_reached': expired,
            'action': 'ALARM' if event == 'ALARM' or expired else 'WAIT'}


def _truth_intervals(schedule, horizon):
    changes = sorted({0.0, horizon, *[event['at_s'] for event in schedule['events']
                                      if 0 < event['at_s'] < horizon],
                      *([schedule['deadline_s']] if 0 < schedule['deadline_s'] < horizon else [])})
    return [{'start_s': start, 'end_s': end, 'snapshot': snapshot(schedule, start)}
            for start, end in itertools.pairwise(changes)]


def integrate_in_force(duration, schedule, publications):
    """Integrate oracle, freshness-only, and unguarded actions over real state changes."""
    if duration < 0 or duration > schedule['duration_s']:
        raise ValueError('integration duration outside trajectory')
    changes = {0.0, duration}
    changes.update(event['at_s'] for event in schedule['events'] if 0 < event['at_s'] < duration)
    if 0 < schedule['deadline_s'] < duration:
        changes.add(schedule['deadline_s'])
    changes.update(pub['at'] for pub in publications if 0 <= pub['at'] < duration)
    points = sorted(changes)
    totals = {mode: {'correct_s': 0.0, 'wrong_s': 0.0, 'undecided_s': 0.0}
              for mode in ('oracle_guarded', 'freshness_only', 'unguarded')}
    active = {mode: None for mode in totals}
    guard_version = {'oracle_guarded': None, 'freshness_only': None}
    by_time = {}
    for publication in publications:
        by_time.setdefault(publication['at'], []).append(publication)
    for start, end in itertools.pairwise(points):
        current = snapshot(schedule, start)
        for mode, version in guard_version.items():
            if version != current['state_version']:
                active[mode] = None
                guard_version[mode] = None
        for publication in by_time.get(start, []):
            if publication['action'] is not None:
                active['unguarded'] = publication['action']
                for mode, eligibility in (('oracle_guarded', 'oracle_eligible'),
                                          ('freshness_only', 'freshness_eligible')):
                    if publication[eligibility]:
                        active[mode] = publication['action']
                        guard_version[mode] = current['state_version']
        for mode, total in totals.items():
            value = active[mode]
            category = ('undecided_s' if value is None else
                        'correct_s' if value == current['action'] else 'wrong_s')
            total[category] += end - start
    for result in totals.values():
        result['accuracy_in_force'] = result['correct_s'] / duration if duration else None
        result['decided_fraction'] = 1 - result['undecided_s'] / duration if duration else None
    return totals


def _message(state):
    return [{'role': 'user', 'content': (
        'Use the stated snapshot and policy. Choose one letter only. '
        'The action is ALARM if the latest event is ALARM or now is at or after the deadline; '
        'otherwise the action is WAIT. All times are seconds on the same external clock. '
        f"Now: {state['elapsed_s']}; deadline: {state['deadline_s']}; "
        f"latest event: {state['latest_event']}; event version: {state['event_version']}.\n"
        'A. ALARM\nB. WAIT\nC. STOP\nD. RECHECK')}]



def _request_metrics(requests):
    attempts = len(requests)
    valid = [row for row in requests if row['prediction'] is not None]
    def fraction(key, rows):
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else None
    def quantile(values, q):
        if not values:
            return None
        values = sorted(values)
        position = (len(values) - 1) * q
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {'attempts': attempts, 'valid_answers': len(valid),
            'invalid_or_failed': attempts - len(valid),
            'snapshot_accuracy_on_valid': fraction('snapshot_correct', valid),
            'completion_accuracy_on_valid': fraction('completion_correct', valid),
            'same_action_fraction': fraction('same_action', requests),
            'same_event_version_fraction': fraction('same_event_version', requests),
            'same_state_fraction': fraction('same_state_version', requests),
            'oracle_eligible_fraction': fraction('oracle_eligible', requests),
            'freshness_eligible_fraction': fraction('freshness_eligible', requests),
            'latency_p50_s': quantile([row['latency_s'] for row in requests], .5),
            'latency_p95_s': quantile([row['latency_s'] for row in requests], .95)}


def _summarize_trajectories(trajectories):
    horizon = sum(row['observed_horizon_s'] for row in trajectories)
    totals = {}
    for mode in ('oracle_guarded', 'freshness_only', 'unguarded'):
        row = {key: sum(trajectory['in_force'][mode][key] for trajectory in trajectories)
               for key in ('correct_s', 'wrong_s', 'undecided_s')}
        row['accuracy_in_force'] = row['correct_s'] / horizon if horizon else None
        row['decided_fraction'] = 1 - row['undecided_s'] / horizon if horizon else None
        totals[mode] = row
    return {'observed_trajectories': len(trajectories),
            'total_requests': sum(len(row['requests']) for row in trajectories),
            'actual_elapsed_s': sum(row['actual_elapsed_s'] for row in trajectories),
            'observed_horizon_s': horizon,
            'request_metrics': _request_metrics([request for row in trajectories for request in row['requests']]),
            **totals}

def run_trajectory(adapter, *, duration, trajectory_id, clock=None, trajectory_index=0,
                   max_tokens=32, timeout_s=30, before_call=None, after_call=None, record_call=None):
    """Run one trajectory using monotonic dispatch and completion timestamps."""
    clock = clock or time
    schedule = make_schedule(duration, trajectory_index)
    start = clock.monotonic()
    requests, publications = [], []
    stopped_reason = None
    while True:
        dispatch_abs = clock.monotonic()
        dispatch = dispatch_abs - start
        if dispatch >= duration:
            break
        if dispatch < 0:
            raise ValueError('monotonic clock moved backwards')
        state = snapshot(schedule, dispatch)
        messages = _message(state)
        trajectory_key = hashlib.sha256(trajectory_id.encode('utf-8')).hexdigest()
        request_id = f'live-{trajectory_key}-{len(requests):06d}'
        reservation = None
        if before_call is not None:
            try:
                reservation = before_call(request_id, messages)
            except RuntimeError as exc:
                stopped_reason = type(exc).__name__
                break
        if record_call is not None:
            record_call({'kind': 'dispatch', 'request_id': request_id, 'trajectory_id': trajectory_id,
                         'dispatch_monotonic_s': dispatch_abs, 'dispatch_s': dispatch,
                         'reserved_usd': str(reservation.max_usd) if reservation else None})
        if clock.monotonic() - start >= duration:
            accounting = (after_call(request_id, {'status': 'not_sent', 'usage': {'cost_usd': 0}}, reservation)
                          if after_call is not None and reservation is not None else None)
            if record_call is not None:
                record_call({'kind': 'not_sent', 'request_id': request_id,
                             'reason': 'trajectory_elapsed_before_send', 'accounting': accounting})
            stopped_reason = 'trajectory_elapsed_before_send'
            break
        try:
            result = adapter.generate(messages, max_tokens=max_tokens, timeout_s=timeout_s,
                                      request_id=request_id)
        except Exception as exc:  # noqa: BLE001 - settle reserved requests after unexpected adapter failures
            result = {'status': 'error', 'text': '', 'error': {'code': type(exc).__name__}}
        completion_abs = clock.monotonic()
        completion = completion_abs - start
        if completion < dispatch:
            raise ValueError('monotonic clock moved backwards')
        end_state = snapshot(schedule, completion)
        prediction = parse_answer(result.get('text')) if result.get('status') == 'ok' else None
        action = _ACTIONS.get(prediction)
        snapshot_correct = action == state['action'] if action is not None else False
        completion_correct = action == end_state['action'] if action is not None else False
        same_version = state['state_version'] == end_state['state_version']
        deadline_crossed = dispatch < schedule['deadline_s'] <= completion
        accounting = after_call(request_id, result, reservation) if after_call is not None else None
        publication_abs = clock.monotonic()
        publication = publication_abs - start
        if publication < completion:
            raise ValueError('monotonic clock moved backwards')
        publication_state = snapshot(schedule, publication)
        same_at_publication = state['state_version'] == publication_state['state_version']
        deadline_crossed_at_publication = dispatch < schedule['deadline_s'] <= publication
        freshness_eligible = bool(action is not None and same_at_publication
                                  and not deadline_crossed_at_publication and publication < duration)
        oracle_eligible = bool(freshness_eligible and snapshot_correct)
        record = {'request_id': request_id, 'messages': messages, 'result': result,
                  'dispatch_monotonic_s': dispatch_abs, 'completion_monotonic_s': completion_abs,
                  'dispatch_s': dispatch, 'completion_s': completion, 'latency_s': completion - dispatch,
                  'publication_s': publication, 'publication_monotonic_s': publication_abs,
                  'dispatch_snapshot': state, 'completion_snapshot': end_state,
                  'publication_snapshot': publication_state, 'same_state_at_publication': same_at_publication,
                  'deadline_crossed_before_publication': deadline_crossed_at_publication,
                  'prediction': prediction, 'predicted_action': action,
                  'snapshot_correct': snapshot_correct, 'completion_correct': completion_correct,
                  'same_action': state['action'] == end_state['action'],
                  'same_event_version': state['event_version'] == end_state['event_version'],
                  'same_state_version': same_version, 'deadline_crossed': deadline_crossed,
                  'freshness_eligible': freshness_eligible, 'oracle_eligible': oracle_eligible,
                  'accounting': accounting}
        requests.append(record)
        if record_call is not None:
            record_call({'kind': 'completion', **record})
        if action is not None:
            publications.append({'at': publication, 'action': action, 'freshness_eligible': freshness_eligible,
                                 'oracle_eligible': oracle_eligible})
        if accounting and accounting.get('blocked'):
            stopped_reason = 'accounting_overrun'
            break
        if len(requests) >= 3 and all(row['result'].get('status') != 'ok' for row in requests[-3:]):
            stopped_reason = 'three_consecutive_transport_failures'
            break
        next_time = dispatch_abs + 0.25
        remaining = min(next_time - clock.monotonic(), start + duration - clock.monotonic())
        if remaining > 0:
            clock.sleep(min(remaining, 60))
    actual = clock.monotonic() - start
    horizon = min(duration, max(0, actual))
    return {'trajectory_id': trajectory_id, 'schedule': schedule, 'start_monotonic_s': start,
            'actual_elapsed_s': actual, 'observed_horizon_s': horizon,
            'overshoot_s': max(0, actual - duration), 'stopping_reason': stopped_reason,
            'truth_intervals': _truth_intervals(schedule, horizon), 'requests': requests,
            'metrics': _request_metrics(requests), 'in_force': integrate_in_force(horizon, schedule, publications)}


def _write_json(path, value):
    target = Path(path)
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(target)


def _append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _safe_config(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace('_', '').replace('-', '')
            if normalized in {'apikey', 'authorization', 'password', 'accesstoken', 'secret', 'token', 'clientsecret'}:
                raise ValueError('credentials cannot appear in config')
            _safe_config(child)
    elif isinstance(value, list):
        for child in value:
            _safe_config(child)


def _local_model_hashes(config):
    model = config.get('model')
    if not isinstance(model, str) or not Path(model).is_dir():
        return None
    directory = Path(model)
    names = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.json', 'merges.txt', 'model.safetensors.index.json'}
    return {path.name: _hash(path) for path in sorted(directory.iterdir())
            if path.is_file() and (path.name in names or path.suffix == '.safetensors')}


def run_live(config_file, output, *, duration=30, trajectories=3, ledger_path=None, clock=None):
    """Persist trajectory requests and results; paid calls require a shared ledger."""
    clock = clock or time
    if not math.isfinite(duration) or duration <= 0 or trajectories <= 0:
        raise ValueError('duration and trajectories must be positive')
    config_file = Path(config_file)
    config = json.loads(config_file.read_text())
    _safe_config(config)
    if not config.get('system_id') or not config.get('track'):
        raise ValueError('system_id and track are required')
    if config.get('paid') and not ledger_path:
        raise ValueError('paid live requests require a shared budget ledger')
    if config.get('paid') and (config.get('type') != 'http' or not config.get('hard_token_cap')
                               or not config.get('billable_cap_confirmed')):
        raise ValueError('paid live requests require an enforced billable token cap')
    ledger = BudgetLedger(ledger_path, limit_usd=25) if config.get('paid') else None
    adapter_config = dict(config)
    if ledger:
        if config.get('credential_source') != 'openrouter':
            raise ValueError('unsupported credential source')
        adapter_config['api_key'] = load_openrouter_key()
        if not adapter_config['api_key']:
            raise ValueError('paid credential unavailable')
        pricing = Pricing(**config['pricing'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    max_tokens, timeout = config.get('max_tokens', 256), config.get('timeout_s', 120)
    run_id = str(uuid.uuid4())
    setup_start = clock.monotonic()
    source_dir = Path(__file__).parent
    manifest = {'run_id': run_id, 'config': config, 'config_sha256': _hash(config_file),
                'source_sha256': {name: _hash(source_dir / name)
                                  for name in ('live.py', 'adapters.py', 'scoring.py', 'budget.py', 'catalog.py')},
                'adapter': None, 'local_model_sha256': None,
                'trajectory_count': trajectories, 'duration_s': duration,
                'setup_s': None, 'status': 'initializing',
                'definition': ('External monotonic truth; freshness_only validates state version only; '
                               'oracle_guarded additionally uses ground-truth correctness and is not deployable.'),
                'time_input': ('Current elapsed time and deadline are supplied externally; '
                               'the model compares values and does not measure elapsed time.'),
                'snapshot_ttl': 'No extra TTL; a snapshot already past deadline stays valid until a state change.',
                'track': config['track']}
    _write_json(out / 'manifest.json', manifest)
    def before_call(request_id, messages):
        estimate = estimate_max_cost(len(json.dumps(messages, ensure_ascii=False).encode()), max_tokens,
                                     pricing, provider_max_tokens_confirmed=True)
        reservation = ledger.reserve(request_id, estimate.maximum_usd,
                                     {'system_id': config['system_id'], 'track': 'live'})
        if not reservation.can_execute:
            raise RuntimeError('request already reserved')
        return reservation
    def after_call(request_id, result, reservation):
        settlement = ledger.settle(request_id, result.get('usage', {}).get('cost_usd'))
        return {'reserved_usd': str(reservation.max_usd), 'charged_usd': str(settlement.charged_usd),
                'unknown': settlement.unknown, 'blocked': settlement.blocked}
    observed = []
    adapter = None
    try:
        manifest['local_model_sha256'] = _local_model_hashes(config)
        adapter = create_adapter(adapter_config)
        manifest['adapter'] = adapter.metadata
        manifest['setup_s'] = clock.monotonic() - setup_start
        manifest['status'] = 'running'
        _write_json(out / 'manifest.json', manifest)
        completed = 0
        for index in range(trajectories):
            result = run_trajectory(adapter, duration=duration, trajectory_id=f'{run_id}:{index:03d}',
                                    trajectory_index=index, clock=clock, max_tokens=max_tokens,
                                    timeout_s=timeout, before_call=before_call if ledger else None,
                                    after_call=after_call if ledger else None,
                                    record_call=lambda row: _append(out / 'requests.jsonl', row))
            _append(out / 'trajectories.jsonl', result)
            observed.append(result)
            _write_json(out / 'summary.json', _summarize_trajectories(observed))
            if result['stopping_reason']:
                manifest['stopping_reason'] = result['stopping_reason']
                break
            completed += 1
        manifest['completed_trajectories'] = completed
        manifest['status'] = 'complete' if completed == trajectories else 'incomplete'
    except Exception as exc:
        manifest['status'] = 'error'
        manifest['error_type'] = type(exc).__name__
        raise
    finally:
        if adapter is not None:
            adapter.close()
        for name in ('requests.jsonl', 'trajectories.jsonl', 'summary.json'):
            if (out / name).exists():
                manifest.setdefault('artifact_sha256', {})[name] = _hash(out / name)
        if ledger:
            manifest['budget'] = ledger.summary()
        _write_json(out / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--trajectories', type=int, default=3)
    parser.add_argument('--ledger')
    args = parser.parse_args(argv)
    run_live(args.config, args.output, duration=args.duration, trajectories=args.trajectories,
             ledger_path=args.ledger)


if __name__ == '__main__':
    main()
