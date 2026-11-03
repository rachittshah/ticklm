"""Build auditable static and live tables from retained benchmark records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .diagnostics import option_text_answer
from .runner import digest, now, read_rows, write_json
from .scoring import parse_answer, summarize

_ACTION_BY_LABEL = {'A': 'ALARM', 'B': 'WAIT', 'C': 'STOP', 'D': 'RECHECK'}
_LIVE_MODES = {
    'unguarded': 'Unguarded publication',
    'freshness_only': 'Freshness-only guard (state version only)',
    'oracle_guarded': 'Oracle-guarded reference (not deployable)',
}
_NATIVE_TYPES = {'claude_code', 'codex'}


def percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _manifest_file_hash(manifest, filename):
    candidates = (filename, f'ticklm_bench/{filename}', f'src/ticklm_bench/{filename}')
    containers = [manifest]
    for key in ('source', 'frozen_source'):
        if isinstance(manifest.get(key), dict):
            containers.append(manifest[key])
    for container in containers:
        hashes = container.get('source_sha256') if isinstance(container, dict) else None
        if isinstance(hashes, dict):
            for name in candidates:
                value = hashes.get(name)
                if isinstance(value, str):
                    return value
        direct = container.get(f'{filename[:-3]}_sha256') if isinstance(container, dict) else None
        if isinstance(direct, str):
            return direct
    return None


def _current_file_hash(filename):
    return digest(Path(__file__).with_name(filename))


def _resolved_model_label(row):
    resolved = row['resolved_models']
    if resolved:
        return ', '.join(resolved)
    adapter_type = row['adapter_type']
    config = row['manifest'].get('config', {})
    if adapter_type == 'mlx':
        requested = config.get('model')
        return f'LOCAL CHECKPOINT ({requested})' if requested else 'LOCAL CHECKPOINT (path unavailable)'
    return 'UNKNOWN (adapter returned no resolved model)'


def _requested_generation_policy(manifest):
    config = manifest.get('config', {})
    adapter = manifest.get('adapter', {})
    adapter_type = adapter.get('type') if isinstance(adapter, dict) else None
    adapter_type = adapter_type or config.get('type') or 'UNKNOWN'
    capabilities = adapter.get('capabilities', {}) if isinstance(adapter, dict) else {}
    cap = capabilities.get('hard_token_cap') if isinstance(capabilities, dict) else None
    is_native = adapter_type in _NATIVE_TYPES
    extra_body = config.get('extra_body') if adapter_type == 'http' else None
    if extra_body is not None:
        extra_body_hash = _canonical_hash(extra_body)
        extra_body_fields = sorted(extra_body.keys()) if isinstance(extra_body, dict) else []
    else:
        extra_body_hash, extra_body_fields = None, []
    return {
        'adapter_type': adapter_type,
        'max_tokens': config.get('max_tokens', 256),
        'hard_token_cap': cap,
        'effort': config.get('effort', 'high' if is_native else None) if is_native else None,
        'thinking': config.get('thinking'),
        'temperature': 0.0 if adapter_type in ('mlx', 'http') else config.get('temperature'),
        'seed': config.get('seed') if adapter_type == 'http' else None,
        'extra_body_sha256': extra_body_hash,
        'extra_body_fields': extra_body_fields,
    }


def _safe_text(value):
    if value is None:
        return '—'
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _diagnostic_scores(records, cases):
    by_id = {row['case']['id']: row for row in records}
    views = defaultdict(lambda: {'correct': 0, 'total': 0, 'invalid': 0})
    for case in cases:
        record = by_id.get(case['id'])
        result = record.get('result', {}) if isinstance(record, dict) else {}
        if result.get('status') != 'ok':
            continue
        view = views[case['view']]
        view['total'] += 1
        prediction = parse_answer(result.get('text', ''))
        if prediction is None:
            prediction = option_text_answer(case, result.get('text', ''))
        if prediction is None:
            view['invalid'] += 1
        elif prediction == case['answer']:
            view['correct'] += 1
    return dict(views)


def _load_static_runs(runs_root, current_scorer_hash, *, rescore):
    root = Path(runs_root)
    manifest_paths = ([root / 'manifest.json'] if (root / 'manifest.json').is_file()
                      else sorted(root.glob('*/manifest.json')))
    runs = []
    for manifest_path in manifest_paths:
        run_dir = manifest_path.parent
        record_path, case_path = run_dir / 'records.jsonl', run_dir / 'cases.jsonl'
        if not record_path.is_file():
            continue
        if not case_path.is_file():
            raise ValueError(f'{run_dir.name}: records exist without frozen cases.jsonl')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        records, cases = read_rows(record_path), read_rows(case_path)
        dataset_sha = digest(case_path)
        recorded_dataset_sha = manifest.get('dataset_sha256')
        if recorded_dataset_sha is not None and recorded_dataset_sha != dataset_sha:
            raise ValueError(f'{run_dir.name}: manifest dataset SHA does not match cases.jsonl')
        expected = manifest.get('expected_requests')
        if expected is not None and expected != len(cases):
            raise ValueError(f'{run_dir.name}: expected request count does not match frozen cases')

        recorded_scorer_hash = _manifest_file_hash(manifest, 'scoring.py')
        scorer_mismatch = recorded_scorer_hash != current_scorer_hash
        if scorer_mismatch and not rescore:
            raise ValueError(f'{run_dir.name}: scoring source hash is missing or differs; rerun with --rescore')
        normalized = []
        for row in records:
            result = row.get('result') if isinstance(row.get('result'), dict) else {}
            status = result.get('status')
            normalized.append({**row, 'predicted': parse_answer(result.get('text', ''))
                               if status == 'ok' else None})
        summary = summarize(normalized, expected_cases=cases)
        elapsed = [row['result'].get('elapsed_s') for row in records
                   if isinstance(row.get('result'), dict)
                   and isinstance(row['result'].get('elapsed_s'), (int, float))
                   and not isinstance(row['result'].get('elapsed_s'), bool)]
        token_totals = {}
        for field in ('input_tokens', 'output_tokens'):
            values = [(row.get('result', {}).get('usage', {}).get(field)) for row in records]
            token_totals[field] = {
                'observed_total': sum(value for value in values
                                      if isinstance(value, (int, float)) and not isinstance(value, bool)),
                'unknown_requests': sum(value is None for value in values),
            }
        adapter = manifest.get('adapter', {})
        adapter_type = adapter.get('type') if isinstance(adapter, dict) else None
        adapter_type = adapter_type or manifest.get('config', {}).get('type') or 'UNKNOWN'
        resolved = sorted({row.get('result', {}).get('resolved_model') for row in records
                           if isinstance(row.get('result'), dict)
                           and isinstance(row['result'].get('resolved_model'), str)
                           and row['result']['resolved_model']})
        model_row = {'resolved_models': resolved, 'adapter_type': adapter_type, 'manifest': manifest}
        policy = _requested_generation_policy(manifest)
        runs.append({
            'directory': run_dir.name,
            'manifest': manifest,
            'dataset_sha256': dataset_sha,
            'expected_requests': len(cases),
            'record_count': len(records),
            'summary': summary,
            'posthoc_option_line': _diagnostic_scores(records, cases),
            'records_sha256': digest(record_path),
            'scoring_source': {'recorded_sha256': recorded_scorer_hash,
                               'current_sha256': current_scorer_hash,
                               'rescored': scorer_mismatch},
            'adapter_type': adapter_type,
            'adapter_capabilities': adapter.get('capabilities', {}) if isinstance(adapter, dict) else {},
            'requested_generation_policy': policy,
            'requested_generation_policy_sha256': _canonical_hash(policy),
            'resolved_models': resolved,
            'resolved_model_display': _resolved_model_label(model_row),
            'latency': {'p50_s': float(np.median(elapsed)) if elapsed else None,
                        'p95_s': float(np.percentile(elapsed, 95)) if elapsed else None,
                        'first_request_s': elapsed[0] if elapsed else None,
                        'wall_inference_s': sum(elapsed)},
            'tokens': token_totals,
        })
    return runs


def _add_comparison_groups(runs):
    quality_members = defaultdict(list)
    policy_members = defaultdict(list)
    identities = {}
    for run_index, run in enumerate(runs):
        config = run['manifest'].get('config', {})
        track = config.get('track', 'UNKNOWN')
        for view, score in run['summary']['views'].items():
            quality_identity = {
                'dataset_sha256': run['dataset_sha256'],
                'case_set_sha256': score.get('case_set_sha256'),
                'scoring_sha256': run['scoring_source']['current_sha256'],
                'view': view,
            }
            quality_id = _canonical_hash(quality_identity)[:16]
            policy_identity = {
                **quality_identity,
                'track': track,
                'expected_requests': run['expected_requests'],
                'adapter_type': run['adapter_type'],
                'generation_policy_sha256': run['requested_generation_policy_sha256'],
            }
            policy_id = _canonical_hash(policy_identity)[:16]
            coverage = score.get('coverage', {})
            eligible = coverage.get('complete') is True and score.get('comparable') is True
            key = (run_index, view)
            identities[key] = (quality_identity, policy_identity, quality_id, policy_id, eligible)
            quality_members[quality_id].append((key, eligible))
            policy_members[policy_id].append((key, eligible))
    for (run_index, view), (_, _, quality_id, policy_id, eligible) in identities.items():
        run = runs[run_index]
        quality_size = sum(member_eligible for _, member_eligible in quality_members[quality_id])
        policy_size = sum(member_eligible for _, member_eligible in policy_members[policy_id])
        run.setdefault('comparisons', {})[view] = {
            'quality_group_id': quality_id,
            'quality_group_size': quality_size,
            'quality_comparable': bool(eligible and quality_size >= 2),
            'policy_group_id': policy_id,
            'policy_group_size': policy_size,
            'policy_aligned': bool(eligible and policy_size >= 2),
        }


def _live_paths(root):
    path = Path(root)
    if (path / 'manifest.json').is_file():
        return [path / 'manifest.json']
    return sorted(path.glob('*/manifest.json'))


def _recompute_live_trajectory(trajectory):
    from .live import integrate_in_force

    schedule = trajectory['schedule']
    horizon = trajectory['observed_horizon_s']
    publications = []
    requests = []
    for original in trajectory['requests']:
        row = dict(original)
        result = row.get('result', {}) if isinstance(row.get('result'), dict) else {}
        prediction = parse_answer(result.get('text', '')) if result.get('status') == 'ok' else None
        action = _ACTION_BY_LABEL.get(prediction)
        dispatch = row['dispatch_snapshot']
        completion = row['completion_snapshot']
        publication = row['publication_snapshot']
        dispatch_s = row['dispatch_s']
        publication_s = row['publication_s']
        same_at_publication = dispatch['state_version'] == publication['state_version']
        deadline_crossed = dispatch_s < schedule['deadline_s'] <= publication_s
        snapshot_correct = action == dispatch['action'] if action is not None else False
        completion_correct = action == completion['action'] if action is not None else False
        freshness_eligible = bool(action is not None and same_at_publication
                                  and not deadline_crossed and publication_s < horizon)
        oracle_eligible = bool(freshness_eligible and snapshot_correct)
        row.update({'prediction': prediction, 'predicted_action': action,
                    'snapshot_correct': snapshot_correct, 'completion_correct': completion_correct,
                    'same_state_at_publication': same_at_publication,
                    'deadline_crossed_before_publication': deadline_crossed,
                    'freshness_eligible': freshness_eligible, 'oracle_eligible': oracle_eligible})
        requests.append(row)
        if action is not None:
            publications.append({'at': publication_s, 'action': action,
                                 'freshness_eligible': freshness_eligible,
                                 'oracle_eligible': oracle_eligible})
    return {'requests': requests, 'in_force': integrate_in_force(horizon, schedule, publications)}


def _validated_in_force(run_name, trajectory, in_force):
    horizon = trajectory.get('observed_horizon_s')
    trajectory_id = trajectory.get('trajectory_id', 'unknown')
    if (not isinstance(horizon, (int, float)) or isinstance(horizon, bool)
            or not np.isfinite(horizon) or horizon < 0):
        raise ValueError(f'{run_name}/{trajectory_id}: invalid observed horizon')
    if not isinstance(in_force, dict):
        raise TypeError(f'{run_name}/{trajectory_id}: missing in-force policy data')
    checked = {}
    for mode in _LIVE_MODES:
        values = in_force.get(mode)
        if not isinstance(values, dict):
            raise TypeError(f'{run_name}/{trajectory_id}: missing in-force policy data for {mode}')
        metrics = {}
        for metric in ('correct_s', 'wrong_s', 'undecided_s'):
            value = values.get(metric)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f'{run_name}/{trajectory_id}: invalid in-force {mode}.{metric}')
            metrics[metric] = float(value)
        partition = sum(metrics.values())
        if not np.isclose(partition, horizon, rtol=1e-6, atol=1e-6):
            raise ValueError(f'{run_name}/{trajectory_id}: in-force {mode} seconds do not sum to horizon')
        checked[mode] = metrics
    return float(horizon), checked


def _load_live_runs(live_root, current_scorer_hash, current_live_hash, *, rescore):
    if live_root is None:
        return []
    live_runs = []
    for manifest_path in _live_paths(live_root):
        run_dir = manifest_path.parent
        trajectory_path = run_dir / 'trajectories.jsonl'
        if not trajectory_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, path in (('trajectories.jsonl', trajectory_path),):
            expected_hash = manifest.get('artifact_sha256', {}).get(name)
            if expected_hash is not None and expected_hash != digest(path):
                raise ValueError(f'{run_dir.name}: {name} hash does not match manifest')
        recorded_scorer = _manifest_file_hash(manifest, 'scoring.py')
        recorded_live = _manifest_file_hash(manifest, 'live.py')
        mismatch = recorded_scorer != current_scorer_hash or recorded_live != current_live_hash
        if mismatch and not rescore:
            raise ValueError(f'{run_dir.name}: live scoring/source hash is missing or differs; rerun with --rescore')
        trajectories = read_rows(trajectory_path)
        total_horizon = 0.0
        mode_totals = {mode: {'correct_s': 0.0, 'wrong_s': 0.0, 'undecided_s': 0.0}
                       for mode in _LIVE_MODES}
        request_rows = []
        for trajectory in trajectories:
            scored = _recompute_live_trajectory(trajectory) if mismatch else None
            if scored is not None:
                requests = scored['requests']
                in_force = scored['in_force']
            else:
                requests = trajectory.get('requests', [])
                in_force = trajectory.get('in_force', {})
            horizon, in_force = _validated_in_force(run_dir.name, trajectory, in_force)
            total_horizon += horizon
            for request in requests:
                result = request.get('result', {}) if isinstance(request.get('result'), dict) else {}
                valid = result.get('status') == 'ok' and parse_answer(result.get('text', '')) is not None
                if not valid:
                    continue
                same = request.get('same_state_at_publication')
                if same is None and request.get('dispatch_snapshot') and request.get('publication_snapshot'):
                    same = (request['dispatch_snapshot'].get('state_version') ==
                            request['publication_snapshot'].get('state_version'))
                crossed = request.get('deadline_crossed_before_publication')
                request_rows.append({'snapshot_correct': bool(request.get('snapshot_correct')),
                                     'completion_correct': bool(request.get('completion_correct')),
                                     'stale': same is False or crossed is True})
            for mode in _LIVE_MODES:
                for metric in mode_totals[mode]:
                    if metric in in_force[mode]:
                        mode_totals[mode][metric] += in_force[mode][metric]
        valid_count = len(request_rows)
        snapshot_correct = sum(row['snapshot_correct'] for row in request_rows)
        stale_count = sum(row['stale'] for row in request_rows)
        completion_correct = sum(row['completion_correct'] for row in request_rows)
        for values in mode_totals.values():
            values['accuracy_in_force'] = values['correct_s'] / total_horizon if total_horizon else None
            values['decided_fraction'] = ((total_horizon - values['undecided_s']) / total_horizon
                                          if total_horizon else None)
        adapter = manifest.get('adapter', {})
        adapter_type = adapter.get('type') if isinstance(adapter, dict) else None
        config = manifest.get('config', {})
        resolved = sorted({request.get('result', {}).get('resolved_model')
                           for trajectory in trajectories
                           for request in trajectory.get('requests', [])
                           if isinstance(request.get('result'), dict)
                           and isinstance(request['result'].get('resolved_model'), str)
                           and request['result']['resolved_model']})
        identity = ', '.join(resolved) if resolved else (
            f'LOCAL CHECKPOINT ({config.get("model")})' if adapter_type == 'mlx'
            else 'UNKNOWN (adapter returned no resolved model)')
        live_runs.append({
            'directory': run_dir.name,
            'status': manifest.get('status', 'UNKNOWN'),
            'track': manifest.get('track', config.get('track', 'UNKNOWN')),
            'adapter_type': adapter_type or 'UNKNOWN',
            'resolved_models': resolved,
            'resolved_model_display': identity,
            'expected_trajectories': manifest.get('trajectory_count'),
            'completed_trajectories': len(trajectories),
            'request_count': sum(len(trajectory.get('requests', [])) for trajectory in trajectories),
            'observed_horizon_s': total_horizon,
            'snapshot': {'correct': snapshot_correct, 'valid': valid_count,
                         'accuracy': snapshot_correct / valid_count if valid_count else None,
                         'completion_correct': completion_correct},
            'stale_before_publication': {'count': stale_count, 'valid': valid_count,
                                         'rate': stale_count / valid_count if valid_count else None},
            'in_force': mode_totals,
            'source_hashes': {'scoring_recorded': recorded_scorer,
                              'scoring_current': current_scorer_hash,
                              'live_recorded': recorded_live,
                              'live_current': current_live_hash,
                              'rescored': mismatch},
        })
    return live_runs


def _macro_text(score):
    point = score.get('macro_accuracy')
    if point is None:
        return '—'
    interval = (score.get('bootstrap_95') or {}).get('macro_accuracy')
    return f'{percent(point)} [{percent(interval[0])}, {percent(interval[1])}]' if interval else percent(point)


def _build_markdown(runs, live_runs, output):
    lines = [
        '# TickLM-Bench v0.1 — measured results', '',
        f'Generated {now()} directly from retained request records.', '',
        ('This is an exploratory synthetic benchmark. It measures supplied temporal facts and rules, '
         'not an internal clock or a general state-of-the-art ranking. Raw and arithmetic-assisted '
         'inputs, sample sizes, and native clients appear separately.'), '',
        ('Quality comparisons require the same case-file SHA, per-view scorer case-set SHA, current '
         'strict scoring code, and complete, structurally valid coverage. Track and harness are shown '
         'but do not by themselves prevent a shared-data quality comparison. Policy-aligned groups '
         'add track, request count, adapter type, and requested generation settings; model weights are '
         'not part of that identity.'), '',
        ('Macro accuracy is shown beside its family-stratified bootstrap interval; micro accuracy is '
         'separate. Degenerate intervals such as [0%, 0%] for all-invalid outputs or [100%, 100%] '
         'for all-correct outputs do not establish zero reasoning ability or broad generalization. '
         'No option-order variants were run, so option-order invariance is not measured.'), '',
    ]
    quality_groups = defaultdict(list)
    for run in runs:
        for view, score in run['summary']['views'].items():
            comparison = run['comparisons'][view]
            quality_groups[(comparison['quality_group_id'], view)].append((run, score, comparison))
    for (group_id, view), members in sorted(quality_groups.items()):
        first = members[0][0]
        score = members[0][1]
        lines.extend([f'## Shared frozen-data quality group {group_id} — {view}', '',
                      (f"Dataset SHA `{first['dataset_sha256']}`; case-set SHA "
                       f"`{score.get('case_set_sha256')}`; scorer SHA "
                       f"`{first['scoring_source']['current_sha256']}`. "
                       f"{sum(item[2]['quality_comparable'] for item in members)} run(s) are complete "
                       'members of this exact quality group.'), '',
                      '| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |',
                      '|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|'])
        for run, view_score, comparison in members:
            config = run['manifest'].get('config', {})
            system = _safe_text(config.get('system_id', run['directory']))
            track = _safe_text(config.get('track', 'UNKNOWN'))
            coverage = view_score['coverage']
            overall, pairs = view_score['overall'], view_score['pairs']
            lines.append(
                f"| {system} | {track} / {run['adapter_type']} | "
                f"{_safe_text(_resolved_model_label(run))} | "
                f"{coverage.get('expected', '—')} / {coverage.get('completed', 0)} | "
                f"{coverage.get('missing', 0)} | {coverage.get('failed', 0)} | "
                f"{overall['correct']}/{overall['total']} | {_macro_text(view_score)} | "
                f"{percent(overall.get('accuracy'))} | {overall.get('invalid', 0)} | "
                f"{pairs.get('both_correct', '?')}/{pairs.get('complete', 0)} | "
                f"`{comparison['quality_group_id']}` | `{comparison['policy_group_id']}` | "
                f"{'yes' if comparison['quality_comparable'] else 'no'} | "
                f"{'yes' if comparison['policy_aligned'] else 'no'} |"
            )
        lines.append('')

    lines.extend(['## Post-hoc exact-option diagnostic', '',
                  ('Added after the first Qwen 0.6B run exposed complete-option answers; it was not '
                   'preregistered. Applied uniformly to every system, this diagnostic accepts the '
                   'frozen strict parser or a response consisting entirely of one exact option line '
                   'rendered in that case prompt. It never extracts a letter from prose and does not '
                   'replace the primary strict score.'), '',
                  '| Run | View | Correct / completed | Invalid |', '|---|---|---:|---:|'])
    for run in runs:
        for view, metric in sorted(run['posthoc_option_line'].items()):
            lines.append(f"| {run['directory']} | {view} | {metric['correct']}/{metric['total']} | {metric['invalid']} |")
    lines.extend(['', '## Requested policy, model identity, and runtime', '',
                  ('Resolved model identity is taken from returned provider metadata. Native runs without '
                   'a resolved identity are marked UNKNOWN; a requested name alone is not proof of the '
                   'served model. “Requested max tokens” is not claimed as an enforced cap unless the '
                   'recorded adapter capability says so.'), '',
                  '| Run | Track / adapter | Requested model | Resolved model identity | Requested max tokens / hard cap | Effort requested | Thinking requested / supported | Seed requested / supported | Streaming / supported | p50 / p95 seconds | First request seconds | Input / output tokens |',
                  '|---|---|---|---|---|---|---|---|---|---:|---:|---:|'])
    for run in runs:
        config = run['manifest'].get('config', {})
        latency = run['latency']
        tokens = run['tokens']
        caps = run['adapter_capabilities'] if isinstance(run['adapter_capabilities'], dict) else {}
        policy = run['requested_generation_policy']
        cap = caps.get('hard_token_cap')
        cap_label = 'yes' if cap is True else 'no' if cap is False else 'unknown'
        max_tokens = policy.get('max_tokens', '—')
        effort = policy.get('effort')
        thinking = policy.get('thinking')
        thinking_support = caps.get('thinking')
        thinking_label = f"{thinking if thinking is not None else '—'} / " + (
            'yes' if thinking_support is True else 'no' if thinking_support is False else 'unknown')
        seed_cap = caps.get('sampling_seed')
        seed_label = f"{policy.get('seed', '—')} / " + ('yes' if seed_cap is True else 'no' if seed_cap is False else 'unknown')
        stream_cap = caps.get('streaming')
        stream_label = 'yes' if stream_cap is True else 'no' if stream_cap is False else 'unknown'
        token_text = '/'.join(str(tokens[key]['observed_total']) +
                              (f" (+{tokens[key]['unknown_requests']} unknown)"
                               if tokens[key]['unknown_requests'] else '')
                              for key in ('input_tokens', 'output_tokens'))
        latency_text = ' / '.join('—' if latency[key] is None else f"{latency[key]:.3f}"
                                  for key in ('p50_s', 'p95_s'))
        first = '—' if latency['first_request_s'] is None else f"{latency['first_request_s']:.3f}"
        lines.append(
            f"| {run['directory']} | {_safe_text(config.get('track', 'UNKNOWN'))} / {run['adapter_type']} | "
            f"{_safe_text(config.get('model_id', config.get('model', 'UNKNOWN')))} | "
            f"{_safe_text(_resolved_model_label(run))} | {max_tokens} / {cap_label} | "
            f"{_safe_text(effort)} | {thinking_label} | {seed_label} | {stream_label} | "
            f"{latency_text} | {first} | {token_text} |"
        )
    lines.extend(['', '## Results by family', '',
                  '| Run | View | Family | Correct / completed | Invalid |', '|---|---|---|---:|---:|'])
    for run in runs:
        for view, score in sorted(run['summary']['views'].items()):
            for family, metric in sorted(score['families'].items()):
                lines.append(f"| {run['directory']} | {view} | {family} | "
                             f"{metric['correct']}/{metric['total']} | {metric['invalid']} |")

    if live_runs:
        lines.extend(['', '## Live simulation', '',
                      ('These rows use time-weighted in-force outcomes from saved live trajectories. '
                       'Snapshot correctness and invalidation before publication are request-level '
                       'diagnostics, distinct from in-force seconds. A freshness-only guard does not '
                       'check semantic correctness; it only checks the state version. The oracle-guarded row uses ground-truth correctness '
                       'and is a reference only, never a deployable controller.'), '',
                      '| Run | Status | Track / adapter | Model identity | Trajectories completed / expected; requests | Snapshot correct / valid | Stale before publication / valid | Policy | Correct / wrong / undecided seconds (horizon) |',
                      '|---|---|---|---|---:|---:|---:|---|---:|'])
        for run in live_runs:
            for mode, title in _LIVE_MODES.items():
                force = run['in_force'][mode]
                horizon = run['observed_horizon_s']
                lines.append(
                    f"| {run['directory']} | {run['status']} | {run['track']} / {run['adapter_type']} | "
                    f"{_safe_text(run['resolved_model_display'])} | "
                    f"{run['completed_trajectories']}/{run['expected_trajectories'] if run['expected_trajectories'] is not None else '—'}; {run['request_count']} | "
                    f"{run['snapshot']['correct']}/{run['snapshot']['valid']} | "
                    f"{run['stale_before_publication']['count']}/{run['stale_before_publication']['valid']} | "
                    f"{title} | {force['correct_s']:.3f} / {force['wrong_s']:.3f} / "
                    f"{force['undecided_s']:.3f} ({horizon:.3f}) |"
                )
        lines.append('')
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_report(runs_root, output, *, live_root=None, rescore=False):
    """Write reports from saved artifacts; refuse scorer drift unless explicitly rescoring."""
    current_scorer_hash = _current_file_hash('scoring.py')
    current_live_hash = _current_file_hash('live.py')
    runs = _load_static_runs(runs_root, current_scorer_hash, rescore=rescore)
    _add_comparison_groups(runs)
    live_runs = _load_live_runs(live_root, current_scorer_hash, current_live_hash, rescore=rescore)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _build_markdown(runs, live_runs, output)
    write_json(output / 'results.json', {
        'generated_at': now(), 'rescore_mode': bool(rescore), 'runs': runs,
        'live_runs': live_runs,
        'comparison_policy': ('quality: complete shared frozen cases and identical scorer; '
                              'policy alignment additionally requires same track, count, adapter type, '
                              'and generation settings; model weights are excluded'),
    })
    return {'runs': len(runs), 'live_runs': len(live_runs), 'output': str(output),
            'requests': sum(run['record_count'] for run in runs),
            'live_requests': sum(run['request_count'] for run in live_runs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--live', help='optional root containing saved live runs')
    parser.add_argument('--rescore', action='store_true',
                        help='explicitly rescore records when the frozen scorer/source hash differs')
    args = parser.parse_args(argv)
    try:
        result = build_report(args.runs, args.output, live_root=args.live, rescore=args.rescore)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'report failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
