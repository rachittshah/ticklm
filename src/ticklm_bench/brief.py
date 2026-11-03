"""Write a plain-language benchmark brief from retained TickLM report JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path

_TIER_ORDER = (112, 448, 28)
_POLICY_ORDER = ('freshness_only', 'unguarded')


def _count(value, default=0):
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_text(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _config(run):
    return run.get('manifest', {}).get('config', {})


def _system_id(run):
    return str(_config(run).get('system_id') or run.get('directory') or 'UNKNOWN')


def _model_id(run):
    config = _config(run)
    adapter = run.get('manifest', {}).get('adapter', {})
    resolved = run.get('resolved_models', [])
    if config.get('model_id'):
        return str(config['model_id'])
    if config.get('model'):
        return str(config['model'])
    if isinstance(adapter, dict) and adapter.get('requested_model'):
        return str(adapter['requested_model'])
    if resolved:
        return ', '.join(str(value) for value in resolved)
    return 'UNKNOWN'


def _harness(run):
    adapter = run.get('adapter_type', 'UNKNOWN')
    track = _config(run).get('track', 'UNKNOWN')
    return f'{adapter} · {track}'


def _expected(run):
    return _count(run.get('expected_requests'))


def _view_summary(run, view):
    return run.get('summary', {}).get('views', {}).get(view, {})


def _view_coverage(run, view):
    coverage = _view_summary(run, view).get('coverage', {})
    expected = _count(coverage.get('expected'))
    completed = _count(coverage.get('completed'))
    failed = _count(coverage.get('failed'))
    missing = _count(coverage.get('missing'))
    return expected, completed, failed, missing


def _coverage_cell(run, view):
    expected, completed, failed, missing = _view_coverage(run, view)
    return f'{completed}/{expected}; failed {failed}, missing {missing}'


def _strict_cell(run, view):
    overall = _view_summary(run, view).get('overall', {})
    return f"{_count(overall.get('correct'))}/{_count(overall.get('total'))}"


def _invalid_pair_cell(run):
    raw_invalid = _count(_view_summary(run, 'raw').get('overall', {}).get('invalid'))
    derived_invalid = _count(_view_summary(run, 'derived').get('overall', {}).get('invalid'))
    return f'{raw_invalid} / {derived_invalid}'


def _pair_cell(run):
    values = []
    for view in ('raw', 'derived'):
        pairs = _view_summary(run, view).get('pairs', {})
        both = _count(pairs.get('both_correct'))
        complete = _count(pairs.get('complete'))
        expected = _count(pairs.get('expected'))
        values.append(f'{both}/{complete} of {expected}')
    return 'raw ' + values[0] + '; derived ' + values[1]


def _output_policy(run):
    config = _config(run)
    policy = run.get('requested_generation_policy', {})
    capabilities = run.get('adapter_capabilities', {})
    max_tokens = policy.get('max_tokens', config.get('max_tokens'))
    cap = capabilities.get('hard_token_cap') if isinstance(capabilities, dict) else None
    cap_text = 'enforced' if cap is True else 'not enforced' if cap is False else 'unknown'
    effort = policy.get('effort', config.get('effort'))
    effort_text = f'; {effort} effort requested' if effort else ''
    token_text = f'{max_tokens} requested tokens' if max_tokens is not None else 'token cap unspecified'
    return f'{token_text}; hard cap {cap_text}{effort_text}'


def _posthoc_cell(run, view):
    metric = run.get('posthoc_option_line', {}).get(view, {})
    return (f"{_count(metric.get('correct'))}/{_count(metric.get('total'))} "
            f"(invalid {_count(metric.get('invalid'))})")


def _tier_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault(_expected(run), []).append(run)
    return groups


def _run_counts(runs):
    expected = sum(_expected(run) for run in runs)
    records = sum(_count(run.get('record_count')) for run in runs)
    completed = failed = missing = strict_invalid = 0
    for run in runs:
        for view in ('raw', 'derived'):
            _, view_completed, view_failed, view_missing = _view_coverage(run, view)
            completed += view_completed
            failed += view_failed
            missing += view_missing
            strict_invalid += _count(_view_summary(run, view).get('overall', {}).get('invalid'))
    return {'runs': len(runs), 'systems': len({_system_id(run) for run in runs}),
            'expected': expected, 'records': records, 'completed': completed,
            'failed': failed, 'missing': missing, 'strict_invalid': strict_invalid}


def _tier_ledger(groups):
    tiers = [tier for tier in _TIER_ORDER if tier in groups]
    tiers.extend(sorted(tier for tier in groups if tier not in _TIER_ORDER))
    lines = [
        '| Requests per run | Runs | Recorded / scheduled attempts | Client failures | Missing | Invalid answer format |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for tier in tiers:
        counts = _run_counts(groups[tier])
        lines.append(
            f"| {tier} | {counts['runs']} | {counts['records']:,}/{counts['expected']:,} | {counts['failed']:,} | "
            f"{counts['missing']:,} | {counts['strict_invalid']:,} |"
        )
    return '\n'.join(lines), tiers


def _static_tables(groups, tiers):
    parts = []
    for tier in tiers:
        runs = sorted(groups[tier], key=lambda run: (_system_id(run).lower(), run.get('directory', '')))
        if tier == 112:
            heading = '### 112-request screening tier'
            detail = ('Each system received the same 112-request screen: 56 raw-timestamp cases and 56 '
                      'derived-value cases. The screen is a subset of the full frozen suite.')
        elif tier == 448:
            heading = '### 448-request full tier'
            detail = ('Each full run covers 224 raw-timestamp cases and 224 derived-value cases from the '
                      '448-case frozen suite.')
        elif tier == 28:
            heading = '### 28-request model/client check'
            detail = ('Each small check has 14 raw-timestamp and 14 derived-value cases. That is seven '
                      'counterfactual pairs per view, so small differences are highly uncertain.')
        else:
            heading = f'### {tier}-request tier'
            detail = 'The table keeps this tier separate because its request count differs.'
        parts.extend([
            heading, '', detail, '',
            '| Requested model | Harness | Raw: correct / scheduled | Derived: correct / scheduled | Failed or missing (raw / derived) | Invalid format (raw / derived) |',
            '|---|---|---:|---:|---:|---:|',
        ])
        for run in runs:
            config = _config(run)
            harness = {'mlx': 'MLX', 'claude_code': 'Claude Code', 'codex': 'Codex',
                       'opencode': 'OpenCode + Ollama'}.get(config.get('type'), config.get('provider', 'HTTP'))
            scores, unavailable = [], []
            for view in ('raw', 'derived'):
                expected, _, failed, missing = _view_coverage(run, view)
                correct = _count(_view_summary(run, view).get('overall', {}).get('correct'))
                scores.append(f'{correct}/{expected}')
                unavailable.append(str(failed + missing))
            parts.append(
                f'| {_as_text(_model_id(run))} | {harness} | {scores[0]} | {scores[1]} | '
                f'{" / ".join(unavailable)} | {_invalid_pair_cell(run)} |'
            )
        parts.extend(['', '**Post-hoc exact-option diagnostic** (correct / completed; invalid):', '',
                      '| System ID | Raw option-line diagnostic | Derived option-line diagnostic |',
                      '|---|---:|---:|'])
        for run in runs:
            parts.append(f"| {_as_text(_system_id(run))} | {_posthoc_cell(run, 'raw')} | "
                         f"{_posthoc_cell(run, 'derived')} |")
        parts.append('')
    return '\n'.join(parts)


def _static_collection_summary(runs, groups, tiers):
    expected_attempts = sum(_expected(run) for run in runs)
    result_rows = sum(_count(run.get('record_count')) for run in runs)
    missing = sum(_run_counts(group)['missing'] for group in groups.values())
    failed = sum(_run_counts(group)['failed'] for group in groups.values())
    unique_systems = len({_system_id(run) for run in runs})
    dataset_hashes = {run.get('dataset_sha256') for run in runs if run.get('dataset_sha256')}
    case_universe = max(tiers) if tiers else 0
    return {
        'expected_attempts': expected_attempts,
        'result_rows': result_rows,
        'missing': missing,
        'failed': failed,
        'systems': unique_systems,
        'dataset_hashes': len(dataset_hashes),
        'case_universe': case_universe,
    }


def _live_aliases(static_runs):
    aliases = {}
    for run in static_runs:
        adapter = run.get('manifest', {}).get('adapter', {})
        identifiers = {
            _system_id(run),
            _model_id(run),
        }
        identifiers.update(str(value) for value in run.get('resolved_models', []))
        if isinstance(adapter, dict) and adapter.get('requested_model'):
            identifiers.add(str(adapter['requested_model']))
        for identifier in identifiers:
            if identifier:
                aliases.setdefault(identifier.lower(), run)
    return aliases


def _match_static(live_run, aliases):
    candidates = [
        live_run.get('directory'),
        *(live_run.get('resolved_models') or []),
        live_run.get('resolved_model_display'),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate
        if cleaned.startswith('LOCAL CHECKPOINT (') and cleaned.endswith(')'):
            cleaned = cleaned[len('LOCAL CHECKPOINT ('):-1]
        if cleaned.lower() in aliases:
            return aliases[cleaned.lower()]
        for identifier, run in aliases.items():
            if ((cleaned.lower() in identifier or identifier in cleaned.lower())
                    and run.get('adapter_type') == live_run.get('adapter_type')):
                return run
    return None


def _live_rows(report, replay):
    live_runs = report.get('live_runs', [])
    aliases = _live_aliases(report.get('runs', []))
    index = {}
    for live in live_runs:
        matched = _match_static(live, aliases)
        if matched:
            index[_system_id(matched)] = live
        index[str(live.get('directory', '')).lower()] = live
    lines = [
        '| System / model | Adapter | Requests | Trajectories / observed time | Snapshot correct / valid | Correct at completion / valid | Stale before publication / valid |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    if replay:
        replay_ids = {row.get('system_id') for row in replay.get('runs', [])}
        for run in sorted(replay['runs'], key=lambda row: row.get('system_id', '')):
            system = str(run.get('system_id', 'UNKNOWN'))
            live = index.get(system) or next((value for key, value in index.items()
                                              if key and key.lower() in system.lower()), None)
            matched_static = aliases.get(system.lower())
            if not matched_static:
                matched_static = next((value for key, value in aliases.items()
                                       if key in system.lower() or system.lower() in key), None)
            model = _model_id(matched_static) if matched_static else system
            adapter = _harness(matched_static) if matched_static else 'from retained replay'
            strict_requests = run.get('strict', {}).get('requests', {})
            attempts = _count(strict_requests.get('attempts'))
            valid = _count(strict_requests.get('valid'))
            snap = _count(strict_requests.get('snapshot_correct'))
            completion = _count(strict_requests.get('completion_correct'))
            stale = live.get('stale_before_publication', {}) if live else {}
            completed_trajectories = _count(run.get('trajectory_count'))
            horizon = float(run.get('observed_horizon_s', 0.0))
            trajectory_durations = [float(row.get('observed_horizon_s', 0.0))
                                    for row in run.get('trajectories', [])]
            if trajectory_durations and all(abs(value - trajectory_durations[0]) < 1e-6
                                            for value in trajectory_durations):
                duration = f'{completed_trajectories} × {trajectory_durations[0]:g}s'
            else:
                duration = f'{completed_trajectories} trajectories'
            lines.append(
                f'| **{_as_text(model)}**<br>{_as_text(system)} | {_as_text(adapter)} | '
                f'{attempts:,} | {duration} ({horizon:.1f}s) | {snap}/{valid} | '
                f'{completion}/{valid} | {stale.get("count", "—")}/{stale.get("valid", "—")} |'
            )
        for live in live_runs:
            matched = _match_static(live, aliases)
            key = _system_id(matched) if matched else str(live.get('directory', '')).lower()
            if key not in replay_ids:
                label = matched and _system_id(matched) or live.get('directory', 'UNKNOWN')
                attempts = _count(live.get('request_count'))
                trajectories = _count(live.get('completed_trajectories'))
                horizon = float(live.get('observed_horizon_s', 0.0))
                snapshot = live.get('snapshot', {})
                stale = live.get('stale_before_publication', {})
                lines.append(
                    f'| **{_as_text(_model_id(matched)) if matched else _as_text(label)}**<br>{_as_text(label)} | '
                    f'{_as_text(_harness(matched)) if matched else _as_text(live.get("adapter_type", "UNKNOWN"))} | '
                    f'{attempts:,} | {trajectories} trajectories ({horizon:.1f}s) | '
                    f'{snapshot.get("correct", 0)}/{snapshot.get("valid", 0)} | '
                    f'{snapshot.get("completion_correct", 0)}/{snapshot.get("valid", 0)} | '
                    f'{stale.get("count", 0)}/{stale.get("valid", 0)} |'
                )
    else:
        for live in sorted(live_runs, key=lambda row: row.get('directory', '')):
            matched = _match_static(live, aliases)
            label = _system_id(matched) if matched else live.get('directory', 'UNKNOWN')
            model = _model_id(matched) if matched else live.get('resolved_model_display', label)
            snapshot = live.get('snapshot', {})
            stale = live.get('stale_before_publication', {})
            lines.append(
                f'| **{_as_text(model)}**<br>{_as_text(label)} | '
                f'{_as_text(_harness(matched)) if matched else _as_text(live.get("adapter_type", "UNKNOWN"))} | '
                f"{_count(live.get('request_count')):,} | "
                f"{_count(live.get('completed_trajectories'))}/{_count(live.get('expected_trajectories'))}; "
                f"{float(live.get('observed_horizon_s', 0.0)):.1f}s | "
                f"{snapshot.get('correct', 0)}/{snapshot.get('valid', 0)} | "
                f"{snapshot.get('completion_correct', 0)}/{snapshot.get('valid', 0)} | "
                f"{stale.get('count', 0)}/{stale.get('valid', 0)} |"
            )
    return '\n'.join(lines)


def _format_seconds(values):
    return ' / '.join(f"{float(values.get(key, 0.0)):.1f}" for key in ('correct_s', 'wrong_s', 'undecided_s'))


def _replay_table(replay, report):
    if not replay:
        return ''
    lines = [
        '| System / score mode | Attempts | Accepted / invalid / failed | Snapshot correct / wrong | Freshness-only correct / wrong / undecided (seconds) | Unguarded correct / wrong / undecided (seconds) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    static_aliases = _live_aliases(report.get('runs', []))
    for run in sorted(replay['runs'], key=lambda row: row.get('system_id', '')):
        system = str(run.get('system_id', 'UNKNOWN'))
        static = static_aliases.get(system.lower())
        model = _model_id(static) if static else system
        for mode, label in (('strict', 'Strict parser'), ('alternative', 'Post-hoc exact-option parser')):
            summary = run.get(mode, {})
            requests = summary.get('requests', {})
            in_force = summary.get('in_force', {})
            freshness = in_force.get('freshness_only', {})
            unguarded = in_force.get('unguarded', {})
            accepted = _count(requests.get('valid'))
            invalid = _count(requests.get('invalid'))
            failed = _count(requests.get('failed'))
            snapshot_correct = _count(requests.get('snapshot_correct'))
            snapshot_wrong = _count(requests.get('snapshot_wrong'))
            lines.append(
                f'| {_as_text(model)} · {label} | {_count(requests.get("attempts")):,} | '
                f'{accepted} / {invalid} / {failed} | {snapshot_correct} / {snapshot_wrong} | '
                f'{_format_seconds(freshness)} | {_format_seconds(unguarded)} |'
            )
    return '\n'.join(lines)


def _dated(generated_at):
    if isinstance(generated_at, str):
        try:
            parsed = datetime.fromisoformat(generated_at)
            return parsed.date().isoformat()
        except ValueError:
            pass
    return date(2026, 11, 3).isoformat()


def _relative_link(target, output_path):
    return Path(os.path.relpath(Path(target).resolve(), Path(output_path).resolve().parent))


def render_brief(report, input_path, output_path, replay=None, replay_path=None):
    if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
        raise TypeError('input must be a TickLM results JSON object containing a runs list')
    runs = report['runs']
    groups = _tier_runs(runs)
    ledger, tiers = _tier_ledger(groups)
    summary = _static_collection_summary(runs, groups, tiers)
    unique_system_ids = len({_system_id(run) for run in runs})
    all_system_ids = sorted({_system_id(run) for run in runs})
    app_date = _dated(report.get('generated_at'))
    result_markdown = Path(input_path).with_name('results.md')
    result_link = _relative_link(result_markdown, output_path) if result_markdown.is_file() else None
    external_validation = Path(__file__).resolve().parents[2] / 'docs' / 'benchmark-external-validation.md'
    external_link = _relative_link(external_validation, output_path)
    replay_sha = replay_path and hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()

    lines = [
        '# TickLM temporal benchmark: what we measured', '',
        f'**Measured 3 November 2026. Report data generated {app_date}. Experimental, synthetic, and not a state-of-the-art ranking.**', '',
        ('On the simple static questions, frontier-client replies were often correct when the strict parser accepted '
         'them. The live runs exposed a different problem: response formatting and delay can make an answer unusable '
         'after the state changes. This does not show that a model cannot understand time. The clock and event '
         'state live in the harness. In these runs, models receive time readings rather than '
         'measuring elapsed wall time themselves.'), '',
        (f"The retained static runs contain **{summary['expected_attempts']:,} scheduled prompt attempts** across "
         f"{len(runs)} run records and **{unique_system_ids} unique system IDs**. The largest frozen case set has "
        f"**{summary['case_universe']} cases**. In this release the 112-case and 28-case tiers are exact subsets "
        'of the 448-case full set; the saved case IDs were checked. '
         f"Across tiers, the report contains {summary['result_rows']:,} result rows, "
         f"{summary['failed']:,} failed replies, and {summary['missing']:,} missing replies. These attempts repeat "
         'cases across systems; they are not new independent cases.'), '',
        '## At a glance', '', ledger, '',
        (f"There are **{summary['dataset_hashes']} frozen dataset hashes** and **{unique_system_ids} unique system/harness IDs**. "
         'The same Qwen3-0.6B and SmolLM3 system IDs appear in both screen and full tiers; those are repeated runs '
         'on more cases, not additional model configurations.'), '',
    ]
    if result_link:
        lines.append(f'[Detailed coverage, family, and latency tables](<{result_link}>) are generated from the same saved run records.')
        lines.append('')
    lines.extend([
        '## Static model runs', '',
        ('The suite is four-choice synthetic English. Each case has a raw-timestamp form and a derived-value form; '
         'the derived prompt includes time differences computed outside the model. The model chooses a labelled action '
         'from those supplied facts. It is not asked to run a clock or fetch time.'), '',
        _static_tables(groups, tiers),
        ('**Reading these tables:** the short tables show `strictly correct / scheduled questions`, so missing and '
         'failed requests cannot inflate the visible fraction. This is delivered task success, not pure reasoning '
         'accuracy. The detailed tables separately score completed replies, coverage and both-correct counterfactual '
         'pairs. Invalid response formats count wrong. The 28-request tier has only seven pairs per view. Direct runs '
         'request a 256-token cap; native clients do not enforce that cap and request xhigh effort.'), '',
        ('The exact-option diagnostic was added after the first Qwen run showed that some models answered with a whole '
         'option line such as `A. ALARM` instead of a bare letter. It was applied uniformly to all systems, but it is '
         '**post-hoc and was not preregistered**. It does not replace or edit the primary strict score.'), '',
        ('Some full-tier small-model runs have strict accuracy of 0/224 in each view while all 224 completed replies '
         'are strict-invalid. That is a format/parser result, not evidence that those models have zero temporal ability. '
         'The separate exact-option diagnostic makes some whole-option replies scoreable; it does not change the '
         'model text or turn a wrong choice into a correct one.'), '',
        '## Live event timing', '',
        ('The live test uses three real-clock 30-second trajectories per system, with scripted events. The harness '
         'records actual elapsed time, including any overrun, and an external timestamped state for each request. '
         'An event clock keeps advancing while a model is '
         'generating; the model does not receive events that arrive after its prompt was sent.'), '',
        _live_rows(report, replay), '',
        ('A response can be correct for its request snapshot and already be outdated when generation finishes. '
         'That is not a semantic reasoning error and does not prove that a model cannot understand time: the new event '
         'was unseen by the model. No model can react to an event it has not received; the controller has to refresh '
         'the prompt or suppress the old output. The report '
         'therefore separates snapshot correctness, correctness at completion, and state/deadline changes before publication.'), '',
        ('The *freshness-only* rule tracks state versions and deadlines; it suppresses stale output but does not judge '
         'whether the model chose the right action. *Unguarded* keeps the last model action active. The *oracle-guarded* '
         'calculation uses the correct answer and is a non-deployable reference. Guard behavior is deterministic harness '
         'logic; it is not model capability.'), '',
    ])
    if replay:
        lines.extend([
            '### Offline output-format replay', '',
            _as_text(replay.get('scope', 'This is an offline parser counterfactual on saved replies.')), '',
            _replay_table(replay, report), '',
            ('The alternate parser accepts the frozen strict format or one exact whole option line. It reuses the '
             'original recorded request and publication times and assumes zero extra parser latency. It is a diagnostic '
             'of response formatting plus choice, not a new model trial.'), '',
        ])
    lines.extend([
        '### What the clock supplies', '',
        ('For example, a development prompt gives **Start = 107.25 seconds** and **Now = 112.75 seconds**. The '
         'correct difference is `112.75 − 107.25 = 5.50 seconds`, below the **5.75-second threshold**, so the answer '
         'is “wait.” In the raw view the model must subtract the supplied timestamps; in the derived view the program '
         'supplies the difference. This is a development example, not a measured live interval. In the live test, '
         'Python `time.monotonic()` records points A and B and the program calculates `B − A`; the model interprets '
         'that elapsed time and the latest supplied event. TimesFM and Bekko were not used to calculate these times '
         'or generate answers in this benchmark. Their earlier forecasting and decision experiments are separate.'), '',
        '```mermaid',
        'flowchart LR',
        '    clock["External clock and event queue"] --> snapshot["Request-time state and timestamps"]',
        '    snapshot --> inputs["Raw timestamps or externally computed differences"]',
        '    inputs --> model["Model chooses an action"]',
        '    clock -. time and events continue during generation .-> check["Completion-time state/version check"]',
        '    model --> check',
        '    check -->|still current| publish["Action may be used"]',
        '    check -->|expired or changed| refresh["Suppress old answer and refresh input"]',
        '```', '',
        '## How to interpret the results', '',
        ('The static task is deliberately simple. High accuracy on accepted static responses shows that a system can '
         'answer these supplied-fact questions; it does not establish broad temporal reasoning, numerical extrapolation, '
         'or general instruction following. A correct snapshot answer can become useless after a long response or a '
         'new event. In the live test, the important failures separate into answer formatting, decision quality at the '
         'snapshot, and freshness at completion.'), '',
        ('Models and harnesses did not receive equal compute. Direct MLX generation used local checkpoints on an Apple '
         'M4 Max with 36 GB memory; checkpoint sizes and quantization differ. Claude Code and Codex ran through existing '
         'signed-in clients with their own prompts and transport. Their 256-token request was not an enforced hard cap; '
         'effort such as `xhigh` is a requested client setting, not independent evidence of internal compute. The local '
         'HTTP Ollama route is reported as its own adapter. Observed latency is illustrative: one host ran local models, '
         'native clients, and other CPU work, so it is not a controlled speed ranking.'), '',
        ('No direct metered provider API calls were charged to the separate $25 budget. Native client usage came through '
         'pre-existing sign-ins; subscription consumption is not equated with zero cost here. No option-order variants '
         'were run. These are small, single-run samples with a few complete counterfactual pairs, especially in the '
         '28-request tier; confidence intervals and zero-width intervals should not be read as seed or deployment uncertainty.'), '',
        ('The evidence supports a modest engineering point: keep the clock and event state outside the model, let a model '
         'choose among supplied actions when useful, and use a deterministic freshness gate before acting on its answer. '
         'It does not show that a weight update is necessary; a fast deterministic policy remains the right baseline for '
         'rules that can be stated exactly.'), '',
        '## Related published work', '',
        ('This synthetic suite is not a replacement for existing temporal benchmarks and has no official leaderboard score. '
         'No TickLM model in this report was evaluated on these external datasets. The primary papers ask related but '
         'different questions:'), '',
        '- [TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](https://arxiv.org/abs/2505.12891) studies temporal questions from wiki, news, and dialogue.',
        '- [MultiTempBench](https://arxiv.org/abs/2603.19017) examines date arithmetic, time-zone conversion, and temporal relations across languages and calendar formats.',
        '- [TRACE](https://arxiv.org/abs/2607.04784) generates interval-constraint tasks with controlled difficulty and trace verification.',
        '- [StreamDecisionBench](https://arxiv.org/abs/2609.38612) measures whether a decision remains correct while language streams evolve and answers arrive late.',
        '- [AgentTime](https://arxiv.org/abs/2610.09944) tests agent runtime estimation, duration following, and control in native agent harnesses.', '',
        ('A separate offline check reproduced **54.105400773%** normalized log-area-under-curve from one existing '
         'StreamDecisionBench author recording. That is the upstream recording’s scorer result, not a TickLM model score '
         'or a score from this synthetic suite; no new model inference was used for it. See the [external validation notes]'
         f'(<{external_link}>).'), '',
        '## Reproducibility and source data', '',
        (f"Input report generated {app_date}. Static results are grouped by frozen dataset hash and system ID; the report input contains "
         f"{summary['dataset_hashes']} dataset hashes. The full JSON and per-run tables retain coverage, model aliases, and parser outputs."), '',
        f"- Input report: `{Path(input_path).name}`",
    ])
    if replay_path:
        lines.append(f'- Offline replay: `{Path(replay_path).name}`; SHA-256 `{replay_sha}`.')
    if result_link:
        lines.append(f'- Machine-readable run summaries and detailed tables: [results JSON](<{_relative_link(input_path, output_path)}>) and [results Markdown](<{result_link}>).')
    lines.extend([
        f"- Largest frozen tier: {summary['case_universe']} cases; the smaller tiers are exact subsets in this release.",
        f'- System IDs represented: {", ".join(f"`{_as_text(value)}`" for value in all_system_ids)}.',
        '- External scorer reproduction, input checks, and limitations: [benchmark external validation](<' + str(external_link) + '>).',
        '',
    ])
    return '\n'.join(lines)


def build_brief(input_path, output_path, *, replay_path=None):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f'input results JSON does not exist: {input_path}')
    report = json.loads(input_path.read_text(encoding='utf-8'))
    replay = None
    if replay_path is not None:
        replay_path = Path(replay_path)
        if not replay_path.is_file():
            raise FileNotFoundError(f'input replay JSON does not exist: {replay_path}')
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        if not isinstance(replay, dict) or not isinstance(replay.get('runs'), list):
            raise TypeError('replay JSON must contain a runs list')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_brief(report, input_path, output_path, replay=replay, replay_path=replay_path)
    output_path.write_text(content, encoding='utf-8')
    return {'output': str(output_path), 'static_runs': len(report.get('runs', [])),
            'live_runs': len(report.get('live_runs', [])),
            'replay_runs': len(replay.get('runs', [])) if replay else 0,
            'systems': len({_system_id(run) for run in report.get('runs', [])})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='TickLM report results.json')
    parser.add_argument('--output', required=True, help='destination Markdown file')
    parser.add_argument('--replay', help='optional offline parser replay JSON')
    args = parser.parse_args(argv)
    try:
        result = build_brief(args.input, args.output, replay_path=args.replay)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f'brief generation failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
