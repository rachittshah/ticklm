"""Render publication-ready static and live charts from ``report.py`` JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_COLORS = {
    'correct': '#23856d',
    'wrong': '#c65a57',
    'undecided': '#c8cdd3',
    'strict': '#245a81',
    'diagnostic': '#d18b28',
    'grid': '#dce1e6',
    'muted': '#5d6873',
    'incomplete': '#7c8792',
}
_LIVE_POLICIES = ('freshness_only', 'unguarded')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _system_id(run):
    config = run.get('manifest', {}).get('config', {})
    return str(config.get('system_id') or run.get('directory') or 'UNKNOWN')


def _track_label(track):
    return {'direct-generation': 'Direct', 'native-client': 'Native'}.get(track, str(track))


def _safe_float(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _validate_report(report):
    if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
        raise TypeError('input must be a TickLM report JSON object with a runs list')
    if not isinstance(report.get('live_runs', []), list):
        raise TypeError('input live_runs field must be a list')


def _format_count(value):
    return '—' if value is None else str(value)


def _live_chart(live_runs, output):
    if not live_runs:
        raise ValueError('report has no live runs to plot')
    rows = []
    references = []
    for run in sorted(live_runs, key=lambda item: (str(item.get('track', '')), item.get('directory', ''))):
        horizon = _safe_float(run.get('observed_horizon_s'), f"{run.get('directory')} horizon")
        if horizon <= 0:
            raise ValueError(f"{run.get('directory')}: observed horizon must be positive")
        policies = run.get('in_force', {})
        for policy in _LIVE_POLICIES:
            values = policies.get(policy)
            if not isinstance(values, dict):
                raise TypeError(f"{run.get('directory')}: missing live policy {policy}")
            times = [_safe_float(values.get(key), f'{policy}.{key}')
                     for key in ('correct_s', 'wrong_s', 'undecided_s')]
            if min(times) < 0 or not np.isclose(sum(times), horizon, rtol=1e-6, atol=1e-6):
                raise ValueError(f"{run.get('directory')}: {policy} durations do not partition the horizon")
            status = str(run.get('status', 'UNKNOWN'))
            completed = run.get('completed_trajectories')
            expected = run.get('expected_trajectories')
            complete = status == 'complete' and completed is not None and completed == expected
            label = f"{_system_id(run)} · {_track_label(run.get('track', 'UNKNOWN'))}"
            rows.append({
                'run': run,
                'policy': policy,
                'times': times,
                'horizon': horizon,
                'label': label + (' · incomplete' if not complete else ''),
                'complete': complete,
            })
        oracle = policies.get('oracle_guarded')
        if isinstance(oracle, dict):
            refs = [_safe_float(oracle.get(key), f'oracle_guarded.{key}')
                    for key in ('correct_s', 'wrong_s', 'undecided_s')]
            if min(refs) < 0 or not np.isclose(sum(refs), horizon, rtol=1e-6, atol=1e-6):
                raise ValueError(f"{run.get('directory')}: oracle reference durations do not partition the horizon")
            references.append((run, refs))

    row_count = len(rows)
    fig_height = max(4.8, 0.50 * row_count + 2.4)
    fig = plt.figure(figsize=(16.8, fig_height), facecolor='white')
    grid = fig.add_gridspec(1, 2, width_ratios=(1.04, 1.32))
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.24, right=0.99, wspace=0.04)
    y = np.arange(row_count)
    ax.set_yticks(y, [row['label'] + ' · ' + row['policy'].replace('_', ' ') for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20), [f'{tick}%' for tick in range(0, 101, 20)])
    ax.set_xlabel('Share of observed horizon')
    ax.set_title('Time-weighted policy outcomes', loc='left', fontsize=12, weight='bold', pad=9)
    ax.xaxis.grid(True, color=_COLORS['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(_COLORS['grid'])
    ax.tick_params(axis='y', length=0, pad=8, labelsize=9)
    ax.tick_params(axis='x', colors=_COLORS['muted'], labelsize=8)

    colors = [_COLORS['correct'], _COLORS['wrong'], _COLORS['undecided']]
    for row_index, row in enumerate(rows):
        left = 0.0
        for value, color, label in zip(row['times'], colors, ('Correct', 'Wrong', 'Undecided'), strict=True):
            width = 100.0 * value / row['horizon']
            ax.barh(row_index, width, left=left, height=0.62, color=color,
                    edgecolor='white', linewidth=0.55)
            if width >= 13:
                text_color = 'white' if label in ('Correct', 'Wrong') else '#39434d'
                ax.text(left + width / 2, row_index, f'{width:.0f}%', ha='center', va='center',
                        fontsize=8, color=text_color, weight='bold')
            left += width
        if not row['complete']:
            ax.axhspan(row_index - 0.34, row_index + 0.34, color='#f4f5f6', zorder=-1)

    legend = [Line2D([0], [0], color=_COLORS[key], lw=8, label=label)
              for key, label in (('correct', 'Correct'), ('wrong', 'Wrong'), ('undecided', 'Undecided'))]
    fig.legend(handles=legend, ncol=3, frameon=False, loc='center left',
               bbox_to_anchor=(0.40, 0.98), fontsize=8, handlelength=1.3)

    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(row_count - 0.5, -0.93)
    table_ax.axis('off')
    columns = [
        (0.03, 'Correct / wrong /\nundecided (seconds)'),
        (0.39, 'Snapshot\ncorrect / valid'),
        (0.55, 'Completion\ncorrect / valid'),
        (0.70, 'Stale before\npublication / valid'),
        (0.88, 'Valid / requests;\ntrajectories'),
    ]
    for x, label in columns:
        table_ax.text(x, -0.63, label, ha='left', va='center', fontsize=8,
                      color=_COLORS['muted'], weight='bold', linespacing=1.15)
    table_ax.plot([0.02, 0.99], [-0.38, -0.38], color=_COLORS['grid'], linewidth=0.9)
    for row_index, row in enumerate(rows):
        run = row['run']
        correct, wrong, undecided = row['times']
        snapshot = run.get('snapshot', {})
        stale = run.get('stale_before_publication', {})
        valid = snapshot.get('valid', 0)
        total = run.get('request_count', '—')
        completed = _format_count(run.get('completed_trajectories'))
        expected = _format_count(run.get('expected_trajectories'))
        values = [
            (0.03, f'{correct:.1f} / {wrong:.1f} / {undecided:.1f}'),
            (0.39, f"{snapshot.get('correct', 0)} / {valid}"),
            (0.55, f"{run.get('snapshot', {}).get('completion_correct', 0)} / {valid}"),
            (0.70, f"{stale.get('count', 0)} / {stale.get('valid', 0)}"),
            (0.88, f'{valid} / {total}; {completed}/{expected}'),
        ]
        for x, value in values:
            table_ax.text(x, row_index, value, ha='left', va='center', fontsize=8.5,
                          color='#26323b' if row['complete'] else _COLORS['incomplete'],
                          family='DejaVu Sans Mono')

    fig.suptitle('Live temporal decision policies', x=0.01, y=0.98, ha='left',
                 fontsize=17, weight='bold', color='#1d2a33')
    footnotes = [
        'Bars show each policy’s share of its observed horizon; the table gives exact correct / wrong / undecided seconds.',
        'Counts use strict parsed outputs. Snapshot and completion correctness are separate; stale means state or deadline changed before publication.',
        'Oracle-guarded ground-truth policy is omitted from the comparison because it is not deployable.',
    ]
    if references:
        refs_text = '; '.join(
            f"{_system_id(run)} reference oracle: {times[0]:.1f} correct / {times[1]:.1f} wrong / "
            f'{times[2]:.1f} undecided s'
            for run, times in references
        )
        footnotes.append(refs_text)
    fig.text(0.01, 0.02, '\n'.join(footnotes), ha='left', va='bottom', fontsize=8,
             color=_COLORS['muted'], linespacing=1.3)
    _save_figure(fig, output, 'live-policy-outcomes', 'Live temporal decision policies')
    plt.close(fig)
    return rows


def _static_chart(runs, output):
    selected = [run for run in runs if run.get('expected_requests') == 112]
    if not selected:
        raise ValueError('report has no 112-request static runs')
    selected.sort(key=lambda run: (_system_id(run).lower(), run.get('adapter_type', '')))
    views = ('raw', 'derived')
    row_count = len(selected)
    fig_height = max(6.8, 0.55 * row_count + 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(15.7, fig_height), sharey=True, facecolor='white')
    fig.subplots_adjust(top=0.90, bottom=0.21, left=0.24, right=0.99, wspace=0.06)
    y = np.arange(row_count)
    row_labels = [f"{_system_id(run)} · {_track_label(run.get('manifest', {}).get('config', {}).get('track', 'UNKNOWN'))}"
                  for run in selected]
    markers = {'direct-generation': 'o', 'native-client': 's'}
    completeness = {}
    for axis, view in zip(axes, views, strict=True):
        axis.set_xlim(-1, 101)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel('Accuracy among the 56 cases in this view (%)')
        axis.set_title('Raw evidence' if view == 'raw' else 'Derived evidence',
                       loc='left', fontsize=12, weight='bold', pad=9)
        axis.set_axisbelow(True)
        axis.xaxis.grid(True, color=_COLORS['grid'], linewidth=0.8)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color(_COLORS['grid'])
        axis.tick_params(axis='y', length=0, pad=7, labelsize=8.5)
        axis.tick_params(axis='x', colors=_COLORS['muted'], labelsize=8)
        for row_index, run in enumerate(selected):
            view_data = run.get('summary', {}).get('views', {}).get(view, {})
            overall = view_data.get('overall', {})
            coverage = view_data.get('coverage', {})
            expected = coverage.get('expected', 56)
            completed = coverage.get('completed', overall.get('total', 0))
            is_complete = coverage.get('complete') is True and completed == expected
            completeness.setdefault(run.get('directory'), {})[view] = {
                'complete': is_complete,
                'expected': expected,
                'completed': completed,
                'strict_correct': overall.get('correct', 0),
                'strict_total': overall.get('total', 0),
            }
            if not is_complete:
                strict_total = overall.get('total', 0)
                strict_acc = (100 * overall.get('correct', 0) / strict_total
                              if strict_total else None)
                acc_text = 'no valid responses' if strict_acc is None else f'available strict {strict_acc:.0f}%'
                note = f"INCOMPLETE TRANSPORT · {completed}/{expected} · {acc_text} · not ranked"
                axis.axhspan(row_index - 0.38, row_index + 0.38, color='#f4f5f6', zorder=-1)
                axis.text(1, row_index, note, ha='left', va='center', fontsize=7.5,
                          color=_COLORS['incomplete'], style='italic')
                continue
            strict = 100 * overall.get('accuracy', 0.0)
            diagnostic = run.get('posthoc_option_line', {}).get(view, {})
            diagnostic_total = diagnostic.get('total', 0)
            if diagnostic_total != expected:
                raise ValueError(f"{run.get('directory')}/{view}: complete coverage has incomplete post-hoc scores")
            posthoc = (100 * diagnostic.get('correct', 0) / diagnostic_total
                       if diagnostic_total else 0.0)
            track = run.get('manifest', {}).get('config', {}).get('track', 'UNKNOWN')
            marker = markers.get(track, 'o')
            axis.plot(strict, row_index - 0.105, marker=marker, linestyle='none',
                      markersize=7.5, markerfacecolor=_COLORS['strict'],
                      markeredgecolor='white', markeredgewidth=0.8, zorder=3)
            axis.plot(posthoc, row_index + 0.105, marker=marker, linestyle='none',
                      markersize=7.5, markerfacecolor=_COLORS['diagnostic'],
                      markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    axes[0].set_yticks(y, row_labels)
    axes[1].tick_params(axis='y', labelleft=False)
    axes[0].invert_yaxis()

    score_legend = [
        Line2D([0], [0], marker='o', linestyle='none', markersize=7,
               markerfacecolor=_COLORS['strict'], markeredgecolor='white',
               label='Strict parser'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=7,
               markerfacecolor=_COLORS['diagnostic'], markeredgecolor='white',
               label='Post-hoc exact-option diagnostic'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=7,
               markerfacecolor='#68737d', markeredgecolor='white',
               label='Direct-generation harness'),
        Line2D([0], [0], marker='s', linestyle='none', markersize=7,
               markerfacecolor='#68737d', markeredgecolor='white',
               label='Native client harness (if complete)'),
    ]
    fig.legend(handles=score_legend, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.075), fontsize=8)
    fig.suptitle('112-request model comparison · strict and post-hoc scoring',
                 x=0.01, y=0.98, ha='left', fontsize=17, weight='bold', color='#1d2a33')
    fig.text(0.01, 0.012,
             'Each view contains 56 cases. Strict counts invalid outputs as incorrect; the separate post-hoc diagnostic accepts an exact option line only.\n'
             'The exact-option diagnostic was added after the first Qwen run; it is post-hoc and was not preregistered.\n'
             'Only complete views are plotted. Incomplete native-client rows show coverage and available-case strict score outside the ranking.\n'
             'Marker shape identifies harness; color identifies score.',
             ha='left', va='bottom', fontsize=8, color=_COLORS['muted'], wrap=True)
    _save_figure(fig, output, 'static-112-tier', '112-request static comparison')
    plt.close(fig)
    return completeness


def _save_figure(fig, output, stem, title):
    svg_path, png_path = output / f'{stem}.svg', output / f'{stem}.png'
    with matplotlib.rc_context({'svg.hashsalt': 'TickLM-Bench-v0.1'}):
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white',
                    metadata={'Date': None, 'Creator': 'TickLM-Bench charts', 'Title': title})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white',
                metadata={'Title': title, 'Software': 'TickLM-Bench charts'})


def build_charts(input_path, output_path):
    """Render two SVG/PNG figures and a small provenance manifest from report JSON."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f'input report does not exist: {input_path}')
    report = json.loads(input_path.read_text(encoding='utf-8'))
    _validate_report(report)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    live_rows = _live_chart(report.get('live_runs', []), output)
    static_coverage = _static_chart(report['runs'], output)
    manifest = {
        'input_report': str(input_path.resolve()),
        'input_report_sha256': _sha256(input_path),
        'chart_source_sha256': _sha256(Path(__file__)),
        'matplotlib_version': matplotlib.__version__,
        'matplotlib_dependency': f'matplotlib=={matplotlib.__version__} (project report extra)',
        'reproduction_command': shlex.join([
            'uv', 'run', 'python', '-m', 'ticklm_bench.charts', '--input', str(input_path),
            '--output', str(output.resolve()),
        ]),
        'figures': {
            'live-policy-outcomes': {'formats': ['svg', 'png'],
                                     'live_rows': len(live_rows),
                                     'policies': list(_LIVE_POLICIES),
                                     'oracle_policy': 'reference only; excluded from main comparison'},
            'static-112-tier': {'formats': ['svg', 'png'],
                                'included_runs': [run.get('directory') for run in report['runs']
                                                  if run.get('expected_requests') == 112],
                                'coverage_by_run': static_coverage,
                                'score': 'micro accuracy; strict and exact-option diagnostic',
                                'incomplete_views_score_points_plotted': False},
        },
        'files': ['live-policy-outcomes.svg', 'live-policy-outcomes.png',
                  'static-112-tier.svg', 'static-112-tier.png'],
    }
    (output / 'charts-manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='report.py results.json')
    parser.add_argument('--output', required=True, help='directory for SVG, PNG, and provenance JSON')
    args = parser.parse_args(argv)
    try:
        manifest = build_charts(args.input, args.output)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f'chart generation failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps({
        'output': str(Path(args.output).resolve()),
        'matplotlib_version': manifest['matplotlib_version'],
        'files': manifest['files'],
        'input_report_sha256': manifest['input_report_sha256'],
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
