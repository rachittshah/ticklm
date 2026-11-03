"""Frozen answer parsing and coverage-aware, episode-clustered benchmark scoring.

Records have ``case`` and ``result`` dictionaries. A result's status must be
``ok`` for its text to be scored. Failed and missing requests are coverage
failures, while malformed successful responses count as incorrect. Supplying the
frozen expected case list is required to assert complete, comparable coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np

_PATTERN = re.compile(r'(?:(?i:answer|final)\s*:\s*)?([ABCD])[.!?]?')


def parse_answer(text):
    """Parse an entire answer or return None; do not extract from prose or reasoning."""
    if not isinstance(text, str):
        return None
    value = text.strip()
    match = _PATTERN.fullmatch(value)
    if match:
        return match.group(1)
    try:
        parsed = json.loads(value, object_pairs_hook=lambda pairs: pairs)
    except (ValueError, TypeError):
        return None
    if (isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], tuple)
            and parsed[0][0] == 'answer' and parsed[0][1] in ('A', 'B', 'C', 'D')):
        return parsed[0][1]
    return None


def _quality(rows):
    completed = [row for row in rows if row['status'] == 'ok']
    total = len(completed)
    correct = sum(row['correct'] for row in completed)
    invalid = sum(row['prediction'] is None for row in completed)
    return {'correct': correct, 'total': total, 'invalid': invalid,
            'accuracy': correct / total if total else None,
            'invalid_rate': invalid / total if total else None}


def _bootstrap(rows, families, samples, seed):
    rng = np.random.default_rng(seed)
    draws = []
    by_family = {}
    for family in families:
        episodes = defaultdict(list)
        for row in rows:
            if row['case']['family'] == family:
                episodes[row['case']['episode_id']].append(row)
        aggregates = np.array([[sum(row['correct'] for row in episodes[key]), len(episodes[key])]
                               for key in sorted(episodes)], dtype=np.int64)
        indices = rng.integers(0, len(aggregates), size=(samples, len(aggregates)))
        resampled = aggregates[indices].sum(axis=1)
        scores = resampled[:, 0] / resampled[:, 1]
        draws.append(scores)
        by_family[family] = [float(v) for v in np.percentile(scores, [2.5, 97.5])]
    macro = np.mean(np.array(draws), axis=0)
    return {'macro_accuracy': [float(v) for v in np.percentile(macro, [2.5, 97.5])],
            'by_family': by_family, 'samples': samples, 'seed': seed,
            'method': 'Episode-cluster bootstrap stratified by family; percentile 95% interval.'}


def summarize(records, expected_cases=None, bootstrap_samples=2000, seed=1103):
    """Summarize one result per case, without pooling raw and derived views.

    ``predicted``, when supplied, must match the frozen parser. An expected case
    list allows missing cases to be counted; without it, completeness is unknown.
    The flip statistic compares predicted and gold action-change patterns, and
    is separate from the stricter both-correct statistic. A malformed answer in
    a completed pair is not flip-consistent.
    """
    if not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError('bootstrap_samples must be positive')
    indexed_records = {}
    for record in records:
        case = record['case']
        case_id = case['id']
        if case_id in indexed_records:
            raise ValueError(f'duplicate record ID: {case_id}')
        indexed_records[case_id] = record
    cases = list(expected_cases) if expected_cases is not None else [r['case'] for r in records]
    indexed_cases = {}
    episode_families = {}
    pair_episodes = {}
    for case in cases:
        case_id = case['id']
        if case_id in indexed_cases:
            raise ValueError(f'duplicate expected case ID: {case_id}')
        if case['view'] not in ('raw', 'derived'):
            raise ValueError('unknown input view')
        if set(case['choices']) != set('ABCD') or case['answer'] not in case['choices']:
            raise ValueError('invalid choices or answer')
        episode, family, pair = case['episode_id'], case['family'], case['pair_id']
        if episode in episode_families and episode_families[episode] != family:
            raise ValueError('episode occurs in multiple families')
        episode_families[episode] = family
        pair_key = (family, pair)
        if pair_key in pair_episodes and pair_episodes[pair_key] != episode:
            raise ValueError('pair occurs in multiple episodes')
        pair_episodes[pair_key] = episode
        indexed_cases[case_id] = case
    unknown = set(indexed_records) - set(indexed_cases)
    if unknown:
        raise ValueError('unknown record case ID')
    rows_by_view = defaultdict(list)
    for case_id, case in indexed_cases.items():
        record = indexed_records.get(case_id)
        if record is not None and record['case'] != case:
            raise ValueError(f'record case differs from expected: {case_id}')
        result = record.get('result') if record is not None else None
        status = result.get('status') if isinstance(result, dict) else None
        status = 'missing' if status in (None, 'missing', 'not_run', 'skipped') else status
        prediction = parse_answer(result.get('text')) if status == 'ok' else None
        if record is not None and 'predicted' in record and record['predicted'] != prediction:
            raise ValueError(f'prediction disagrees with frozen parser: {case_id}')
        rows_by_view[case['view']].append({'case': case, 'status': status,
                                           'received': record is not None, 'prediction': prediction,
                                           'correct': prediction == case['answer'] if status == 'ok' else False})
    summaries = {}
    for view, rows in sorted(rows_by_view.items()):
        families = sorted({row['case']['family'] for row in rows})
        quality = {family: _quality([row for row in rows if row['case']['family'] == family])
                   for family in families}
        missing = sum(row['status'] == 'missing' for row in rows)
        failed = sum(row['status'] not in ('missing', 'ok') for row in rows)
        complete = expected_cases is not None and missing == 0 and failed == 0
        groups = defaultdict(list)
        for row in rows:
            case = row['case']
            groups[(case['family'], case['pair_id'])].append(row)
        structurally_complete = all(
            len(group) == 2 and {r['case']['variant'] for r in group} == {0, 1}
            and group[0]['case']['choices'] == group[1]['case']['choices']
            and group[0]['case']['choices'][group[0]['case']['answer']] !=
            group[1]['case']['choices'][group[1]['case']['answer']]
            for group in groups.values())
        comparable = bool(complete and structurally_complete and groups)
        scored_pairs = [group for group in groups.values()
                        if len(group) == 2 and all(row['status'] == 'ok' for row in group)]
        both = sum(all(row['correct'] for row in group) for group in scored_pairs)
        flip = 0
        for group in scored_pairs:
            if any(row['prediction'] is None for row in group):
                continue
            predicted = [row['case']['choices'][row['prediction']] for row in group]
            gold = [row['case']['choices'][row['case']['answer']] for row in group]
            flip += (predicted[0] != predicted[1]) == (gold[0] != gold[1])
        count = len(scored_pairs)
        accuracies = [quality[family]['accuracy'] for family in families]
        macro = sum(accuracies) / len(accuracies) if all(x is not None for x in accuracies) else None
        canonical = json.dumps(sorted((row['case'] for row in rows), key=lambda case: case['id']),
                               sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        case_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest() if expected_cases is not None else None
        summaries[view] = {
            'case_set_sha256': case_hash,
            'coverage': {'expected': len(rows) if expected_cases is not None else None,
                         'received': sum(row['received'] for row in rows),
                         'completed': sum(row['status'] == 'ok' for row in rows),
                         'missing': missing, 'failed': failed,
                         'complete': complete if expected_cases is not None else None},
            'overall': _quality(rows), 'families': quality, 'macro_accuracy': macro,
            'pairs': {'expected': len(groups), 'complete': count, 'both_correct': both,
                      'both_correct_rate': both / count if count else None,
                      'flip_consistent': flip, 'flip_consistency_rate': flip / count if count else None},
            'comparable': comparable,
            'bootstrap_95': _bootstrap(rows, families, bootstrap_samples, seed) if comparable else None,
        }
    return {'views': summaries}
