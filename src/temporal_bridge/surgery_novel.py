"""Predeclared zero-shot rule diagnostic with external clock arithmetic.

These synthetic cases are additional evaluation only, not training or model selection.
Run with ``python -m temporal_bridge.surgery_novel evaluate ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from time import perf_counter

from mlx_lm import load

from .surgery import (
    atomic_json,
    base_identity,
    context_limit,
    evaluate_cases,
    load_adapter,
    reserve_output,
    runtime_provenance,
    save_dataset,
    write_records,
)
from .surgery_features import with_feature_view

_ACTIONS = {'alert': 'Raise an alert now', 'wait': 'Keep waiting',
            'restart': 'Restart the clock now', 'cancel': 'Cancel monitoring'}


def rule_action(family, state):
    """Compute the rule's action from raw times and bounds."""
    elapsed = state['now'] - state['last_event']
    if family == 'novel_inverted':
        return 'alert' if elapsed < state['threshold'] else 'wait'
    if family == 'novel_familiar_control':
        return 'alert' if elapsed >= state['threshold'] else 'wait'
    if family == 'novel_window':
        if state['threshold'] > state['upper']:
            raise ValueError('lower bound exceeds upper bound')
        return 'alert' if state['threshold'] <= elapsed <= state['upper'] else 'wait'
    raise ValueError(f'Unknown rule: {family}')


def _body(family, state, choices):
    if family in ('novel_inverted', 'novel_familiar_control'):
        comparison = ('strictly less than' if family == 'novel_inverted' else
                      'greater than or equal to')
        policy = (f'Raise an alert if elapsed silence is {comparison} the threshold; '
                  'otherwise keep waiting.')
        bounds = f"Silence threshold: {state['threshold']} seconds."
    else:
        policy = ('Raise an alert if elapsed silence is between the lower and upper bounds, '
                  'including both bounds; otherwise keep waiting.')
        bounds = f"Lower bound: {state['threshold']} seconds. Upper bound: {state['upper']} seconds."
    options = '\n'.join(f'{letter}. {_ACTIONS[action]}' for letter, action in choices.items())
    return (f'Choose the required action. Reply with its letter only. {policy} '
            f"Last event: {state['last_event']} seconds. Current time: {state['now']} seconds. "
            f'{bounds}\n{options}')


def build_novel_cases(pairs_per_rule=32, seed=1103):
    """Build raw, paired diagnostic cases with balanced labels at the default size."""
    if not isinstance(pairs_per_rule, int) or pairs_per_rule <= 0:
        raise ValueError('pairs_per_rule must be positive')
    cases = []
    for family in ('novel_inverted', 'novel_window'):
        for index in range(pairs_per_rule):
            rng = random.Random(f'{seed}:{family}:{index}:novel-v1')
            last_event = rng.randrange(400, 800) / 4
            threshold = rng.randrange(16, 40) / 4
            delta = rng.choice((0.25, 0.5, 0.75, 1.0))
            common = {'last_event': last_event, 'threshold': threshold}
            if family == 'novel_inverted':
                elapsed = (threshold - delta, threshold + (0 if index % 2 else delta))
            else:
                upper = threshold + rng.randrange(8, 20) / 4
                common['upper'] = upper
                inside_offset = 0 if index % 4 < 2 else delta
                elapsed = ((threshold - delta, threshold + inside_offset) if index % 2 == 0 else
                           (upper - inside_offset, upper + delta))
            states = [{**common, 'now': last_event + value} for value in elapsed]
            positions = list(range(4))
            random.Random(f'{seed}:{family}:{index // 4}:novel-positions').shuffle(positions)
            pos_alert = positions[index % 4]
            pos_wait = positions[(index + 1 + index // 4 % 3) % 4]
            ordered = [None] * 4
            ordered[pos_alert], ordered[pos_wait] = 'alert', 'wait'
            distractors = ['restart', 'cancel']
            rng.shuffle(distractors)
            for slot in range(4):
                if ordered[slot] is None:
                    ordered[slot] = distractors.pop()
            choices = dict(zip('ABCD', ordered, strict=True))
            if rng.randrange(2):
                states.reverse()
            pair_id = f'{family}-{index:05d}'
            for variant, state in enumerate(states):
                action = rule_action(family, state)
                answer = next(letter for letter, choice in choices.items() if choice == action)
                cases.append({'id': f'{pair_id}-{variant}', 'pair_id': pair_id, 'variant': variant,
                              'family': family, 'split': 'novel', 'state': state,
                              'choices': choices.copy(), 'answer': answer, 'correct_action': action,
                              'prompt': _body(family, state, choices)})
    for parent in [case for case in cases if case['family'] == 'novel_inverted']:
        family = 'novel_familiar_control'
        pair_id = parent['pair_id'].replace('novel_inverted', family, 1)
        state, choices = parent['state'].copy(), parent['choices'].copy()
        action = rule_action(family, state)
        answer = next(letter for letter, choice in choices.items() if choice == action)
        cases.append({**parent, 'id': f'{pair_id}-{parent["variant"]}', 'pair_id': pair_id,
                      'parent_id': parent['id'], 'parent_pair_id': parent['pair_id'],
                      'family': family, 'state': state, 'choices': choices,
                      'answer': answer, 'correct_action': action,
                      'prompt': _body(family, state, choices)})
    return cases


def clock_feature_cases(cases):
    """Use the established silence feature renderer without its original task rule."""
    prepared = [{**case, 'family': 'silence'} for case in cases]
    featured = with_feature_view(prepared, 'clock-features')
    for source, case in zip(cases, featured, strict=True):
        case['family'] = source['family']
    return featured


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs-per-rule', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1103)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if args.pairs_per_rule <= 0 or args.batch_size <= 0:
        parser.error('pairs-per-rule and batch-size must be positive')
    started = perf_counter()
    reserve_output(args.output)
    identity = base_identity(args.model)
    model, tokenizer = load(args.model)
    adapter_counts, adapter_hashes = None, None
    if args.adapter:
        adapter_counts = load_adapter(model, args.adapter, identity, args.revision)
        adapter_hashes = {name: _file_hash(Path(args.adapter) / name)
                          for name in ('manifest.json', 'adapter_config.json', 'adapters.safetensors')}
    provenance = runtime_provenance()
    provenance['source_sha256']['surgery_novel.py'] = _file_hash(__file__)
    out = Path(args.output)
    atomic_json(out / 'run.json', {'args': vars(args), 'base_hashes': identity,
                'revision_provenance': 'caller supplied', 'runtime': provenance,
                'adapter_parameters': adapter_counts, 'adapter_hashes': adapter_hashes,
                'scope': 'Extra zero-shot diagnostic; not training or model selection.'})
    raw = build_novel_cases(args.pairs_per_rule, args.seed)
    cases = clock_feature_cases(raw)
    save_dataset(out, 'novel_raw', raw)
    save_dataset(out, 'novel_clock_features', cases)
    records, metrics = evaluate_cases(model, tokenizer, cases, context_limit(args.model), args.batch_size)
    groups = {}
    for record in records:
        groups.setdefault((record['family'], record['pair_id']), []).append(record['correct'])
    metrics['per_family_pair_both_correct'] = {
        family: sum(all(result) for (kind, _), result in groups.items() if kind == family) /
        sum(1 for kind, _ in groups if kind == family)
        for family in sorted({kind for kind, _ in groups})}
    metrics['scope'] = 'Extra zero-shot diagnostic using external clock arithmetic.'
    write_records(out / 'cases.jsonl', records)
    atomic_json(out / 'metrics.json', metrics)
    atomic_json(out / 'timing.json', {'total_wall_s': perf_counter() - started})


if __name__ == '__main__':
    main()
