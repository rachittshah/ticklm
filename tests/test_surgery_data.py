import re
from collections import Counter, defaultdict

import pytest

from temporal_bridge.surgery_data import (
    build_cases,
    build_curriculum_cases,
    build_retention_cases,
    build_robustness_cases,
    deterministic_answer,
    render_prompt,
)


@pytest.mark.parametrize(
    ('family', 'state', 'expected'),
    [
        ('silence', {'now': 12.5, 'last_event': 10, 'threshold': 2.5}, 'alert'),
        ('silence', {'now': 12.49, 'last_event': 10, 'threshold': 2.5}, 'wait'),
        ('deadline', {'now': 10.5, 'duration': 2.25, 'deadline': 12.75}, 'start'),
        ('deadline', {'now': 10.5, 'duration': 2.26, 'deadline': 12.75}, 'defer'),
        ('relevance', {'event': 'cancel', 'plan_id': 'P', 'target_id': 'P',
                       'plan_at': 10, 'event_at': 9.75}, 'continue'),
        ('relevance', {'event': 'revision', 'plan_id': 'P', 'target_id': 'P',
                       'plan_at': 10, 'event_at': 10}, 'replan'),
        ('relevance', {'event': 'cancel', 'plan_id': 'P', 'target_id': 'P',
                       'plan_at': 10, 'event_at': 10.25}, 'stop'),
        ('relevance', {'event': 'revision', 'plan_id': 'P', 'target_id': 'Q',
                       'plan_at': 10, 'event_at': 10.25}, 'continue'),
        ('sustained', {'now': 15, 'started': 10, 'reset': None, 'threshold': 5}, 'act'),
        ('sustained', {'now': 15, 'started': 10, 'reset': 12.5, 'threshold': 5}, 'monitor'),
        ('sustained', {'now': 17.5, 'started': 10, 'reset': 12.5, 'threshold': 5}, 'act'),
    ],
)
def test_oracle_respects_hand_checked_boundaries(family, state, expected):
    case = {'family': family, 'state': state, 'choices': {'A': 'decoy', 'B': expected}}
    assert deterministic_answer(case) == 'B'


def test_pairs_flip_one_input_and_preserve_choices():
    for split in ('train', 'dev', 'test'):
        cases = build_cases(split)
        pairs = defaultdict(list)
        for case in cases:
            pairs[case['pair_id']].append(case)
            assert deterministic_answer(case) == case['answer']
            assert case['choices'][case['answer']] == case['correct_action']
        assert len(pairs) == 128
        for pair in pairs.values():
            assert len(pair) == 2
            left, right = pair
            assert left['choices'] == right['choices']
            assert left['answer'] != right['answer']
            assert sum(left['state'][key] != right['state'][key] for key in left['state']) == 1


def test_splits_are_repeatable_disjoint_balanced_and_have_no_metadata_in_prompts():
    splits = {split: build_cases(split) for split in ('train', 'dev', 'test')}
    assert splits['test'] == build_cases('test')
    ids = [{case['id'] for case in cases} for cases in splits.values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
    for split, cases in splits.items():
        assert Counter(case['answer'] for case in cases) == dict.fromkeys('ABCD', 64)
        for case in cases:
            assert case['id'] not in case['prompt']
            assert case['pair_id'] not in case['prompt']
            assert not re.search(rf'\b{split}\b', case['prompt'].lower())
            assert 'correct_action' not in case['prompt']
            assert 'variant' not in case['prompt'].lower()
            assert len(case['prompt'].split()) < 250


def test_retention_is_disjoint_and_balanced():
    splits = {split: build_retention_cases(split) for split in ('train', 'dev', 'test')}
    prompts = [{case['prompt'] for case in cases} for cases in splits.values()]
    assert all(40 <= len(cases) <= 80 for cases in splits.values())
    assert all(not prompts[i] & prompts[j] for i in range(3) for j in range(i))
    for cases in splits.values():
        assert max(Counter(c['answer'] for c in cases).values()) - min(
            Counter(c['answer'] for c in cases).values()
        ) <= 1
        assert all(c['answer'] in 'ABCD' for c in cases)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        build_cases('validation')
    with pytest.raises(ValueError):
        build_cases('test', pairs_per_family=0)
    with pytest.raises(ValueError):
        build_retention_cases('validation')


def test_pair_contents_are_stable_when_other_pair_counts_change():
    short = {case['id']: case for case in build_cases('dev', pairs_per_family=4)}
    long = {case['id']: case for case in build_cases('dev', pairs_per_family=8)}
    assert all(long[case_id] == case for case_id, case in short.items())


def test_oracle_ignores_stored_labels_and_numeric_splits_are_disjoint():
    cases = build_cases('test')
    for case in cases:
        altered = {**case, 'answer': 'incorrect', 'correct_action': 'incorrect'}
        assert deterministic_answer(altered) == case['answer']
    ranges = []
    for split in ('train', 'dev', 'test'):
        thresholds = {
            case['state']['threshold'] for case in build_cases(split)
            if 'threshold' in case['state']
        }
        ranges.append(thresholds)
    assert all(not ranges[i] & ranges[j] for i in range(3) for j in range(i))


def test_margins_and_fractional_inputs_do_not_reveal_the_label():
    for split in ('train', 'dev', 'test'):
        for family in ('silence', 'deadline', 'sustained'):
            by_action = defaultdict(list)
            for case in build_cases(split):
                if case['family'] != family:
                    continue
                state = case['state']
                if family == 'silence':
                    margin = state['now'] - state['last_event'] - state['threshold']
                elif family == 'deadline':
                    margin = state['deadline'] - state['now'] - state['duration']
                else:
                    start = state['reset'] if state['reset'] is not None else state['started']
                    margin = state['now'] - start - state['threshold']
                by_action[case['correct_action']].append((margin, state['now']))
            positive = by_action[{'silence': 'alert', 'deadline': 'start', 'sustained': 'act'}[family]]
            negative = by_action[{'silence': 'wait', 'deadline': 'defer', 'sustained': 'monitor'}[family]]
            assert 0 < sum(margin == 0 for margin, _ in positive) < len(positive) / 2
            assert len({margin for margin, _ in positive if margin > 0}) >= 2
            assert len({margin for margin, _ in negative if margin < 0}) >= 2
            assert any(now % 1 for _, now in positive)
            assert any(now % 1 for _, now in negative)


def test_sustained_resets_are_balanced_across_both_actions():
    for split in ('train', 'dev', 'test'):
        counts = Counter((case['correct_action'], case['state']['reset'] is not None)
                         for case in build_cases(split) if case['family'] == 'sustained')
        assert counts == {('act', False): 16, ('act', True): 16,
                          ('monitor', False): 16, ('monitor', True): 16}


def test_relevance_requires_event_order_and_scope():
    for split in ('train', 'dev', 'test'):
        pairs = defaultdict(list)
        for case in build_cases(split):
            if case['family'] == 'relevance':
                pairs[case['pair_id']].append(case)
        changed = Counter(next(key for key in a['state'] if a['state'][key] != b['state'][key])
                          for a, b in pairs.values())
        assert changed['event_at'] >= 8
        assert changed['target_id'] >= 8
        assert {case['correct_action'] for pair in pairs.values() for case in pair} == {
            'continue', 'replan', 'stop'}


def test_body_templates_and_normalized_prompts_are_held_out():
    bodies = {}
    templates = {}
    for split in ('train', 'dev', 'test'):
        cases = build_cases(split)
        bodies[split] = {' '.join(case['prompt'].splitlines()[1].lower().split()) for case in cases}
        templates[split] = defaultdict(set)
        for case in cases:
            body = case['prompt'].splitlines()[1].lower()
            body = re.sub(r'\d+(?:\.\d+)?', '#', body)
            body = re.sub(r'\bnone\b', '# seconds', body)
            body = re.sub(r'\b(cancel|revision)\b', 'eventkind', body)
            templates[split][case['family']].add(body)
    for i, split in enumerate(('train', 'dev', 'test')):
        for earlier in ('train', 'dev', 'test')[:i]:
            assert not bodies[split] & bodies[earlier]
            for family in templates[split]:
                assert not templates[split][family] & templates[earlier][family]
    for family in templates['train']:
        assert len(templates['train'][family]) >= 3


def test_relevance_answer_letters_are_balanced_within_event_and_action():
    for split in ('train', 'dev', 'test'):
        counts = Counter((case['state']['event'], case['correct_action'], case['answer'])
                         for case in build_cases(split) if case['family'] == 'relevance')
        for event, action in (('cancel', 'continue'), ('cancel', 'stop'),
                              ('revision', 'continue'), ('revision', 'replan')):
            assert [counts[event, action, letter] for letter in 'ABCD'] == [4] * 4


def test_public_renderer_recreates_generator_prompts():
    for split in ('train', 'dev', 'test'):
        for case in build_cases(split, pairs_per_family=4):
            index = int(case['pair_id'].rsplit('-', 1)[1])
            assert render_prompt(split, case['family'], case['state'], case['choices'], index) == case['prompt']


def test_robustness_controls_preserve_primary_cases_and_pair_properties():
    original = build_cases('test')
    controls = build_robustness_cases('test')
    assert build_cases('test') == original
    assert set(controls) == {'option_rotation', 'time_translation', 'in_range_numbers'}
    assert build_robustness_cases('test') == controls
    original_ids = {case['id'] for case in original}
    seen_ids = set(original_ids)
    for name, cases in controls.items():
        assert len(cases) == len(original)
        pairs = defaultdict(list)
        for case in cases:
            assert case['id'] not in seen_ids
            seen_ids.add(case['id'])
            assert case['parent_id'] == case['sourcecase']
            assert case['control'] == name
            assert case['split'] == 'test'
            assert deterministic_answer(case) == case['answer']
            pairs[case['pair_id']].append(case)
        assert len(pairs) == 128
        assert all(len(pair) == 2 and pair[0]['choices'] == pair[1]['choices']
                   and pair[0]['answer'] != pair[1]['answer'] for pair in pairs.values())
        assert Counter(case['answer'] for case in cases) == dict.fromkeys('ABCD', 64)


def test_option_rotation_changes_only_the_letter_and_order():
    source = {case['id']: case for case in build_cases('test')}
    for case in build_robustness_cases('test')['option_rotation']:
        old = source[case['parent_id']]
        assert case['state'] == old['state']
        assert case['correct_action'] == old['correct_action']
        assert case['answer'] == 'ABCD'[('ABCD'.index(old['answer']) + 1) % 4]
        assert all(case['choices']['ABCD'[(i + 1) % 4]] == old['choices'][letter]
                   for i, letter in enumerate('ABCD'))
        assert case['prompt'] != old['prompt']


def test_translation_moves_all_absolute_times_by_same_offset():
    source = {case['id']: case for case in build_cases('test')}
    time_fields = {'now', 'last_event', 'deadline', 'plan_at', 'event_at', 'started', 'reset'}
    for case in build_robustness_cases('test')['time_translation']:
        old = source[case['parent_id']]
        assert case['answer'] == old['answer']
        assert case['choices'] == old['choices']
        for key, value in old['state'].items():
            if key in time_fields and value is not None:
                assert case['state'][key] == value + case['time_offset']
            else:
                assert case['state'][key] == value


def test_in_range_uses_heldout_wording_and_training_numeric_domain():
    train = build_cases('train')
    original = build_cases('test')
    control = build_robustness_cases('test')['in_range_numbers']
    train_bodies = {case['prompt'].splitlines()[1] for case in train}
    assert not train_bodies & {case['prompt'].splitlines()[1] for case in control}
    assert {case['state'].get('threshold') for case in control} <= {
        case['state'].get('threshold') for case in train}
    assert {case['state'].get('plan_at') for case in control if case['family'] == 'relevance'} <= {
        n / 4 for n in range(41, 121)}
    assert {case['state']['now'] for case in control if 'now' in case['state']} != {
        case['state']['now'] for case in original if 'now' in case['state']}
    assert all(case['source_split'] == 'train' and case['source_seed'] != 1103 for case in control)


def test_curriculum_preserves_labels_and_scales_each_raw_input():
    source = {case['id']: case for case in build_cases('train', pairs_per_family=128)}
    absolute = {'now', 'last_event', 'deadline', 'plan_at', 'event_at', 'started', 'reset'}
    durations = {'duration', 'threshold'}
    cases = build_curriculum_cases()
    assert len(cases) == 1024
    assert cases == build_curriculum_cases()
    assert {case['scale'] for case in cases} == {0.25, 0.5, 1, 2, 4, 8, 16}
    assert min(case['time_offset'] for case in cases) == 0
    assert max(case['time_offset'] for case in cases) == 100000
    for case in cases:
        original = source[case['sourcecase']]
        assert case['parent_id'] == original['id']
        assert case['source_split'] == case['split'] == 'train'
        assert case['control'] == 'curriculum'
        assert case['answer'] == original['answer'] == deterministic_answer(case)
        assert case['correct_action'] == original['correct_action']
        assert case['choices'] == original['choices']
        assert case['id'] != original['id']
        for key, value in original['state'].items():
            if value is None:
                expected = None
            elif key in absolute:
                expected = case['scale'] * value + case['time_offset']
            elif key in durations:
                expected = case['scale'] * value
            else:
                expected = value
            assert case['state'][key] == expected


def test_curriculum_pairs_share_transform_and_never_overlap_heldout_prompts():
    originals = {split: build_cases(split) for split in ('train', 'dev', 'test')}
    pairs = defaultdict(list)
    cases = build_curriculum_cases()
    for case in cases:
        pairs[case['pair_id']].append(case)
    assert len(pairs) == 512
    for pair in pairs.values():
        assert len(pair) == 2
        left, right = pair
        assert left['scale'] == right['scale']
        assert left['time_offset'] == right['time_offset']
        assert left['choices'] == right['choices']
        assert left['answer'] != right['answer']
        assert sum(left['state'][k] != right['state'][k] for k in left['state']) == 1
    prompts = {case['prompt'] for case in cases}
    for split in ('dev', 'test'):
        assert not prompts & {case['prompt'] for case in originals[split]}
    assert all(build_cases(split) == cases for split, cases in originals.items())


def test_curriculum_covers_small_and_large_offsets_for_each_family():
    for family in ('silence', 'deadline', 'relevance', 'sustained'):
        offsets = {case['time_offset'] for case in build_curriculum_cases()
                   if case['family'] == family}
        assert 0 in offsets and 100000 in offsets
        assert any(1 <= offset <= 99 for offset in offsets)
        assert any(100 <= offset <= 999 for offset in offsets)
        assert any(1000 <= offset <= 9999 for offset in offsets)
        assert any(10000 <= offset <= 99999 for offset in offsets)
