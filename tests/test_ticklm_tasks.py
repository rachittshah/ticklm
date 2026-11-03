from collections import Counter, defaultdict

import pytest

from ticklm_bench.tasks import audit_oracle, build_suite, deterministic_oracle


@pytest.mark.parametrize(('family', 'state', 'expected'), [
    ('duration', {'start': 10, 'now': 12.5, 'limit': 2.5}, 'alert'),
    ('duration', {'start': 10, 'now': 12.25, 'limit': 2.5}, 'wait'),
    ('policy', {'start': 10, 'now': 12.5, 'lower': 2.5, 'upper': 4, 'policy': 'less'}, 'wait'),
    ('policy', {'start': 10, 'now': 12.5, 'lower': 2.5, 'upper': 4, 'policy': 'inside'}, 'alert'),
    ('interval', {'a_start': 1, 'a_end': 3, 'b_start': 3, 'b_end': 4, 'closure': 'closed'}, 'overlap'),
    ('interval', {'a_start': 1, 'a_end': 3, 'b_start': 3, 'b_end': 4, 'closure': 'half-open'}, 'separate'),
    ('revision', {'plan': 'P', 'target': 'P', 'plan_at': 10, 'event_at': 9.5, 'now': 12, 'event': 'cancel'}, 'continue'),
    ('revision', {'plan': 'P', 'target': 'P', 'plan_at': 10, 'event_at': 10, 'now': 12, 'event': 'cancel'}, 'stop'),
    ('event_window', {'start': 10, 'end': 12, 'event_time': 12, 'arrival': 13, 'close': 13}, 'outside'),
    ('event_window', {'start': 10, 'end': 12, 'event_time': 11, 'arrival': 13, 'close': 13}, 'accept'),
    ('calendar', {'date': '2024-02-29', 'minute': 1410, 'offset': 60, 'add': 0}, 'next_day'),
    ('calendar', {'date': '2024-03-01', 'minute': 30, 'offset': -60, 'add': 0}, 'previous_day'),
    ('calendar', {'date': '2024-12-31', 'minute': 1410, 'offset': 60, 'add': 0}, 'next_day'),
    ('chronology', {'r1': 'A<B', 'r2': 'B<C'}, 'before'),
    ('chronology', {'r1': 'B<A', 'r2': 'C<B'}, 'after'),
    ('chronology', {'r1': 'A<B', 'r2': 'C<B'}, 'unknown'),
    ('chronology', {'r1': 'A<B', 'r2': 'B<A'}, 'inconsistent'),
])
def test_independent_oracles_on_hand_checked_boundaries(family, state, expected):
    case = {'family': family, 'state': state}
    assert deterministic_oracle(case) == expected
    assert audit_oracle(case) == expected


def test_default_and_screen_sizes_and_oracle_agreement():
    assert len(build_suite('dev')) == 448
    assert len(build_suite('dev', pairs_per_family=4)) == 112
    cases = build_suite('dev')
    assert cases == build_suite('dev')
    assert len({case['id'] for case in cases}) == len(cases)
    assert {case['family'] for case in cases} == {
        'duration', 'policy', 'interval', 'revision', 'event_window', 'calendar', 'chronology'}
    for case in cases:
        assert deterministic_oracle(case) == audit_oracle(case)
        assert case['choices'][case['answer']] == deterministic_oracle(case)
        assert case['id'] not in case['prompt']
        assert case['pair_id'] not in case['prompt']
        assert case['rule_id'] not in case['prompt']
        assert case['template_id'] not in case['prompt']
        assert len(case['prompt'].split()) < 500


def test_pairs_flip_one_fact_and_views_share_episode():
    cases = build_suite('dev')
    pairs = defaultdict(list)
    episodes = defaultdict(list)
    for case in cases:
        pairs[case['pair_id'], case['view']].append(case)
        episodes[case['episode_id']].append(case)
    assert len(episodes) == 112
    for pair in pairs.values():
        assert len(pair) == 2
        a, b = pair
        assert a['choices'] == b['choices']
        assert a['answer'] != b['answer']
        assert sum(a['state'][key] != b['state'][key] for key in a['state']) == 1
    for episode in episodes.values():
        assert len(episode) == 4
        assert Counter(case['view'] for case in episode) == {'raw': 2, 'derived': 2}
        for variant in (0, 1):
            raw, derived = sorted((c for c in episode if c['variant'] == variant), key=lambda c: c['view'])
            assert raw['state'] == derived['state']
            assert raw['answer'] == derived['answer']
            assert raw['choices'] == derived['choices']


def test_answer_positions_balanced_for_each_family_and_view():
    cases = build_suite('dev')
    for family in {c['family'] for c in cases}:
        for view in ('raw', 'derived'):
            assert Counter(c['answer'] for c in cases if c['family'] == family and c['view'] == view) == dict.fromkeys('ABCD', 8)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        build_suite('train')
    with pytest.raises(ValueError):
        build_suite('dev', pairs_per_family=0)


def test_family_coverage_and_feature_prefixes():
    cases = build_suite('dev')
    assert {c['state']['policy'] for c in cases if c['family'] == 'policy'} == {
        'less', 'at_least', 'inside', 'outside'}
    assert {c['state']['closure'] for c in cases if c['family'] == 'interval'} == {'closed', 'half-open'}
    assert {deterministic_oracle(c) for c in cases if c['family'] == 'chronology'} == {
        'before', 'unknown', 'inconsistent'}
    assert {deterministic_oracle(c) for c in cases if c['family'] == 'event_window'} == {
        'accept', 'late', 'outside'}
    for case in cases:
        if case['view'] == 'derived':
            feature = case['prompt'].splitlines()[1].lower()
            assert feature.startswith('arithmetic information:')
            assert all(term not in feature for term in ('true', 'false', 'alert', 'wait', 'overlap',
                                                         'accept', 'outside', 'late', 'inconsistent'))
