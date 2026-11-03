from copy import deepcopy

import pytest

from temporal_bridge.surgery_features import with_feature_view


@pytest.mark.parametrize(
    ('family', 'state', 'expected'),
    [
        ('silence', {'now': 12.5, 'last_event': 10.0, 'threshold': 3},
         'Elapsed since last event: 2.5 seconds; silence threshold: 3 seconds.'),
        ('deadline', {'now': 10.25, 'deadline': 15.5, 'duration': 4},
         'Time remaining until deadline: 5.25 seconds; required duration: 4 seconds.'),
        ('sustained', {'now': 5, 'started': -10, 'reset': 0, 'threshold': 6},
         'Elapsed since latest reset or start: 5 seconds; sustained threshold: 6 seconds.'),
        ('sustained', {'now': 5, 'started': 1, 'reset': None, 'threshold': 6},
         'Elapsed since latest reset or start: 4 seconds; sustained threshold: 6 seconds.'),
        ('relevance', {'event': 'cancel', 'plan_id': 'P', 'target_id': 'Q',
                       'plan_at': 12, 'event_at': 11.5},
         'Event time relative to plan issue: -0.5 seconds; plan ID: P; target ID: Q; event: cancel.'),
    ],
)
def test_clock_features_are_raw_arithmetic_not_decisions(family, state, expected):
    case = {'family': family, 'state': state, 'prompt': 'Original question',
            'choices': {'A': 'one', 'B': 'two'}, 'answer': 'A'}
    before = deepcopy(case)
    result = with_feature_view([case], 'clock-features')[0]
    assert result['prompt'] == f'Clock features: {expected}\nOriginal question'
    assert {key: value for key, value in result.items() if key != 'prompt'} == {
        key: value for key, value in case.items() if key != 'prompt'}
    result['state']['extra'] = 1
    assert case == before


def test_raw_view_deep_copies_without_changing_content():
    case = {'family': 'silence', 'state': {'now': 1}, 'prompt': 'x', 'answer': 'A'}
    result = with_feature_view([case], 'raw')
    assert result == [case]
    result[0]['state']['now'] = 2
    assert case['state']['now'] == 1


@pytest.mark.parametrize(
    ('family', 'state', 'time_fields'),
    [
        ('silence', {'now': 12.5, 'last_event': 10, 'threshold': 3}, ('now', 'last_event')),
        ('deadline', {'now': 10, 'deadline': 12.5, 'duration': 3}, ('now', 'deadline')),
        ('sustained', {'now': 5, 'started': -10, 'reset': 0, 'threshold': 6},
         ('now', 'started', 'reset')),
        ('relevance', {'plan_id': 'P', 'target_id': 'Q', 'event': 'cancel',
                       'plan_at': 10, 'event_at': 9.5}, ('plan_at', 'event_at')),
    ],
)
def test_clock_features_are_invariant_to_common_clock_translation(family, state, time_fields):
    case = {'family': family, 'state': state, 'prompt': 'Original question', 'answer': 'A'}
    shifted = deepcopy(case)
    for key in time_fields:
        shifted['state'][key] += 1000
    assert with_feature_view([case], 'clock-features')[0]['prompt'] == (
        with_feature_view([shifted], 'clock-features')[0]['prompt'])


def test_clock_features_do_not_depend_on_answer_or_hidden_metadata():
    case = {'family': 'relevance', 'state': {'plan_id': 'P', 'target_id': 'P', 'event': 'revision',
                                            'plan_at': 5, 'event_at': 7},
            'prompt': 'Original question', 'answer': 'A', 'correct_action': 'one', 'id': 'first'}
    changed = {**case, 'answer': 'D', 'correct_action': 'other', 'id': 'second'}
    assert with_feature_view([case], 'clock-features')[0]['prompt'] == (
        with_feature_view([changed], 'clock-features')[0]['prompt'])


def test_invalid_view_and_family_are_rejected():
    with pytest.raises(ValueError):
        with_feature_view([], 'unsupported')
    with pytest.raises(ValueError):
        with_feature_view([{'family': 'other', 'state': {}, 'prompt': 'x'}], 'clock-features')
