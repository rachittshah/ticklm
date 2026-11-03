import hashlib
import json
from collections import Counter, defaultdict

import pytest

from temporal_bridge import surgery_novel as novel


@pytest.mark.parametrize(('family', 'state', 'expected'), [
    ('novel_familiar_control', {'now': 12.49, 'last_event': 10, 'threshold': 2.5}, 'wait'),
    ('novel_familiar_control', {'now': 12.5, 'last_event': 10, 'threshold': 2.5}, 'alert'),
    ('novel_inverted', {'now': 12.49, 'last_event': 10, 'threshold': 2.5}, 'alert'),
    ('novel_inverted', {'now': 12.5, 'last_event': 10, 'threshold': 2.5}, 'wait'),
    ('novel_inverted', {'now': 12.75, 'last_event': 10, 'threshold': 2.5}, 'wait'),
    ('novel_window', {'now': 12.49, 'last_event': 10, 'threshold': 2.5, 'upper': 4}, 'wait'),
    ('novel_window', {'now': 12.5, 'last_event': 10, 'threshold': 2.5, 'upper': 4}, 'alert'),
    ('novel_window', {'now': 14, 'last_event': 10, 'threshold': 2.5, 'upper': 4}, 'alert'),
    ('novel_window', {'now': 14.25, 'last_event': 10, 'threshold': 2.5, 'upper': 4}, 'wait'),
])
def test_rule_boundaries(family, state, expected):
    assert novel.rule_action(family, state) == expected


def test_pairs_are_balanced_and_change_only_clock_input():
    cases = novel.build_novel_cases()
    assert cases == novel.build_novel_cases()
    assert len(cases) == 192
    for family in ('novel_inverted', 'novel_window', 'novel_familiar_control'):
        group = [c for c in cases if c['family'] == family]
        assert Counter(c['answer'] for c in group) == dict.fromkeys('ABCD', 16)
        pairs = defaultdict(list)
        for case in group:
            assert case['choices'][case['answer']] == novel.rule_action(family, case['state'])
            pairs[case['pair_id']].append(case)
        assert len(pairs) == 32
        for left, right in pairs.values():
            assert left['choices'] == right['choices']
            assert left['answer'] != right['answer']
            assert [k for k in left['state'] if left['state'][k] != right['state'][k]] == ['now']


def test_window_pairs_cross_both_boundaries_and_include_equality():
    pairs = defaultdict(list)
    for case in novel.build_novel_cases():
        if case['family'] == 'novel_window':
            pairs[case['pair_id']].append(case)
    edges = Counter()
    for pair in pairs.values():
        low = pair[0]['state']['threshold']
        high = pair[0]['state']['upper']
        elapsed = [c['state']['now'] - c['state']['last_event'] for c in pair]
        edges['lower' if min(elapsed) < low else 'upper'] += 1
        edges['equal'] += any(e in (low, high) for e in elapsed)
    assert edges['lower'] == edges['upper'] == 16
    assert edges['equal'] >= 4


def test_features_reuse_formatter_without_contradictory_old_rule_or_metadata_leaks():
    raw = novel.build_novel_cases()
    featured = novel.clock_feature_cases(raw)
    assert all(not c['prompt'].startswith('Clock features: ') for c in raw)
    for before, after in zip(raw, featured, strict=True):
        assert after['family'] == before['family']
        assert after['prompt'].endswith(before['prompt'])
        prefix = after['prompt'].splitlines()[0]
        assert prefix.startswith('Clock features: Elapsed since last event: ')
        assert 'silence threshold:' in prefix
        assert 'correct' not in prefix.lower() and 'alert' not in prefix.lower()
        assert after['id'] not in after['prompt']
        assert 'wait below' not in after['prompt'].lower()
        assert 'alert at or above' not in after['prompt'].lower()
        assert after['choices'] == before['choices'] and after['answer'] == before['answer']


def test_evaluate_cli_saves_both_views_metrics_and_adapter_identity(tmp_path, monkeypatch):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in ('manifest.json', 'adapter_config.json', 'adapters.safetensors'):
        (adapter / name).write_bytes(name.encode())
    observed = []
    monkeypatch.setattr(novel, 'base_identity', lambda model: {'base': 'hash'})
    monkeypatch.setattr(novel, 'load', lambda model: ('model', 'tokenizer'))
    monkeypatch.setattr(novel, 'load_adapter', lambda model, path, identity, revision:
                        observed.append((model, path, identity, revision)) or {'count': 1})
    monkeypatch.setattr(novel, 'runtime_provenance', lambda: {'source_sha256': {}})
    monkeypatch.setattr(novel, 'context_limit', lambda model: 1000)
    def evaluate(model, tokenizer, cases, limit, batch_size):
        assert all(c['prompt'].startswith('Clock features: ') for c in cases)
        records = [{**c, 'prediction': c['answer'], 'correct': True} for c in cases]
        return records, {'accuracy': 1.0}
    monkeypatch.setattr(novel, 'evaluate_cases', evaluate)
    out = tmp_path / 'run'
    novel.main(['evaluate', '--model', 'unused', '--revision', 'r', '--adapter', str(adapter),
                '--output', str(out), '--pairs-per-rule', '4'])
    assert observed == [('model', str(adapter), {'base': 'hash'}, 'r')]
    run = json.loads((out / 'run.json').read_text())
    assert run['adapter_hashes']['manifest.json'] == hashlib.sha256(b'manifest.json').hexdigest()
    for name in ('novel_raw', 'novel_clock_features'):
        assert run['datasets'][name]['sha256'] == hashlib.sha256(
            (out / run['datasets'][name]['path']).read_bytes()).hexdigest()
    assert json.loads((out / 'metrics.json').read_text())['accuracy'] == 1.0
    assert len((out / 'cases.jsonl').read_text().splitlines()) == 24
    assert set(json.loads((out / 'metrics.json').read_text())['per_family_pair_both_correct']) == {
        'novel_inverted', 'novel_window', 'novel_familiar_control'}


def test_each_rule_includes_fractional_elapsed_values_for_both_actions():
    for family in ('novel_inverted', 'novel_window', 'novel_familiar_control'):
        group = [case for case in novel.build_novel_cases() if case['family'] == family]
        for action in ('alert', 'wait'):
            elapsed = [case['state']['now'] - case['state']['last_event']
                       for case in group if case['correct_action'] == action]
            assert any(value % 1 for value in elapsed)
            assert len(set(elapsed)) > 1
    inverted = [case for case in novel.build_novel_cases() if case['family'] == 'novel_inverted']
    assert any(case['state']['now'] - case['state']['last_event'] == case['state']['threshold']
               and case['correct_action'] == 'wait' for case in inverted)


def test_familiar_cases_match_inverted_inputs_and_reverse_the_gold():
    cases = novel.build_novel_cases()
    original = {case['id']: case for case in cases if case['family'] == 'novel_inverted'}
    familiar = [case for case in cases if case['family'] == 'novel_familiar_control']
    assert len(familiar) == len(original) == 64
    for case in familiar:
        parent = original[case['parent_id']]
        assert case['parent_pair_id'] == parent['pair_id']
        assert case['pair_id'] != parent['pair_id']
        assert case['state'] == parent['state']
        assert case['choices'] == parent['choices']
        assert case['variant'] == parent['variant']
        assert case['answer'] != parent['answer']
        assert case['correct_action'] != parent['correct_action']
        assert case['prompt'] == parent['prompt'].replace(
            'strictly less than', 'greater than or equal to')
        assert case['family'] not in case['prompt']
