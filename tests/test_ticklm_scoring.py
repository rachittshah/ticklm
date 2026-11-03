import pytest

from ticklm_bench.scoring import parse_answer, summarize


@pytest.mark.parametrize(('text', 'answer'), [
    (' A ', 'A'), ('D.', 'D'), ('Answer: B', 'B'), ('final: C!', 'C'),
    ('{"answer": "A"}', 'A'), (' { "answer" : "D" } ', 'D'),
    ('a', None), ('The answer is A', None), ('A B', None),
    ('<think>x</think>A', None), ('Answer: A or B', None),
    ('{"answer":"A", "extra":1}', None), ('{"answer":"A","answer":"B"}', None),
    ('["A"]', None), ('{"answer":"a"}', None), ('', None), (None, None),
])
def test_parser_accepts_only_complete_permitted_formats(text, answer):
    assert parse_answer(text) == answer


def _fixtures():
    cases = []
    for family, episode in [('first', 'p1'), ('first', 'p2'), ('second', 'p3')]:
        for view in ('raw', 'derived'):
            for variant, answer in enumerate(('A', 'B')):
                cases.append({'id': f'{episode}-{view}-{variant}', 'episode_id': episode,
                              'pair_id': episode, 'family': family, 'view': view,
                              'variant': variant, 'answer': answer,
                              'choices': {'A': 'one', 'B': 'two', 'C': 'three', 'D': 'four'}})
    raw_answers = {'p1': ('A', 'B'), 'p2': ('C', 'D'), 'p3': ('A', 'bad output')}
    records = []
    for case in cases:
        text = raw_answers[case['episode_id']][case['variant']] if case['view'] == 'raw' else case['answer']
        records.append({'case': case, 'result': {'status': 'ok', 'text': text}})
    return cases, records


def test_summarize_separates_views_and_scores_flips_and_invalid():
    cases, records = _fixtures()
    summary = summarize(records, expected_cases=cases, bootstrap_samples=200, seed=1103)
    assert set(summary['views']) == {'raw', 'derived'}
    raw, derived = summary['views']['raw'], summary['views']['derived']
    assert raw['overall']['correct'] == 3 and raw['overall']['total'] == 6
    assert raw['overall']['invalid'] == 1
    assert raw['macro_accuracy'] == pytest.approx(.5)
    assert raw['pairs']['complete'] == 3
    assert raw['pairs']['both_correct'] == 1
    assert raw['pairs']['flip_consistent'] == 2
    assert raw['comparable'] is True
    assert derived['overall']['accuracy'] == 1
    assert derived['pairs']['both_correct'] == 3
    assert raw['bootstrap_95']['macro_accuracy'] == summarize(
        records, expected_cases=cases, bootstrap_samples=200, seed=1103)['views']['raw']['bootstrap_95']['macro_accuracy']
    assert raw['bootstrap_95']['macro_accuracy'][0] <= .5 <= raw['bootstrap_95']['macro_accuracy'][1]


def test_missing_and_failed_are_coverage_failures_not_quality_answers():
    cases, records = _fixtures()
    records = [r for r in records if r['case']['id'] != 'p1-raw-0']
    next(r for r in records if r['case']['id'] == 'p2-raw-0')['result'] = {'status': 'timeout', 'text': 'A'}
    raw = summarize(records, expected_cases=cases)['views']['raw']
    assert raw['coverage']['expected'] == 6
    assert raw['coverage']['missing'] == 1
    assert raw['coverage']['failed'] == 1
    assert raw['coverage']['completed'] == 4
    assert raw['overall']['total'] == 4
    assert raw['comparable'] is False
    assert raw['bootstrap_95'] is None
    assert raw['pairs']['complete'] == 1


def test_unknown_denominator_cannot_claim_complete_coverage():
    _, records = _fixtures()
    raw = summarize(records)['views']['raw']
    assert raw['coverage']['expected'] is None
    assert raw['comparable'] is False


def test_duplicate_unknown_or_mismatched_predictions_fail_closed():
    cases, records = _fixtures()
    with pytest.raises(ValueError, match='duplicate'):
        summarize(records + [records[0]], expected_cases=cases)
    with pytest.raises(ValueError, match='unknown'):
        summarize(records, expected_cases=cases[:-1])
    records[0]['predicted'] = 'D'
    with pytest.raises(ValueError, match='parser'):
        summarize(records, expected_cases=cases)


def test_rejects_partial_expected_pair_and_cross_family_episode():
    cases, records = _fixtures()
    truncated = [c for c in cases if c['id'] != 'p1-raw-0']
    records = [r for r in records if r['case']['id'] != 'p1-raw-0']
    assert summarize(records, expected_cases=truncated)['views']['raw']['comparable'] is False
    broken = [dict(c) for c in cases]
    next(c for c in broken if c['id'] == 'p1-derived-0')['family'] = 'second'
    with pytest.raises(ValueError, match='episode'):
        summarize([], expected_cases=broken)


def test_case_fingerprint_is_order_invariant_and_changes_with_content():
    cases, records = _fixtures()
    original = summarize(records, expected_cases=cases, bootstrap_samples=20)
    reordered = summarize(list(reversed(records)), expected_cases=list(reversed(cases)), bootstrap_samples=20)
    for view in ('raw', 'derived'):
        assert original['views'][view]['case_set_sha256'] == reordered['views'][view]['case_set_sha256']
    changed_cases = [dict(case) for case in cases]
    changed_cases[0]['prompt'] = 'changed prompt'
    changed_records = [{'case': changed_cases[i], 'result': record['result']}
                       for i, record in enumerate(records)]
    changed = summarize(changed_records, expected_cases=changed_cases, bootstrap_samples=20)
    assert original['views']['raw']['case_set_sha256'] != changed['views']['raw']['case_set_sha256']
    assert summarize(records)['views']['raw']['case_set_sha256'] is None


def test_invalid_expected_pair_structure_cannot_be_comparable():
    cases, records = _fixtures()
    altered = [dict(case) for case in cases]
    next(case for case in altered if case['id'] == 'p1-raw-1')['answer'] = 'A'
    updated = [{'case': altered[i], 'result': record['result']} for i, record in enumerate(records)]
    view = summarize(updated, expected_cases=altered)['views']['raw']
    assert view['coverage']['complete'] is True
    assert view['comparable'] is False
    assert view['bootstrap_95'] is None
