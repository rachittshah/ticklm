import json

import pytest

from temporal_bridge import surgery_report as report


def cases():
    choices = {'A': 'wait', 'B': 'act', 'C': 'other', 'D': 'stop'}
    return [{'id': f'{p}-{i}', 'pair_id': p, 'family': family,
             'prompt': f'prompt {p}-{i}', 'answer': 'AB'[i], 'choices': choices,
             'prediction': 'A'}
            for p, family in [('p1', 'silence'), ('p2', 'deadline')] for i in range(2)]


def test_identical_and_perfect_comparisons_resample_pairs():
    base = cases()
    perfect = [{**c, 'prediction': c['answer']} for c in reversed(base)]
    same = report.compare_records(base, list(reversed(base)), seed=1103, reps=1000)
    assert same['overall']['item_accuracy']['difference'] == 0
    assert same['overall']['item_accuracy']['difference_ci_95'] == [0, 0]
    result = report.compare_records(base, perfect, seed=1103, reps=1000)
    overall = result['overall']
    assert result['pair_count'] == 2
    assert overall['item_accuracy']['base'] == .5
    assert overall['item_accuracy']['candidate'] == 1
    assert overall['item_accuracy']['difference_ci_95'] == [.5, .5]
    assert overall['pair_both_correct']['difference'] == 1
    assert overall['prediction_changed_within_pair']['difference'] == 1
    assert set(result['per_family']) == {'silence', 'deadline'}


@pytest.mark.parametrize('field,value', [('prompt', 'different'), ('answer', 'D'),
                                         ('pair_id', 'other'), ('choices', {'A': 'other',
                                          'B': 'act', 'C': 'other', 'D': 'stop'}),
                                         ('family', 'other')])
def test_rejects_mismatched_case_fields(field, value):
    base = cases()
    candidate = [dict(c) for c in base]
    candidate[0][field] = value
    with pytest.raises(ValueError, match='mismatch'):
        report.compare_records(base, candidate, seed=1, reps=20)


def test_rejects_missing_duplicate_and_malformed_pairs():
    base = cases()
    with pytest.raises(ValueError, match='IDs'):
        report.compare_records(base, base[:-1], seed=1, reps=20)
    with pytest.raises(ValueError, match='duplicate'):
        report.compare_records(base, base + [base[0]], seed=1, reps=20)
    incomplete = base[:-1]
    with pytest.raises(ValueError, match='exactly two'):
        report.compare_records(incomplete, incomplete, seed=1, reps=20)
    oversized = base + [{**base[0], 'id': 'extra'}]
    with pytest.raises(ValueError, match='exactly two'):
        report.compare_records(oversized, oversized, seed=1, reps=20)


def test_cli_records_hashes_and_refuses_overwrite(tmp_path):
    base, candidate, out = tmp_path / 'base.jsonl', tmp_path / 'candidate.jsonl', tmp_path / 'report.json'
    base.write_text(''.join(json.dumps(c) + '\n' for c in cases()))
    candidate.write_text(''.join(json.dumps(c) + '\n' for c in reversed(cases())))
    args = ['compare', '--base', str(base), '--candidate', str(candidate), '--output', str(out),
            '--seed', '9', '--bootstrap-reps', '50']
    report.main(args)
    result = json.loads(out.read_text())
    assert result['seed'] == 9 and result['bootstrap_reps'] == 50
    assert len(result['source_hashes']['base_sha256']) == 64
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        report.main(args)
    assert out.read_bytes() == original
