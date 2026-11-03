import json
from argparse import Namespace

import mlx.core as mx
import numpy as np
import pytest

from temporal_bridge import surgery_controls as controls


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_distribution_sums_match_reference_and_ignore_masked_tokens():
    base = np.array([[2., 0., -1.], [1., 3., -2.], [100., -100., -100.]], dtype=np.float32)
    adapted = np.array([[0., 1., -1.], [1., 3., -2.], [-100., 100., -100.]], dtype=np.float32)
    targets = np.array([0, 1, 2], dtype=np.int32)
    result = controls.distribution_sums(mx.array(base), mx.array(adapted), mx.array(targets),
                                        mx.array([True, True, False]))
    mx.eval(*result)
    def logsoftmax(x):
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    b, a = logsoftmax(base[:2]), logsoftmax(adapted[:2])
    expected_kl = (np.exp(b) * (b - a)).sum()
    assert float(result[0]) == pytest.approx(float(-b[np.arange(2), targets[:2]].sum()), abs=1e-6)
    assert float(result[1]) == pytest.approx(float(-a[np.arange(2), targets[:2]].sum()), abs=1e-6)
    assert float(result[2]) == pytest.approx(float(expected_kl), abs=1e-6)
    assert int(result[3]) == 2


def test_identical_distributions_have_zero_kl():
    logits = mx.array([[.1, .2], [-.4, .7]])
    result = controls.distribution_sums(logits, logits, mx.array([0, 1]), mx.array([True, True]))
    mx.eval(*result)
    assert float(result[2]) == pytest.approx(0., abs=1e-7)
    assert float(result[0]) == pytest.approx(float(result[1]))


def test_retention_passages_are_disjoint_and_split_guarded():
    dev = controls.retention_passages('dev')
    test = controls.retention_passages('test')
    assert dev and test and not set(dev).intersection(test)
    with pytest.raises(ValueError):
        controls.retention_passages('train')


def test_output_refuses_existing_nonempty_directory(tmp_path):
    (tmp_path / 'keep').write_text('original')
    with pytest.raises(FileExistsError):
        controls.reserve_output(tmp_path)
    assert (tmp_path / 'keep').read_text() == 'original'


def test_retention_scores_each_noninitial_token_once(tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [0, 1, 2] if text == 'short' else [0, 1, 2, 1]

    class Model:
        def eval(self):
            pass

        def __call__(self, inputs):
            return mx.broadcast_to(mx.array([0., 1., 2.]), (1, inputs.shape[1], 3))

    monkeypatch.setattr(controls, 'retention_passages', lambda split: ('short', 'long'))
    monkeypatch.setattr(controls, 'context_limit', lambda path: 16)
    args = Namespace(model='unused', split='dev', adapter=None, output=str(tmp_path))
    controls.retention(Model(), Tokenizer(), args, {})
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    records = [json.loads(line) for line in (tmp_path / 'passages.jsonl').read_text().splitlines()]
    assert metrics['scored_tokens'] == 5
    assert [r['scored_tokens'] for r in records] == [2, 3]
    assert metrics['mean_kl_base_to_adapted'] == pytest.approx(0.)
    assert metrics['base_cross_entropy'] == pytest.approx(
        sum(r['base_cross_entropy'] * r['scored_tokens'] for r in records) / 5)


def test_probe_learns_separable_training_features_without_using_dev_statistics():
    train = np.array([[4, 1], [3, 0], [-4, 1], [-3, 0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    head = controls.fit_probe(train, labels, steps=300, learning_rate=.03, l2=.01)
    np.testing.assert_allclose(head['mean'], train.mean(axis=0))
    assert (controls.probe_predict(head, train).argmax(axis=1) == labels).all()
    # Inference must use the saved training statistics, even on a shifted set.
    shifted = train + 100
    expected = ((shifted - head['mean']) / head['scale']) @ head['weight'] + head['bias']
    np.testing.assert_allclose(controls.probe_predict(head, shifted), expected)


def test_probe_extracts_each_actual_last_prompt_position():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if messages[-1]['content'] == 'short' else [1, 2, 3]

    class Model:
        def eval(self):
            pass
        def model(self, tokens):
            return mx.stack([tokens, tokens * 2], axis=-1)

    cases = [{'id': 'a', 'prompt': 'short', 'answer': 'A'},
             {'id': 'b', 'prompt': 'long', 'answer': 'B'}]
    features = controls.extract_probe_features(Model(), Tokenizer(), cases, 8, 2)
    np.testing.assert_array_equal(features, [[2, 4], [3, 6]])


def test_probe_test_split_requires_saved_head_before_output_or_model_load(tmp_path):
    with pytest.raises(SystemExit):
        controls.main(['probe', '--split', 'test', '--model', 'unused', '--revision', 'r',
                       '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_probe_saves_head_and_reuses_it_for_held_out_split(tmp_path, monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return [1, int(messages[-1]['content']) + 2]

    class Model:
        def freeze(self):
            pass
        def eval(self):
            pass
        def model(self, tokens):
            return mx.stack([tokens, tokens * 2], axis=-1)

    def cases(split, pairs_per_family, seed):
        return [{'id': f'{split}-{i}', 'pair_id': f'{split}-{i // 2}', 'split': split,
                 'family': 'sample', 'prompt': str(i), 'answer': 'ABCD'[i]}
                for i in range(4)]

    monkeypatch.setattr(controls, 'build_cases', cases)
    monkeypatch.setattr(controls, 'context_limit', lambda path: 16)
    dev, test = tmp_path / 'dev', tmp_path / 'test'
    dev.mkdir()
    test.mkdir()
    args = Namespace(model='unused', revision='rev', seed=3, output=str(dev), split='dev',
                     batch_size=2, pairs_per_family=1, train_pairs_per_family=1, probe_head=None)
    controls.probe(Model(), Tokenizer(), args, {'base': 'hash'})
    before = (dev / 'head.npz').read_bytes()
    args.output, args.split, args.probe_head = str(test), 'test', str(dev / 'head.npz')
    controls.probe(Model(), Tokenizer(), args, {'base': 'hash'})
    assert (dev / 'head.npz').read_bytes() == before
    assert json.loads((test / 'metrics.json').read_text())['complete_pairs'] == 2
    with pytest.raises(ValueError, match='does not match'):
        controls.probe(Model(), Tokenizer(), args, {'base': 'different'})


def test_robustness_applies_clock_view_to_primary_and_each_control(tmp_path, monkeypatch):
    seen = []

    def evaluate(model, tokenizer, cases, limit, batch_size):
        seen.append(cases)
        records = [{**case, 'prediction': case['answer'], 'correct': True} for case in cases]
        return records, {'accuracy': 1.0}

    monkeypatch.setattr(controls, 'evaluate_cases', evaluate)
    monkeypatch.setattr(controls, 'context_limit', lambda path: 1000)
    args = Namespace(model='unused', split='dev', output=str(tmp_path), seed=1103,
                     pairs_per_family=1, batch_size=2, input_view='clock-features')
    controls.robustness(None, None, args)
    assert len(seen) == 4
    assert all(case['prompt'].startswith('Clock features: ') for cases in seen for case in cases)
    for name in ('primary', 'option_rotation', 'time_translation', 'in_range_numbers'):
        saved = json.loads((tmp_path / 'datasets' / f'robustness_{name}.json').read_text())
        assert saved and all(case['prompt'].startswith('Clock features: ') for case in saved)


@pytest.mark.parametrize('command', ['retention'])
def test_nonraw_view_is_rejected_before_output_or_model_load(tmp_path, command):
    with pytest.raises(SystemExit):
        controls.main([command, '--input-view', 'clock-features', '--model', 'unused',
                       '--revision', 'r', '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_probe_applies_feature_view_to_train_and_dev_and_checks_head_view(tmp_path, monkeypatch):
    extracted = []
    def cases(split, pairs_per_family, seed):
        return [{'id': f'{split}-{i}', 'pair_id': f'{split}-{i // 2}', 'split': split,
                 'family': 'silence', 'state': {'now': 12.5, 'last_event': 10, 'threshold': 3},
                 'prompt': f'question {i}', 'answer': 'ABCD'[i]} for i in range(4)]
    def extract(model, tokenizer, examples, limit, batch_size):
        extracted.append(examples)
        return np.array([[float(i), float(i * i)] for i in range(len(examples))], dtype=np.float32)
    class Model:
        def freeze(self):
            pass
    monkeypatch.setattr(controls, 'build_cases', cases)
    monkeypatch.setattr(controls, 'extract_probe_features', extract)
    monkeypatch.setattr(controls, 'context_limit', lambda path: 16)
    dev = tmp_path / 'dev'
    dev.mkdir()
    args = Namespace(model='unused', revision='rev', seed=3, output=str(dev), split='dev',
                     batch_size=2, pairs_per_family=1, train_pairs_per_family=1, probe_head=None,
                     input_view='clock-features')
    controls.probe(Model(), None, args, {'base': 'hash'})
    assert [batch[0]['split'] for batch in extracted] == ['train', 'dev']
    assert all(c['prompt'].startswith('Clock features: ') for batch in extracted for c in batch)
    for split in ('train', 'dev'):
        data = json.loads((dev / 'datasets' / f'probe_{split}.json').read_text())
        assert all(c['prompt'].startswith('Clock features: ') for c in data)
    manifest = json.loads((dev / 'head_manifest.json').read_text())
    assert manifest['input_view'] == 'clock-features'
    args.probe_head = str(dev / 'head.npz')
    args.output = str(tmp_path / 'reuse')
    (tmp_path / 'reuse').mkdir()
    controls.probe(Model(), None, args, {'base': 'hash'})
    args.input_view = 'raw'
    with pytest.raises(ValueError, match='view'):
        controls.probe(Model(), None, args, {'base': 'hash'})


def test_legacy_probe_manifest_defaults_to_raw_and_rejects_clock_features(tmp_path, monkeypatch):
    class Model:
        def freeze(self):
            pass
    monkeypatch.setattr(controls, 'build_cases', lambda split, pairs_per_family, seed: [
        {'id': f'{split}-{i}', 'pair_id': f'{split}-{i // 2}', 'split': split, 'family': 'fixture',
         'prompt': f'question {i}', 'answer': 'ABCD'[i]} for i in range(4)])
    monkeypatch.setattr(controls, 'extract_probe_features', lambda model, tokenizer, cases, limit, batch_size:
                        np.array([[float(i), float(i * i)] for i in range(len(cases))], dtype=np.float32))
    monkeypatch.setattr(controls, 'context_limit', lambda path: 16)
    dev = tmp_path / 'dev'
    dev.mkdir()
    args = Namespace(model='unused', revision='rev', seed=3, output=str(dev), split='dev',
                     batch_size=2, pairs_per_family=1, train_pairs_per_family=1, probe_head=None,
                     input_view='raw')
    controls.probe(Model(), None, args, {'base': 'hash'})
    manifest_path = dev / 'head_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest.pop('input_view') == 'raw'
    manifest_path.write_text(json.dumps(manifest))
    args.probe_head = str(dev / 'head.npz')
    args.output = str(tmp_path / 'reuse')
    (tmp_path / 'reuse').mkdir()
    controls.probe(Model(), None, args, {'base': 'hash'})
    args.input_view = 'clock-features'
    with pytest.raises(ValueError, match='view'):
        controls.probe(Model(), None, args, {'base': 'hash'})


def test_probe_cli_accepts_clock_features_before_output_reservation(tmp_path, monkeypatch):
    def stop(path):
        raise RuntimeError('accepted by parser')
    monkeypatch.setattr(controls, 'reserve_output', stop)
    with pytest.raises(RuntimeError, match='accepted by parser'):
        controls.main(['probe', '--input-view', 'clock-features', '--model', 'unused',
                       '--revision', 'r', '--output', str(tmp_path / 'out')])
