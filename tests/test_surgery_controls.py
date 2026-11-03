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
