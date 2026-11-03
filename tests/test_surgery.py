import hashlib
import json
import random
from argparse import Namespace

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs

from temporal_bridge import surgery


def test_right_padding_scores_actual_last_token():
    tokens, lengths = surgery.pad_tokens([[1, 2], [3, 4, 5]], 0, 4)
    assert tokens.tolist() == [[1, 2, 0], [3, 4, 5]]
    logits = mx.array([[[0, 0], [5, 1], [0, 9]], [[0, 0], [0, 0], [1, 7]]])
    np.testing.assert_array_equal(np.asarray(surgery.last_logits(logits, lengths)), [[5, 1], [1, 7]])
    with pytest.raises(ValueError, match="context"):
        surgery.pad_tokens([[1, 2, 3, 4, 5]], 0, 4)


def test_choice_labels_must_be_single_distinct_tokens():
    class Tokenizer:
        def encode(self, label, add_special_tokens=False):
            return [ord(label)]
    assert surgery.choice_ids(Tokenizer()) == [65, 66, 67, 68]
    class Bad(Tokenizer):
        def encode(self, label, add_special_tokens=False):
            return [1]
    with pytest.raises(ValueError, match="distinct"):
        surgery.choice_ids(Bad())


def test_adapter_keys_and_shapes_are_strict():
    expected = {"layer.lora_a": mx.zeros((2, 3)), "layer.lora_b": mx.zeros((3, 2))}
    surgery.validate_adapter_tensors(expected, dict(expected))
    with pytest.raises(ValueError, match="missing"):
        surgery.validate_adapter_tensors(expected, {"layer.lora_a": expected["layer.lora_a"]})
    with pytest.raises(ValueError, match="shape"):
        surgery.validate_adapter_tensors(expected, {**expected, "layer.lora_a": mx.zeros((1, 3))})


def test_split_guard_rejects_contamination():
    surgery.require_split([{"split": "train"}], "train")
    with pytest.raises(ValueError, match="split"):
        surgery.require_split([{"split": "test"}], "train")


def test_base_identity_detects_changed_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = surgery.base_identity(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert surgery.base_identity(tmp_path) != first


def test_tiny_qwen_adapter_training_and_reload(tmp_path):
    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=4096, num_key_value_heads=4, max_position_embeddings=64,
                     head_dim=8, tie_word_embeddings=True, rope_theta=10000.0)
    mx.random.seed(9)
    model = Model(args)
    base = dict(tree_flatten(model.parameters()))
    config, counts = surgery.configure_adapter(model, 1, 1)
    assert counts["fraction"] <= .01
    assert all(name.endswith(("lora_a", "lora_b")) for name in counts["tensors"])
    base_before = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())
                   if not k.endswith(("lora_a", "lora_b"))}
    def loss_fn(model):
        return nn.losses.cross_entropy(model(mx.array([[1, 2, 3]]))[:, -1, :4], mx.array([2]), reduction="mean")
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), loss)
    base_after = {k: np.asarray(v) for k, v in tree_flatten(model.parameters())
                  if not k.endswith(("lora_a", "lora_b"))}
    assert base_before.keys() == base_after.keys()
    for name, values in base_before.items():
        np.testing.assert_array_equal(values, base_after[name])
    expected = model(mx.array([[1, 2, 3]]))
    mx.eval(expected)
    surgery.save_adapter(model, tmp_path / "adapter", config, 1, {"x": "hash"}, "rev", counts, 1)
    restored = Model(args)
    restored.load_weights(list(base.items()))
    surgery.load_adapter(restored, tmp_path / "adapter", {"x": "hash"}, "rev")
    actual = restored(mx.array([[1, 2, 3]]))
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), atol=1e-5)


def test_unpaired_retention_examples_score_without_pair_id():
    class Tokenizer:
        pad_token_id = 0
        def encode(self, label, add_special_tokens=False):
            return ["ABCD".index(label)]
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]
        def decode(self, tokens):
            return "ABCD"[tokens[0]]
    class Model:
        def eval(self):
            pass
        def __call__(self, tokens):
            return mx.broadcast_to(mx.array([1., 0., 0., 0.]), (len(tokens), tokens.shape[1], 4))
    cases = [{"id": "r1", "family": "retention", "prompt": "question", "answer": "A"},
             {"id": "r2", "family": "retention", "prompt": "question", "answer": "B"}]
    records, metrics = surgery.evaluate_cases(Model(), Tokenizer(), cases, 8, 1)
    assert records[0]["correct"]
    assert metrics["pair_both_correct"] is None
    assert metrics["unrestricted_letter_validity"] == 1
    assert metrics["unrestricted_correct_letter_accuracy"] == .5
    assert metrics["peak_memory_scope"] == "process since startup"
    assert records[0]["unrestricted_correct_letter"]
    assert not records[1]["unrestricted_correct_letter"]


def test_choice_and_full_loss_use_different_normalizations():
    logits = mx.array([[1., 2., 3., 4., 9.]])
    labels = mx.array([1])
    ids = [0, 1, 2, 3]
    choice = surgery.answer_loss(logits, labels, ids, "choice")
    full = surgery.answer_loss(logits, labels, ids, "full")
    mx.eval(choice, full)
    assert float(full) > float(choice) + 3


def test_provenance_includes_versions_source_and_saved_dataset(tmp_path):
    provenance = surgery.runtime_provenance()
    assert provenance["packages"]["mlx-lm"]
    assert len(provenance["source_sha256"]["surgery.py"]) == 64
    cases = [{"id": "one", "split": "train"}]
    surgery.save_dataset(tmp_path, "train", cases)
    manifest = json.loads((tmp_path / "run.json").read_text())
    data = (tmp_path / "datasets" / "train.json").read_bytes()
    assert json.loads(data) == cases
    assert manifest["datasets"]["train"]["sha256"] == hashlib.sha256(data).hexdigest()


def test_checkpoint_header_counts_stored_tensors(tmp_path):
    mx.save_safetensors(str(tmp_path / "model.safetensors"),
                        {"one": mx.zeros((2, 3)), "two": mx.zeros((3,))})
    assert surgery.checkpoint_parameter_count(tmp_path) == 9


def test_shuffle_is_fixed_preserves_truth_and_only_accepts_train():
    cases = [{"id": str(i), "split": "train", "answer": "ABCD"[i % 4]} for i in range(24)]
    original = json.loads(json.dumps(cases))
    shuffled, labels = surgery.training_labels(cases, seed=1103, shuffle=True)
    assert cases == original
    assert shuffled == surgery.training_labels(cases, seed=1103, shuffle=True)[0]
    assert sorted(c["answer"] for c in shuffled) == sorted(c["answer"] for c in cases)
    assert any(a["answer"] != b["answer"] for a, b in zip(cases, shuffled, strict=True))
    assert all(r["true_answer"] == c["answer"] and r["training_answer"] == s["answer"]
               for r, c, s in zip(labels, cases, shuffled, strict=True))
    with pytest.raises(ValueError, match="split"):
        surgery.training_labels([{**cases[0], "split": "dev"}], 1103, True)


def test_output_reservation_refuses_nonempty_without_changes(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "run.json").write_text("keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        surgery.reserve_output(existing)
    assert (existing / "run.json").read_text() == "keep"
    fresh = tmp_path / "fresh"
    surgery.reserve_output(fresh)
    with pytest.raises(FileExistsError, match="nonempty"):
        surgery.reserve_output(fresh)


def test_paired_batch_sampling_preserves_complete_pairs():
    prepared = [({"pair_id": f"p{i}", "id": f"p{i}-{j}"}, [1])
                for i in range(4) for j in range(2)]
    groups = surgery.group_pairs(prepared)
    sample = surgery.sample_paired_batch(groups, 6, random.Random(19))
    assert len(sample) == 6
    for left, right in zip(sample[::2], sample[1::2], strict=True):
        assert left[0]["pair_id"] == right[0]["pair_id"]
        assert left[0]["id"] != right[0]["id"]
    with pytest.raises(ValueError, match="even"):
        surgery.sample_paired_batch(groups, 3, random.Random(1))
    with pytest.raises(ValueError, match="exactly two"):
        surgery.group_pairs(prepared[:-1])


def test_curriculum_pairs_are_accepted_by_real_batch_grouper():
    from temporal_bridge.surgery_data import build_curriculum_cases
    cases = build_curriculum_cases(pairs_per_family=4, seed=1103)
    groups = surgery.group_pairs([(case, [1]) for case in cases])
    assert len(groups) * 2 == len(cases)
    assert all(case["split"] == "train" for case in cases)


@pytest.mark.parametrize("loss,anchor_every,curriculum,paired", [
    ("choice", 0, False, False), ("full", 0, False, False), ("choice", 2, True, True)])
def test_training_loop_keeps_dev_selection_and_reload_parity(
        tmp_path, monkeypatch, loss, anchor_every, curriculum, paired):
    from temporal_bridge import surgery_data

    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=4096, num_key_value_heads=4, max_position_embeddings=64,
                     head_dim=8, tie_word_embeddings=True, rope_theta=10000.0)
    model = Model(args)
    base = list(tree_flatten(model.parameters()))
    class Tokenizer:
        pad_token_id = 0
        def encode(self, label, add_special_tokens=False):
            return ["ABCD".index(label) + 3]
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]
        def decode(self, tokens):
            return str(tokens[0])
    requested = []
    def cases(split, **kwargs):
        assert split in ("train", "dev")
        requested.append((split, kwargs["pairs_per_family"]))
        return [{"id": f"{split}-{i}", "pair_id": split, "split": split,
                 "family": "temporal", "prompt": "question", "answer": "A"} for i in range(2)]
    requested_retention = []
    def curriculum_cases(pairs_per_family, seed):
        requested.append(("curriculum_train", pairs_per_family))
        return [{"id": f"curriculum-{i}", "pair_id": "curriculum", "split": "train",
                 "family": "temporal", "prompt": "question", "answer": "A"} for i in range(2)]
    def retention(split):
        assert split in ("train", "dev")
        requested_retention.append(split)
        return [{"id": "retention", "split": split, "family": "retention",
                 "prompt": "question", "answer": "A"}]
    def load_base(path):
        fresh = Model(args)
        fresh.load_weights(base)
        return fresh, Tokenizer()
    monkeypatch.setattr(surgery_data, "build_cases", cases)
    monkeypatch.setattr(surgery_data, "build_curriculum_cases", curriculum_cases, raising=False)
    monkeypatch.setattr(surgery_data, "build_retention_cases", retention)
    monkeypatch.setattr(surgery, "load", load_base)
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 64}')
    cli = Namespace(model=str(tmp_path), output=str(tmp_path / "run"), seed=9,
                    pairs_per_family=3, dev_pairs_per_family=1, layers=1, rank=1, learning_rate=.01,
                    steps=2, batch_size=2, eval_batch_size=2, revision="rev", loss=loss,
                    anchor_every=anchor_every, shuffle_labels=True, curriculum=curriculum,
                    paired_batches=paired)
    surgery.run_train(cli, model, Tokenizer(), {"base": "hash"})
    assert json.loads((tmp_path / "run" / "reload_parity.json").read_text())["passed"]
    assert (tmp_path / "run" / "checkpoint-000001" / "retention_dev_metrics.json").exists()
    assert requested == [(("curriculum_train" if curriculum else "train"), 3), ("dev", 1)]
    logs = json.loads((tmp_path / "run" / "training_log.json").read_text())
    assert logs[0]["data_source"] == "temporal_train"
    assert logs[0]["loss_mode"] == loss
    assert logs[1]["data_source"] == ("retention_train" if anchor_every else "temporal_train")
    assert logs[1]["loss_mode"] == ("full" if anchor_every else loss)
    assert requested_retention == (["dev", "train"] if anchor_every else ["dev"])
    actual_labels = json.loads((tmp_path / "run" / "datasets" / "training_labels.json").read_text())
    train_cases = json.loads((tmp_path / "run" / "datasets" / "temporal_train.json").read_text())
    assert [r["true_answer"] for r in actual_labels] == [case["answer"] for case in train_cases]
