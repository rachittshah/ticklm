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


def test_adapter_provenance_records_hashes_and_cross_view(tmp_path):
    (tmp_path / "manifest.json").write_text('{"training_input_view": "clock-features"}')
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapters.safetensors").write_bytes(b"weights")
    details = surgery.adapter_provenance(tmp_path, "raw")
    assert details["training_input_view"] == "clock-features"
    assert details["evaluation_input_view"] == "raw"
    assert details["input_view_mismatch"] is True
    for name in ("manifest.json", "adapter_config.json", "adapters.safetensors"):
        assert details["files_sha256"][name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert surgery.adapter_provenance(tmp_path, "clock-features")["input_view_mismatch"] is False
    assert surgery.adapter_provenance(tmp_path)["input_view_mismatch"] is None
    (tmp_path / "manifest.json").write_text("{}")
    assert surgery.adapter_provenance(tmp_path, "raw")["training_input_view"] == "raw"


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


def test_broader_target_set_trains_only_lora_and_loads_strictly(tmp_path):
    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=4096, num_key_value_heads=4, max_position_embeddings=64,
                     head_dim=8, tie_word_embeddings=True, rope_theta=10000.0)
    model = Model(args)
    config, counts = surgery.configure_adapter(model, 1, 1, target_set="qv-mlp")
    names = counts["tensors"]
    assert counts["fraction"] <= .01
    assert len(names) == 10
    assert all(name.endswith(("lora_a", "lora_b")) for name in names)
    assert any("mlp.gate_proj" in name for name in names)
    assert any("mlp.up_proj" in name for name in names)
    assert any("mlp.down_proj" in name for name in names)
    surgery.save_adapter(model, tmp_path / "adapter", config, 1, {"base": "x"}, "rev", counts, 0)
    surgery.load_adapter(Model(args), tmp_path / "adapter", {"base": "x"}, "rev")
    config_path = tmp_path / "adapter" / "adapter_config.json"
    bad = json.loads(config_path.read_text())
    bad["lora_parameters"]["keys"] = ["mlp.gate_proj"]
    config_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unsupported adapter"):
        surgery.load_adapter(Model(args), tmp_path / "adapter", {"base": "x"}, "rev")


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


def test_distill_kl_is_zero_for_same_logits_positive_for_change_and_teacher_stopped():
    teacher_logits = mx.array([[1., 2., 3.], [2., 0., -1.]])
    teacher = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    same = surgery.distill_kl(teacher_logits, teacher)
    different = surgery.distill_kl(teacher_logits + mx.array([[2., 0., 0.], [0., 1., 0.]]), teacher)
    student_grad = mx.grad(lambda student: surgery.distill_kl(student, teacher))(
        teacher_logits + mx.array([[2., 0., 0.], [0., 1., 0.]]))
    teacher_grad = mx.grad(lambda target: surgery.distill_kl(teacher_logits, target))(teacher)
    mx.eval(same, different, student_grad, teacher_grad)
    assert abs(float(same)) < 1e-6
    assert float(different) > 0.01
    np.testing.assert_array_equal(np.asarray(teacher_grad), np.zeros((2, 3)))
    assert np.isfinite(np.asarray(student_grad)).all()
    assert np.max(np.abs(np.asarray(student_grad))) > .01


def test_provenance_includes_versions_source_and_saved_dataset(tmp_path):
    provenance = surgery.runtime_provenance()
    assert provenance["packages"]["mlx-lm"]
    assert len(provenance["source_sha256"]["surgery.py"]) == 64
    if (surgery.Path(surgery.__file__).parent / "surgery_features.py").exists():
        assert len(provenance["source_sha256"]["surgery_features.py"]) == 64
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


@pytest.mark.parametrize("loss,anchor_every,curriculum,paired,anchor_mode", [
    ("choice", 0, False, False, "mcq"), ("full", 0, False, False, "mcq"),
    ("choice", 2, True, True, "mcq"), ("choice", 2, True, True, "distill")])
def test_training_loop_keeps_dev_selection_and_reload_parity(
        tmp_path, monkeypatch, loss, anchor_every, curriculum, paired, anchor_mode):
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
            return ["ABCD".index(label) + 3] if len(label) == 1 and label in "ABCD" else [7, 8]
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
    views = []
    def transform(cases, view):
        views.append((cases[0]["split"], view))
        return [{**case, "prompt": case["prompt"] + " FEATURE"} for case in cases]
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
    monkeypatch.setattr(surgery, "apply_input_view", transform, raising=False)
    events = []
    real_cache = surgery.cache_distill_teacher
    real_configure = surgery.configure_adapter
    def observed_cache(*a, **kw):
        events.append("cache")
        return real_cache(*a, **kw)
    def observed_configure(*a, **kw):
        events.append("configure")
        return real_configure(*a, **kw)
    monkeypatch.setattr(surgery, "cache_distill_teacher", observed_cache)
    monkeypatch.setattr(surgery, "configure_adapter", observed_configure)
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 64}')
    model.eval()
    teacher_logits = model(mx.array([[1, 2, 7, 8]]))[0, 1:4, :].astype(mx.float32)
    expected_teacher = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    mx.eval(expected_teacher)
    cli = Namespace(model=str(tmp_path), output=str(tmp_path / "run"), seed=9,
                    pairs_per_family=3, dev_pairs_per_family=1, layers=1, rank=1, learning_rate=.01,
                    steps=2, batch_size=2, eval_batch_size=2, revision="rev", loss=loss,
                    anchor_every=anchor_every, shuffle_labels=True, curriculum=curriculum,
                    paired_batches=paired, input_view="clock-features", anchor_mode=anchor_mode,
                    target_set="qv")
    surgery.run_train(cli, model, Tokenizer(), {"base": "hash"})
    assert json.loads((tmp_path / "run" / "reload_parity.json").read_text())["passed"]
    assert (tmp_path / "run" / "checkpoint-000001" / "retention_dev_metrics.json").exists()
    checkpoint_manifest = json.loads((tmp_path / "run" / "checkpoint-000001" / "manifest.json").read_text())
    assert checkpoint_manifest["training_input_view"] == "clock-features"
    assert requested == [(("curriculum_train" if curriculum else "train"), 3), ("dev", 1)]
    logs = json.loads((tmp_path / "run" / "training_log.json").read_text())
    assert logs[0]["data_source"] == "temporal_train"
    assert logs[0]["loss_mode"] == loss
    assert logs[1]["data_source"] == ("retention_distill_train" if anchor_every and anchor_mode == "distill"
                                       else "retention_train" if anchor_every else "temporal_train")
    assert logs[1]["loss_mode"] == ("kl_distill" if anchor_every and anchor_mode == "distill"
                                     else "full" if anchor_every else loss)
    assert requested_retention == (["dev", "train"] if anchor_every and anchor_mode == "mcq" else ["dev"])
    assert views == [("train", "clock-features"), ("dev", "clock-features")]
    actual_labels = json.loads((tmp_path / "run" / "datasets" / "training_labels.json").read_text())
    train_cases = json.loads((tmp_path / "run" / "datasets" / "temporal_train.json").read_text())
    assert [r["true_answer"] for r in actual_labels] == [case["answer"] for case in train_cases]
    assert train_cases[0]["prompt"].endswith(" FEATURE")
    retention_cases = json.loads((tmp_path / "run" / "datasets" / "retention_dev.json").read_text())
    assert retention_cases[0]["prompt"] == "question"
    if anchor_mode == "distill":
        assert events[:2] == ["cache", "configure"]
        examples = json.loads((tmp_path / "run" / "datasets" / "distill_train.json").read_text())
        assert len(examples) == 8
        assert examples[0]["prediction_positions"] == [1, 2, 3]
        assert examples[0]["input_tokens"] == [1, 2, 7, 8]
        assert examples[0]["includes_post_continuation_prediction"] is True
        teacher_file = tmp_path / "run" / "distill_teacher.safetensors"
        saved = mx.load(str(teacher_file))
        np.testing.assert_allclose(np.asarray(saved["teacher_000"]), np.asarray(expected_teacher), atol=1e-6)
        run = json.loads((tmp_path / "run" / "run.json").read_text())
        assert run["distill_teacher"]["sha256"] == hashlib.sha256(teacher_file.read_bytes()).hexdigest()


def test_evaluate_command_applies_view_only_to_temporal_cases(tmp_path, monkeypatch):
    import sys

    from temporal_bridge import surgery_data

    cases = [{"id": "t", "split": "dev", "answer": "A", "prompt": "temporal"}]
    retention = [{"id": "r", "split": "dev", "answer": "A", "prompt": "retention"}]
    evaluated = []
    monkeypatch.setattr(surgery_data, "build_cases", lambda *a, **k: cases)
    monkeypatch.setattr(surgery_data, "build_retention_cases", lambda **k: retention)
    monkeypatch.setattr(surgery_data, "deterministic_answer", lambda case: "A")
    monkeypatch.setattr(surgery, "apply_input_view",
                        lambda source, view: [{**c, "prompt": c["prompt"] + " " + view} for c in source],
                        raising=False)
    monkeypatch.setattr(surgery, "base_identity", lambda path: {})
    monkeypatch.setattr(surgery, "load", lambda path: (object(), object()))
    monkeypatch.setattr(surgery, "load_adapter", lambda *args: {})
    monkeypatch.setattr(surgery, "get_total_parameters", lambda model: 1)
    monkeypatch.setattr(surgery, "checkpoint_parameter_count", lambda path: 1)
    monkeypatch.setattr(surgery, "context_limit", lambda path: 64)
    def evaluate(model, tokenizer, source, limit, batch_size):
        evaluated.append(source)
        return source, {}
    monkeypatch.setattr(surgery, "evaluate_cases", evaluate)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "manifest.json").write_text('{}')
    (adapter / "adapter_config.json").write_text('{}')
    (adapter / "adapters.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(sys, "argv", ["surgery", "evaluate", "--model", "unused", "--revision", "rev",
                                      "--output", str(tmp_path / "run"), "--input-view", "clock-features",
                                      "--adapter", str(adapter)])
    surgery.main()
    assert evaluated[0][0]["prompt"] == "temporal clock-features"
    assert evaluated[1][0]["prompt"] == "retention"
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["args"]["input_view"] == "clock-features"
    assert manifest["adapter_provenance"]["input_view_mismatch"] is True
    monkeypatch.setattr(surgery, "inspect_model", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["surgery", "inspect", "--model", "unused", "--revision", "rev",
                                      "--output", str(tmp_path / "inspect"), "--adapter", str(adapter)])
    surgery.main()
    inspected = json.loads((tmp_path / "inspect" / "run.json").read_text())["adapter_provenance"]
    assert inspected["evaluation_input_view"] is None
    assert inspected["input_view_mismatch"] is None
