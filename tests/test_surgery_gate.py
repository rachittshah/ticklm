import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.lora import LoRALinear

from temporal_bridge import surgery, surgery_gate


@pytest.fixture(autouse=True)
def cpu_only():
    prior = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prior)


def tiny_model():
    args = ModelArgs(model_type='qwen3', hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                     vocab_size=4096, num_key_value_heads=4, max_position_embeddings=64,
                     head_dim=8, tie_word_embeddings=True, rope_theta=10000.0)
    mx.random.seed(9)
    model = Model(args)
    model.eval()
    return model


def test_no_adapter_fails_closed():
    with pytest.raises(ValueError, match='adapter'), surgery_gate.adapter_mode(tiny_model(), enabled=False):
        pass


def test_nested_modes_and_exception_restore_original_scales():
    model = tiny_model()
    surgery.configure_adapter(model, 1, 1)
    modules = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    modules[0].scale, modules[1].scale = 0.75, 1.25
    with surgery_gate.adapter_mode(model, enabled=False):
        assert [m.scale for m in modules] == [0, 0]
        with surgery_gate.adapter_mode(model, enabled=True):
            assert [m.scale for m in modules] == [.75, 1.25]
        assert [m.scale for m in modules] == [0, 0]
        with pytest.raises(RuntimeError, match='boom'), surgery_gate.adapter_mode(model, enabled=True):
            raise RuntimeError('boom')
        assert [m.scale for m in modules] == [0, 0]
    assert [m.scale for m in modules] == [.75, 1.25]


def test_tiny_qwen_disabled_logits_equal_base_and_active_is_distinct():
    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4]])
    base = model(tokens)
    mx.eval(base)
    original = np.array(base)
    refs = surgery_gate.base_tensor_references(model)
    surgery.configure_adapter(model, 1, 1)
    assert surgery_gate.compare_base_tensor_references(refs, model)['all_same_objects']
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_b = mx.ones_like(m.lora_b) * 0.5
    model.eval()
    active = model(tokens)
    with surgery_gate.adapter_mode(model, enabled=False):
        disabled = model(tokens)
        mx.eval(disabled)
    mx.eval(active)
    np.testing.assert_allclose(np.asarray(disabled), original, atol=1e-5, rtol=0)
    assert np.max(np.abs(np.asarray(active) - original)) > 1e-5
