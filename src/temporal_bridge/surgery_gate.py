"""Explicitly gate a LoRA adapter on a single shared MLX model.

Every use of a shared model must go through this context, including ordinary
language requests. Do not share it with code that bypasses the gate or with
asynchronous tasks that can interleave contexts on the same thread.
Use a fresh attention cache for each request; never carry a cache across modes.
Materialize lazy MLX results before leaving the context, and do not reuse compiled
graphs that captured a different adapter scale.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

from .surgery import (
    CONCISE_SYSTEM,
    GENERATION_PROMPTS,
    adapter_provenance,
    atomic_json,
    base_identity,
    load_adapter,
    reserve_output,
    runtime_provenance,
    tokens_for,
)

_LOCK = threading.RLock()
_ACTIVE = {}
_UNSUPPORTED = (LoRAEmbedding, LoRASwitchLinear, DoRALinear, DoRAEmbedding)


def _adapter_modules(model):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, _UNSUPPORTED):
            raise TypeError(f"unsupported adapter module at {name}")
        if isinstance(module, LoRALinear):
            modules.append((name, module))
    if not modules:
        raise ValueError("model has no supported LoRA adapter")
    return modules


@contextmanager
def adapter_mode(model, enabled):
    """Set configured scales or zero; restore previous scales on exit, even if nested."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    with _LOCK:
        modules = _adapter_modules(model)
        key = id(model)
        outer = _ACTIVE.get(key)
        if outer is None:
            original = [(name, module, module.scale) for name, module in modules]
            _ACTIVE[key] = (model, original)
        else:
            original = outer[1]
            if [(n, id(m)) for n, m in modules] != [(n, id(m)) for n, m, _ in original]:
                raise RuntimeError("adapter modules changed inside gated context")
        previous = [(module, module.scale) for _, module in modules]
        try:
            for _, module, scale in original:
                module.scale = scale if enabled else 0.0
            yield model
        finally:
            for module, scale in previous:
                module.scale = scale
            if outer is None:
                del _ACTIVE[key]


def base_tensor_references(model):
    """Map base names to array objects, normalizing LoRA wrappers to original names."""
    adapters = [(name, module) for name, module in model.named_modules()
                if isinstance(module, LoRALinear)]
    wrappers = {name + ".linear.": name + "." for name, _ in adapters}
    result = {}
    for name, value in tree_flatten(model.parameters()):
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}:
            continue
        for prefix, original in wrappers.items():
            if name.startswith(prefix):
                name = original + name[len(prefix):]
                break
        if name in result:
            raise ValueError(f"duplicate normalized tensor name: {name}")
        result[name] = value
    return result


def compare_base_tensor_references(before, model):
    after = base_tensor_references(model)
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(name for name in set(before) & set(after) if before[name] is not after[name])
    return {"before_count": len(before), "after_count": len(after),
            "missing": missing, "added": added, "different_array_objects": changed,
            "all_same_objects": not (missing or added or changed)}


def _atomic_npz(path, arrays):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-gate-")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _measure(model, tokenizer):
    model.eval()
    outputs, logits = [], []
    for prompt in GENERATION_PROMPTS:
        ids = tokens_for(tokenizer, prompt, CONCISE_SYSTEM)
        started = perf_counter()
        value = model(mx.array([ids]))[0, -1, :].astype(mx.float32)
        mx.eval(value)
        logits.append(np.array(value, dtype=np.float32, copy=True))
        response = generate(model, tokenizer, prompt=ids, max_tokens=80, verbose=False,
                            sampler=make_sampler(temp=0.0))
        outputs.append({"prompt": prompt, "prompt_tokens": ids, "response": response,
                        "wall_s": perf_counter() - started})
    return outputs, logits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = perf_counter()
    reserve_output(args.output)
    output = Path(args.output)
    identity = base_identity(args.model)
    model, tokenizer = load(args.model)
    model.eval()
    before_refs = base_tensor_references(model)
    base_responses, base_logits = _measure(model, tokenizer)
    counts = load_adapter(model, args.adapter, identity, args.revision)
    model.eval()
    modules = _adapter_modules(model)
    scales = {name: float(module.scale) for name, module in modules}
    identity_result = compare_base_tensor_references(before_refs, model)
    with adapter_mode(model, enabled=True):
        active_responses, active_logits = _measure(model, tokenizer)
    with adapter_mode(model, enabled=False):
        disabled_responses, disabled_logits = _measure(model, tokenizer)
    after_scales = {name: float(module.scale) for name, module in modules}
    differences = [float(np.max(np.abs(a - b))) for a, b in zip(base_logits, disabled_logits, strict=True)]
    exact = [a["response"] == b["response"] for a, b in zip(base_responses, disabled_responses, strict=True)]
    provenance = runtime_provenance()
    provenance["source_sha256"]["surgery_gate.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(output / "run.json", {"args": vars(args), "base_hashes": identity,
                "adapter": adapter_provenance(args.adapter), "adapter_parameters": counts,
                "runtime": provenance, "revision_provenance": "caller supplied"})
    atomic_json(output / "generation.json", {"base": base_responses, "active": active_responses,
                                               "disabled": disabled_responses})
    _atomic_npz(output / "logits.npz", {f"{name}_{i}": logits for name, values in
                (("base", base_logits), ("active", active_logits), ("disabled", disabled_logits))
                for i, logits in enumerate(values)})
    result = {"max_absolute_logit_differences": differences, "logit_tolerance": 1e-5,
              "greedy_response_exact_matches": exact, "base_tensor_identity": identity_result,
              "adapter_module_count": len(modules), "configured_scales": scales,
              "scales_after_contexts": after_scales,
              "passed": max(differences) <= 1e-5 and all(exact)
                        and identity_result["all_same_objects"] and scales == after_scales,
              "scope": "Explicit caller-selected routing; not evidence that always-on adapter preserves language.",
              "total_wall_s": perf_counter() - started, "mlx_process_peak_memory_bytes": mx.get_peak_memory()}
    atomic_json(output / "metrics.json", result)
    if not result["passed"]:
        raise ValueError("gated parity check failed; see metrics.json")


if __name__ == "__main__":
    main()
