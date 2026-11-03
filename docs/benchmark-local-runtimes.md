# Local runtime and live-loop evidence

These runs use the frozen benchmark and the source archive at `artifacts/benchmark-v1-20261103/execution-8421a60/src`, one GPU-backed model at a time. They do not establish a general model ranking. The local results are execution records, not external benchmark scores.

## Live trajectories

Both configurations ran three independently scheduled 30-second trajectories through `ticklm_bench.live`, with an external monotonic clock supplying the current elapsed time and deadline to the model. The model compared the supplied values; it did not measure time itself. Snapshot decisions, freshness, and correctness are distinct: the freshness-only guard requires a parseable response for the current state version; the oracle guard additionally uses ground-truth correctness and is a non-deployable reference.

| Configuration | Requests | Invalid or failed | Actual elapsed | Undecided in observed 90 s |
|---|---:|---:|---:|---:|
| Qwen3 0.6B MLX | 343 | 343 | 90.05 s | 90 s |
| SmolLM3 MLX | 38 | 38 | 95.03 s | 90 s |

The extra wall time includes an in-flight request finishing after the measured horizon. No answer passed the frozen strict parser in these runs, so snapshot/completion accuracy on valid answers is undefined; the decision tracks were undecided for the whole observed horizon. These results do not isolate temporal reasoning from answer formatting.

Reproduce on an equivalent local installation:

```sh
export PYTHONPATH=artifacts/benchmark-v1-20261103/execution-8421a60/src
.venv/bin/python -m ticklm_bench.live --config benchmarks/ticklm-v0.1/configs/qwen3-06b-mlx.json --output artifacts/benchmark-v1-20261103/live/qwen3-06b --duration 30 --trajectories 3
.venv/bin/python -m ticklm_bench.live --config artifacts/benchmark-v1-20261103/provision/smollm3/config.json --output artifacts/benchmark-v1-20261103/live/smollm3 --duration 30 --trajectories 3
```

Each output directory contains its manifest, detailed requests, truth intervals, and summary. Model files and source/configuration hashes are recorded in the manifests. An existing output directory must not be overwritten.

## Ollama HTTP screen

The 112-case frozen screen used the committed `ollama-qwen3-06b.json` configuration and a development preflight. The request contained `reasoning_effort: none`; the retained direct preflight, product preflight, and 112 responses contained no exposed reasoning text or `<think>` block. That observation cannot prove what internal computation occurred. The model returned explanatory or option text rather than the required single letter: all 112 transports completed, but all 112 were invalid under the frozen parser (56 raw and 56 derived). A zero score under that parser is therefore not a clean estimate of temporal competence.

```sh
PYTHONPATH=artifacts/benchmark-v1-20261103/execution-8421a60/src .venv/bin/python -m ticklm_bench.sweep --plan artifacts/benchmark-v1-20261103/local-runtime-plan.json
```

The plan names the frozen input, configuration, retained preflight, and output directory. The run manifest identifies the archive package path and hashes every loaded source file. Its `git_commit` field resolves against the surrounding working tree rather than the exported archive; use the archive identity and recorded file hashes for execution provenance.

## OpenCode micro screen

The committed OpenCode/Ollama configuration ran the predeclared 28-case micro subset, selected as one paired episode per family in both views. Twenty-seven requests returned successfully and one reached the configured 120-second timeout. All 27 returned responses were invalid under the frozen parser; derived coverage was 14/14 and raw coverage was 13/14. The native client's prompt, internal calls and generation limits are not the same as those of the HTTP adapter, so this is a separate system result. Compare it only on the identical micro subset; do not pool it with the full screen.

```sh
PYTHONPATH=artifacts/benchmark-v1-20261103/execution-8421a60/src .venv/bin/python -m ticklm_bench.runner run --dataset artifacts/benchmark-v1-20261103/frozen-micro/micro.jsonl --config benchmarks/ticklm-v0.1/configs/opencode-ollama-qwen3-06b.json --output artifacts/benchmark-v1-20261103/runs/opencode-ollama-micro
```

The retained earlier development preflight is `artifacts/ticklm-bench-preflight/opencode-ollama-qwen3-06b-v2.json`. The OpenCode configuration SHA-256 is `13365288ed39e9a0dd97f8adadded8618ab762a29a8be73c3d39ffdd41c57083`; the micro input SHA-256 is `122b912e70dc6c51e39921baeba2d0ae91967cbccd2e2e87ec7f4f8b363b8c83`.

## Private llama.cpp HTTP screen

After the OpenCode run, the operator unloaded only `qwen3:0.6b` with Ollama's `keep_alive: 0` and verified that `/api/ps` listed no loaded models. A separate server was then started on loopback, using local Homebrew llama.cpp 0.4.0 and the same on-disk GGUF:

```sh
/opt/homebrew/bin/llama-server --model /Users/rachitt/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa --host 127.0.0.1 --port 18084 --alias qwen3-0.6b-q4 --reasoning off
PYTHONPATH=artifacts/benchmark-v1-20261103/execution-8421a60/src .venv/bin/python -m ticklm_bench.sweep --plan artifacts/benchmark-v1-20261103/llamacpp-runtime-plan.json
```

The server's PID was 31726. The operator stopped that server after the run, confirmed that port 18084 no longer had a listener, and confirmed that Ollama still had no loaded model. The model's independently computed SHA-256 is `7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa`; the llama.cpp configuration SHA-256 is `8900a528d8daff8335bc1777128ff8411141d6c5d88fc2b13692c2faa66282e2`. The 112-case screen SHA-256 is `869a458b245e4d8c1aae519eccf802600140be2f694db42deae447ae724648d5`.

The server's model list, direct development response, and product development preflight resolved to `qwen3-0.6b-q4`; those responses exposed no reasoning text. All 112 screen calls returned successfully with that resolved alias, but all 112 responses failed the frozen parser, and no retained response exposed reasoning text or `<think>` blocks. The configuration, plans, server log, model-list response, dev responses, raw screen records, summaries, and manifests preserve the details; absence of exposed reasoning is not proof of the model's internal computation.

The exact unload and post-run checks were:

```sh
curl -fsS --max-time 30 http://127.0.0.1:11434/api/generate -H 'Content-Type: application/json' -d '{"model":"qwen3:0.6b","keep_alive":0}'
curl -fsS --max-time 5 http://127.0.0.1:11434/api/ps
lsof -nP -iTCP:18084 -sTCP:LISTEN
```

The no-listener check exits 1 when successful. Direct probe evidence is retained as `ollama-direct-dev-check.json`, `llamacpp-models.json`, and `llamacpp-direct-dev-check.json` under `artifacts/benchmark-v1-20261103`; product preflights are under `local-preflight/`. Source archives, artifacts, local model files and environments are not included in this document.
