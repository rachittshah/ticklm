# Temporal Bridge

A local experiment in giving a frozen language model compact, explicit temporal evidence from observed measurements and a frozen time-series forecaster. It reuses [Rachitt's MLX TimesFM implementation](https://github.com/rachittshah/mlx-tsfm), measures a cheap classical forecaster as a hard control, and keeps both neural models unchanged.

**Status:** implementation and development search complete; independent held-out comparisons are running. Do not interpret this repository as evidence of improved general temporal intelligence. Results and retained evidence will be added after review.

## What it does

```
regularly sampled history ──┬── observed time/value capsule ──┐
                           │                                │
                           └── TimesFM / cheap forecast     ├── frozen Qwen ── answer
                                  └── projected capsule ────┘
```

The representation records sampling interval, past versus future, compact curve knots, and measured interpolation error. It does not receive the question or its answer while compressing evidence. Forecast uncertainty is pointwise; it does not imply a calibrated probability for a whole future path.

The study distinguishes observed-history interpretation from forecast-assisted decisions. All forecast-free observed arms share the same historical representation. The numerical forecaster is also scored directly without an LLM.

## Local setup

Python 3.12 and Apple Silicon are required for the measured MLX path. The experiment uses the local Qwen3-4B abliterated 4-bit checkpoint as an available small-model subject, not a representative frontier model.

```bash
cd /Users/rachitt/labs/temporal-bridge
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]' -e /Users/rachitt/mlx-tsfm
mkdir -p data artifacts
curl -L --fail https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -o data/ETTh1.csv
```

Recorded ETTh1 SHA-256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`.

The local TimesFM implementation used here is commit `c834636e97f98940d95e34601a17de3886bf79d3`. TimesFM weights are pinned to revision `43046b85ec22d584a13f8098c2ed39c889e129c2` of `google/timesfm-3.0-pytorch`. The model's non-commercial weight license applies separately from the implementation; this repository does not redistribute weights. [Official model card](https://huggingface.co/google/timesfm-3.0-pytorch).

Supply a local MLX language-model directory with `--model`. The default is `/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit`. Manifests record exact weight/tokenizer hashes, package versions, external TSFM source hashes and data identity. Model weights, downloaded data and scratch artifacts are ignored by Git.

## Reproduce the experiment

Use a new explicit run directory; existing artifacts are not overwritten.

```bash
.venv/bin/python -m temporal_bridge.backends --smoke --model /path/to/local/mlx-model
.venv/bin/python -m temporal_bridge.experiment dev --run-dir artifacts/my-run --model /path/to/local/mlx-model
.venv/bin/python -m temporal_bridge.experiment test --run-dir artifacts/my-run --model /path/to/local/mlx-model
.venv/bin/python -m temporal_bridge.experiment demo --run-dir artifacts/my-run --model /path/to/local/mlx-model
```

Development evaluates uniform/adaptive curves at 256/384/512 evidence-token caps on 36 cases. Selection maximizes development accuracy, breaking ties by lower prompt-token count. The selected configuration is frozen before 72 held-out cases. The test refuses changed code, data, model identity, dependencies or checkpoint weights.

Seven final arms: question-only, raw values, uniform history, temporal statistics, cheap forecast, TimesFM forecast and mismatched TimesFM forecast. Every prompt and answer is retained. Scoring restricts the next token to A/B/C/D and therefore measures constrained choice, not general free-form reasoning; separate generated examples are also retained. Exact questions and labels are in the benchmark module.

The real-data arm has 36 held-out windows of one ETTh1 temperature series and one future-maximum-bin question. Its uncertainty is exploratory window-level evidence. Synthetic observed questions include reversal and elapsed-time scaling partners. The study does not establish calendar reasoning, causal understanding, irregular-time reasoning or uncontaminated cross-domain generalization.

## Verify

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
git diff --check
```

Current validation: 41 tests pass; Ruff passes. Tests exercise clock serialization, numeric error certificates, invalid data, split separation, hidden-label isolation, exact option scoring, checkpoint pinning and frozen-run identity.

## Research record

- [Design and alternatives](docs/design.md)
- [Dated primary-source literature review](docs/literature.md)
- [Independent scientific review](docs/review-before-test.md)
- [Three further research hypotheses](docs/research-next.md)

Prior work already covers TSFM–LLM fusion, soft-token alignment, forecast quantile prompts and Apple MLX deployment. This work tests a particular quality/cost trade-off; it does not claim the broad architecture is novel.
