# Temporal weight adaptation: retained evidence

The human-readable account is [the research journey](../../docs/weight-surgery-journey.md). The machine-readable [run index](journey-index.json) maps configurations, source hashes, datasets, and measurements to the original local artifact paths.

## First development checkpoint

[`completed-development-runs.tar.gz`](completed-development-runs.tar.gz) contains the exact saved JSON/JSONL records, datasets, training logs, manifests, and validation transcripts for the original model, four-layer adapter, all-layer adapter, frozen linear probe, and Fable 5.1 review. Both successful and failed verification transcripts are retained. It was created after these runs finished, while the subsequent curriculum run was still active.

SHA-256: `2cc0909442ddd77c53970b26c8a8a3111dd5740318975c907f3bb2923cc21521`.

Inspect without extracting:

```sh
tar -tzf evidence/weight-surgery-20261103/completed-development-runs.tar.gz
tar -xOzf evidence/weight-surgery-20261103/completed-development-runs.tar.gz base-dev/metrics.json
```

The archive excludes model and adapter weight files (`*.safetensors`) and probe feature/head arrays (`*.npz`). Those remain under ignored `artifacts/weight-surgery-20261103/` locally. Their configurations and the trained adapter tensor shapes are retained in the manifests. This is an evidence/code push, not a model-weight release. The pinned original Qwen weights remain unchanged.

Dataset hashes matter independently of source hashes: adding helper functions changed the data module's source hash while the original saved development dataset remained byte-identical. The journal documents this distinction. No final test evaluation was included in this first development archive.

## Subsequent development evidence

| Archive | Contents | SHA-256 |
|---|---|---|
| [curriculum-development-run.tar.gz](curriculum-development-run.tar.gz) | All five checkpoints' development predictions and metrics, exact datasets, training log, reload check, and 205-test verification transcript. Adapter weights excluded. | `7fe87e98cd128784ba59b7b1d824267a2ecc9f10717bb9f966664656ed198e23` |
| [duration-baseline-and-retention-failure.tar.gz](duration-baseline-and-retention-failure.tar.gz) | Unadapted duration-input baseline, raw curriculum token-distribution and generation checks, and pair-bootstrap comparison with the original raw-input baseline. Includes the repeated incorrect `100` response to `17 plus 25`. | `b6a9c814e65739400e17cdd87e27f6bab066286da337a2ba615a8ffc6315cafa` |

These archives contain development results. The duration-input baseline is a distinct condition: the program supplies subtracted durations, so it cannot be interpreted as a test of the model independently calculating elapsed time. Pair-bootstrap intervals resample complete pairs and describe variability within this synthetic sample.

## Completed experiment

Read [the result report](../../docs/weight-surgery-results.md) for the model identities, comparison table, actual clock readings, language failures, routing design, and limits. [selection-lock.json](selection-lock.json) fixes the chosen checkpoint and protocol before held-out evaluation; [novel-rule-protocol.json](novel-rule-protocol.json) fixes the additional unfamiliar-rule diagnostic before its model outputs were produced.

| Archive | Contents | SHA-256 |
|---|---|---|
| [clock-feature-development.tar.gz](clock-feature-development.tar.gz) | First successful clock-feature training run, its language regression and robustness controls, first gate verification, unaugmented frozen-probe control, and intermediate verification transcripts. | `c37b074d3a04b83366c6ffc37dfab7c40cbca593f7f177bdceb4a54c2e5cbc19` |
| [preservation-development.tar.gz](preservation-development.tar.gz) | Selected distillation training run, its remaining language failure, exact gated parity, frozen classifier trained on the identical temporal curriculum, and the second Fable 5.1 review with its prompt. | `d6287b14b843720241a2b41caf28e7f4b2a042192bdc277c1643b982bb614f49` |
| [final-evaluation.tar.gz](final-evaluation.tar.gz) | Full shuffled-answer control run; held-out base, selected, shuffled and frozen-head results; raw-input transfer; unfamiliar-rule results; live base/selected runs; final robustness and language checks; five paired-bootstrap reports; final software verification. | `e0d3fda3858f396aa1fbeab7b66baedef02dfd4389a4c08a02f5aa7040523f7c` |

All archives exclude `*.safetensors` and `*.npz`; local weights and array artifacts remain available under the original ignored run directories. This includes the trained adapters, teacher distributions, probe features/head, and gate-check full-vocabulary arrays. JSON measurements, exact datasets, per-case predictions, ordinary generated text, source hashes, training-label assignments, configurations, and logs are retained in the archives.

For example, inspect the primary result and actual live decisions directly:

```sh
tar -xOzf evidence/weight-surgery-20261103/final-evaluation.tar.gz selected-test-features/metrics.json
tar -xOzf evidence/weight-surgery-20261103/final-evaluation.tar.gz live-selected/cases.jsonl
tar -xOzf evidence/weight-surgery-20261103/final-evaluation.tar.gz novel-selected/metrics.json
```

The shuffled control's automatic suggestion was step 240; the locked comparison deliberately uses **step 480**, matching the selected real-label adapter. The selected model's ordinary-language mode must disable the adapter. Its perfect held-out trained-policy result must be read alongside **0/32 reversed-rule pairs**, **16/32 window-rule pairs**, and its incorrect active-mode arithmetic answer.
