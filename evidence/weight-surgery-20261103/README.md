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
