# TickLM-Bench v0.1

A reproducible first study of temporal reasoning, answer formatting, client overhead, and whether an answer remains useful while time and events advance. Research cutoff: **3 November 2026**.

- **2,436 static attempts** across **19 model/harness configurations**, with 448 distinct frozen cases and separate 112-, 448-, and 28-request tiers. Fifteen client failures are retained; no missing request records.
- **406 live requests** across twelve 30-second real-clock trajectories. Snapshot correctness, completion correctness, freshness, and correct/wrong/undecided time are separate measurements.
- Local MLX models from the Qwen, SmolLM, Gemma, LFM, Llama, and Phi families; Claude Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 through CCA startup; GPT-6 Astra, GPT-6.1 Sol, GPT-6.1 Terra and GPT-6 Luna through Codex; and local Ollama, llama.cpp and OpenCode trials. Exo was excluded as requested.
- Frozen strict scoring remains intact. Explicitly post-hoc exact-option diagnostics and an offline live replay reveal formatting sensitivity without rewriting original results.
- The official StreamDecisionBench scorer reproduced **54.1054%**, matching the authors' published **54.11%**, from their retained recording. That is a scorer replication, not new inference by our models on StreamDecisionBench.

Read the [human-readable report](https://github.com/rachittshah/ticklm/blob/ticklm-bench-v0.1.0/benchmarks/ticklm-v0.1/report.md), [guide and Mermaid diagram](https://github.com/rachittshah/ticklm/blob/ticklm-bench-v0.1.0/benchmarks/ticklm-v0.1/README.md), [detailed tables](https://github.com/rachittshah/ticklm/blob/ticklm-bench-v0.1.0/benchmarks/ticklm-v0.1/results.md), and [research journey](https://github.com/rachittshah/ticklm/blob/ticklm-bench-v0.1.0/docs/benchmark-journey.md).

The evidence archive contains 205 explicitly selected files: prompts and responses, run journals, model revisions and hashes, frozen data, source snapshots, verification logs, and the small external-scorer reproduction. It excludes model weights, authentication, private profiles, virtual environments and quarantine. The scoped license covers the benchmark and its original client integrations; model and upstream artifact licenses remain separate.

Validation: **382 tests passed, one opt-in real-model test skipped**, Ruff clean, wheel installation and CLI smoke checks passed, and an isolated install reproduced the exact frozen screening hash. Additional metered API spend was **$0** of the authorized $25; existing subscriptions and local compute were used.

This is an exploratory research release, not an established state-of-the-art ranking. The synthetic suite is small, shares templates across splits, and shows substantial output-format sensitivity. Native clients have hidden context and unenforced token caps. An unseen event during inference is an information/freshness problem, not by itself a reasoning error.
