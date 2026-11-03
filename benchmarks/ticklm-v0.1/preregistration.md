# Frozen screening protocol — 3 November 2026

Frozen at 07:55:56 UTC, before any test-model requests. Development preflights are excluded from all scores. Seed: `11032026`. The parser, task generator and source hashes are recorded in the freeze manifest. This is an exploratory v0.1 protocol, not a power analysis or a formal external preregistration.

| Tier | Requests per system | Independent paired episodes | SHA-256 of JSONL |
|---|---:|---:|---|
| Full | 448 | 112 | `814f7a4c586324f3a9bf5e4244a8a361327920f693555f200bba82b96abb1c80` |
| Screen | 112 | 28 | `869a458b245e4d8c1aae519eccf802600140be2f694db42deae447ae724648d5` |
| Development | 56 | 14 | `94c80fd9523d97f60bc74c69884c636ddb854ad2b9a692e8ba96b99ef7b1fa79` |

The screen is an exact subset of full cases. Every episode contains two changed-fact variants and raw/derived views. Seven families receive equal weight. Views and tiers have separate tables. Generated text is scored; no hidden choice-logit advantage. Exactly one letter, an entire `Answer: A`/`Final: A` response, or an exact answer JSON object is accepted by the shared frozen parser. Formatting failures count wrong. Transport failures remain coverage failures, exclude a run from comparable rankings, and are never silently retried.

The common instruction asks for a single letter. Direct-generation runs use greedy decoding and a 256-token cap. Thinking is disabled where the actual template supports it; otherwise support is unknown. Native-client runs request xhigh reasoning and a 256-token cap, but the cap is unsupported and their internal instructions/usage differ. They form their own table. Model inclusion is based on local compatibility, family diversity, access and cost, never on observed test scores.

Initial inclusion list: cached Qwen3-0.6B BF16, cached Qwen3-4B-abliterated 4-bit, Ollama Qwen3-0.6B Q4_K_M, Claude Fable 5.1 through CCA startup, and GPT-6 Astra through Codex. Candidate downloads from the captured public Hugging Face catalog: SmolLM3-3B-4bit, Gemma-3-1B-it-4bit, LFM2.5-1.2B-Instruct-4bit, Llama-3.2-3B-Instruct-4bit, Phi-4-mini-instruct-4bit, and Qwen3.5-4B-OptiQ-4bit. Include compatible checkpoints that complete a development preflight; record download/runtime failures. Add OpenCode with local inference and llama.cpp with the same Ollama GGUF if their isolation and identity checks pass. Exo is excluded at the user's request.

Deeper 448-request runs are preselected for Qwen3-0.6B BF16 and SmolLM3-3B because they provide two inexpensive model families; feasibility may remove a run, test performance may not choose replacements. Live runs are preselected for Qwen3-0.6B, one other local family, Claude Fable and Codex, with three fixed 30-second trajectories each. Native calls may finish after the trajectory; this is recorded and supplies no in-force credit beyond its end.

Additional configured-provider API models may be included only after route identity, pricing and billable token caps are verified, using the shared $25 ledger. Record any such access-driven additions before their first test request. The provider preference changed from unavailable OpenRouter to another already configured provider before paid execution. No paid API calls had occurred at freeze time.

Primary metrics: accuracy and complete-pair success per view; invalid output and transport failure rates; family-level accuracy; episode-clustered, family-stratified bootstrap intervals. Live metrics: snapshot correctness, completion correctness, state freshness, latency, and correct/incorrect/undecided time with a decision in force. No composite mixes these quantities. Option permutation robustness, external benchmark replication and cross-hardware repeatability remain unmeasured unless explicitly added as separate studies.
