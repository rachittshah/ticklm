# TickLM-Bench research and execution log

Date: **3 November 2026**. This continues the [TimePatch research journey](weight-surgery-journey.md). The new task was to build and publish a broader benchmark and report, evaluate accessible models and client harnesses, retain failures, and keep pushing completed work to GitHub.

## Scope and decisions

The user authorized up to **$25 in additional paid model API usage**, beyond local inference and existing sign-ins, and then explicitly excluded exo. The connected local setup exposed Claude and Codex sign-ins but no verified metered API key route in the checked configuration. The active CCA startup account was verified as subscription-backed with extra usage disabled. OpenCode's stored Anthropic OAuth credential was expired and was not copied or refreshed. Direct paid API execution remained unused; this is not a claim that subscriptions, hardware, or electricity are free.

The research review found close prior art, particularly StreamDecisionBench for decisions that remain in force as events arrive, TRACE and QSTRBench for generated temporal constraints, MultiTempBench for calendar arithmetic, TIME for broader real-world questions, and AgentTime for runtime behavior. We consequently framed the release as a reproducible comparative study, not the first temporal benchmark or an established state-of-the-art result.

The seven-family task generator and strict parser were frozen before test inference. Full and screening hashes were committed, and the screen was explicitly a subset of the full suite. Qwen3-0.6B and SmolLM3 were selected for the larger tier before test outcomes. The later micro tier was selected to make expensive client-harness coverage practical; it preserves exact questions from the screen and is reported separately.

## Git checkpoints

| Commit | What became reviewable | Why it mattered |
|---|---|---|
| `26a7938` | Protocol, scope, $25 budget and rollback plan | Established authorization and measurement boundaries before implementation |
| `1c6a64c` | Frozen tasks, independent oracle checks, adapters, budget ledger and runner | Bound test prompts and scoring before inference |
| `8421a60` | Live loop, source checks, strict-versus-option-line diagnostic and isolated OpenCode support | Separated freshness from semantic correctness and created the immutable execution archive |
| `5f69d96` | Report generator, model expansion plan and evidence exporter | Made counts, coverage, model identity and accounting inspectable |
| `ea353d7` | Measured checkpoint and external-scorer replication | Published 2,156 static records and 25 live decisions available at that point |
| `ecfedb4` | Reproducible charts and external evidence allowlist | Bound figures to their input hashes and kept private state out of bundles |
| `caf260d` | Verified offline live-parser replay | Isolated output-format sensitivity while preserving original measurements |

Later publication commits add the finished report, figures and evidence. Commit hashes identify checkpoints, not a claim that every run executed the surrounding repository's current HEAD.

## Problems found and how they were handled

**Output format dominated some local scores.** Qwen's first screen returned complete option lines rather than only letters, producing 100% strict-invalid outputs. The original rule and scores were kept. A separate post-hoc diagnostic accepts only an entire exact option line, uniformly across systems. A later live replay uses recorded publication times, verifies that the strict replay reproduces every original integral, then tests exact action-option lines. Neither is a preregistered primary score or a new live intervention.

**A guard initially consulted the oracle.** Review caught that a publication policy which knows whether an answer is correct is not deployable. The live harness now reports a practical freshness-only check, an unguarded policy, and a clearly labeled oracle-assisted reference. The model receives no credit for the controller's clock, version checks, or ground truth.

**Source edits overlapped early experiments.** The exact historical adapter files were recovered and hash-verified. Their MLX and Claude/Codex core functions have identical normalized syntax trees, but the import-to-source-hash race is still disclosed. Subsequent runs use the immutable `execution-8421a60` archive. The `git_commit` field can describe the surrounding checkout at launch; the archived executable source and its file hashes are authoritative for those runs. Raw manifests were not rewritten to hide that distinction.

**Comparable counts were not enough.** The report now checks frozen dataset hashes, case-set hashes, scorer identity and complete coverage. It distinguishes raw/derived views, tiers, requested generation policies and native-client controls. Failed requests remain coverage failures; a model with 12 correct completed answers from 14 expected is not displayed as 14/14.

**Client overhead differed sharply.** Native clients do not enforce the requested output-token limit. OpenCode's development run correlated with two local inference calls, one with about 14,849 input tokens; this was timing evidence, not a captured request-ID proof. Those harnesses are measured as complete systems. The benchmark does not claim equal computation or isolated-hardware latency across them.

**The external benchmark release required inspection.** TIME's inspected release was missing evaluation prompts and had apparent scorer/schema issues; we did not invent an official score. StreamDecisionBench's pinned scorer was executable. In an isolated environment without inherited credentials, 66 targeted upstream tests passed and the authors' saved recording reproduced 54.1054% integrated score, matching their published 54.11%. This is an external scorer replication, not new inference by our models.

**An early discovery command treated `cca --help` as a profile name.** The newly created profile was moved to ignored local quarantine, without following its symlinks. It is excluded from publication. An isolated OpenCode diagnostic service was stopped after discovery; private authentication was not copied. Evidence export uses explicit file allowlists, credential-pattern checks and archive hash verification rather than archiving the whole working directory.

## Evidence and limitations

Every primary run retains the frozen cases, configuration, request-start journal, per-request results, source hashes, runtime metadata and summary. Failure records are not retried into better-looking scores. Model downloads retain repository revisions and file hashes; model weights are excluded from Git and release assets. The final report, parser replay and figures are generated from retained data.

The first complete repository validation reported **374 passed, 1 skipped**, with Ruff clean. Later modules add their own checks; the final release records the final suite result. A built wheel was installed into an isolated Python 3.13 environment and generated the exact screening hash `869a458b245e4d8c1aae519eccf802600140be2f694db42deae447ae724648d5`; actual model experiments used the recorded Python 3.12 environment. This packaging check is not a second hardware/model replication.

The study remains small and synthetic. Development and test share templates; calendar and chronology coverage are narrow; some distractor actions never become the correct answer; output formatting confounds several strict scores. Native clients and CPU-only analysis sometimes ran concurrently with local work, so measured latency describes this execution environment rather than a controlled hardware ranking. The [benchmark guide](../benchmarks/ticklm-v0.1/README.md) and final report state these limits alongside the results.
