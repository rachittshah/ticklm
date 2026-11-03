# TickLM-Bench v0.1: protocol and implementation checkpoint

Date and research cutoff: **3 November 2026**. Branch: `benchmark/ticklm-bench-v1`, based on `8e1c61e`. The user authorized a broad model/harness evaluation and publication, with a **$25 total paid API budget** beyond local inference and existing client subscriptions. **Exo is excluded at the user's explicit correction**; the initial interpretation as exo-explore's inference runtime was not the requested harness. No exo installation or launch occurred.

## Aim and claim boundary

Build a reproducible temporal reasoning benchmark with a separate live-system evaluation, then publish code, generated datasets, model/harness identities, failed attempts, measurements and a readable report. The existing TimePatch result is historical evidence of four learned policies, not a fresh general leaderboard result. “State of the art” is a research ambition, not an established property or a claim that this new benchmark covers every model.

## Design alternatives considered

| Approach | Strength | Limitation |
|---|---|---|
| Reproduce published static temporal sets | Existing external comparators and established tasks | Public contamination and no direct live-state test |
| Generate interval/constraint problems only | Exact mathematical oracles and controlled difficulty | Already covered by TRACEBench and QSTRBench; narrow task distribution |
| Measure streaming decisions only | Exposes stale answers and serving latency | StreamDecisionBench already covers this; hardware and clients dominate some differences |
| **Paired hybrid with explicit access conditions** | Separates changed rules, arithmetic assistance, semantic judgment, and live timing | Requires more controls and separate score tables; synthetic results remain limited |

Choose the fourth approach. Do not merge static answer quality, exact-choice logit scoring, uncapped client results, and live timing into one score.

## Dataset contract

Create an independent `ticklm_bench` package. Seven strata: duration/deadline arithmetic; varied and composed temporal policies; interval relations; scoped event revisions/cancellation; event-time windows with arrival order; Gregorian calendar and explicit UTC-offset arithmetic; narrative chronology constraints. Include equality/boundary cases, changed comparison directions, irrelevant events and explicit insufficient-information cases where the family supports them.

Each episode yields two counterfactual variants with one meaningful fact changed and different correct answers. Keep option mappings fixed within a pair and balance labels. Render matched `raw` and `derived` access conditions: derived features expose primitive arithmetic only, never an action, status, Boolean comparison, signed threshold margin, rule closure or hidden gold. Metadata, IDs, split, seeds and internal rule representation never enter prompts. Exact integer/rational or timezone-aware arithmetic supplies the oracle; independently implemented checks and hand-calculated boundary tests audit it.

The full default suite uses 16 pairs per stratum and both views: 448 requests per system. A fixed broad screening suite uses four pairs per stratum and both views: 112 requests. These are separate named tiers, with separate confidence intervals and tables. They are exploratory sample sizes, not a power claim. Freeze test content hashes, ordering, model inclusion policy and parser before evaluating test outputs. Release the new cases with the report and acknowledge that they become public afterward. New seeds do not prove absence of training contamination.

## Models and harnesses

Use a model-agnostic request/result contract and capability metadata. Primary output scoring uses generated text and a strict, shared answer parser; invalid or missing answers count as failures. Exact local candidate logits may be reported only as a separate track. Record requested/resolved model, source revision or file hash, quantization, tokenizer/chat template, generation settings, thinking policy, runtime version, hardware and endpoint identity. Native Claude Code/Codex or other agent clients belong to a separate model-plus-harness track when hidden instructions, token caps or sampling controls differ.

Discover accessible local models and current provider catalogs. Evaluate a broad set of feasible model families within the memory, storage and API budget; publish inclusion/exclusion reasons rather than imply every catalog entry was served. No model results determine inclusion. Run one GPU model at a time. Download only pinned public checkpoints compatible with the installed runtime and retain weight/license metadata. Do not redistribute weights. Existing TimePatch can be an explicitly specialized baseline, not the benchmark's preferred model.

## Cost and evidence

Before any paid request, reserve a conservative maximum cost in a transactional ledger; reject requests that would exceed **$25**. Record the provider pricing snapshot, actual reported usage/cost when available, reservation, attempt ID, errors and any unknown accounting. Never infer zero cost from missing usage. Enforce API output limits where supported; clients that cannot enforce them are labeled separately. Do not put credentials, private profiles or local authentication files in logs or publication assets.

Retain per-case prompts/responses, parse outcomes, request/first-visible-text/completion monotonic times, retries, timeouts, finish reasons and model identity. A failed preflight is a coverage failure, not a zero-quality model result. Publish macro-family accuracy, complete-pair success, invalid rate, option invariance, and pair/trajectory-clustered uncertainty. Show sample sizes and matched comparisons.

## Live track

Use prescribed exogenous event schedules, a real monotonic clock and fresh conversations. Keep snapshot-semantic correctness, correctness at completion, stale-answer eligibility, time with a correct decision in force, and latency separate. Retain events and dispatch/completion snapshots. Deterministic controllers own the clock and state; a model receives no credit for guard correctness. A replay is not an atomic external publisher or a production reliability guarantee. Count client and tool time, and separate cold startup from warm inference.

## Prior art verified this session

- [StreamDecisionBench](https://arxiv.org/abs/2609.38612), 29 September 2026, revised 9 October: streaming decisions in force and latency/judgment attribution. This is close prior art, not an idea to claim as new.
- [TRACEBench](https://arxiv.org/abs/2607.04784), 6 July 2026: dynamically generated interval constraints and trace validation.
- [QSTRBench](https://arxiv.org/abs/2605.18380), 18 May 2026: qualitative temporal calculi and representation variation.
- [MultiTempBench](https://arxiv.org/abs/2603.19017), 19 March 2026: temporal arithmetic and language/calendar/format variation.
- [AgentTime](https://arxiv.org/abs/2610.09944), 7 October 2026, and [Timely-Eval](https://arxiv.org/abs/2601.16486), 23 January 2026: native agent duration and budget behavior.
- [TIME](https://arxiv.org/abs/2505.12891), 19 May 2025: real-world temporal question answering across encyclopedia, news and dialogue settings.

External benchmark replication requires inspecting the official release, split, license and scorer. Do not label independently generated tasks as official scores on those datasets, or claim exhaustive year-to-date literature coverage.

## Ownership, rollback and publication

The data worker owns the task generator/oracle and its tests. The adapter worker owns model/harness adapters and tests. The budget/catalog worker owns safe discovery and paid-request accounting. The parent owns the runner, live integration, comparison/report export, documentation and Git checkpoints. Everyone preserves other edits.

This work touches more than three files; the branch and this plan precede implementation. Existing TimePatch code/results remain intact. New weights, environments, credentials and raw private discovery material remain ignored. Rollback is switching to the parent branch and stopping only servers launched for this benchmark; no existing model files or services are overwritten. Publish the reviewed benchmark, data and report through the existing public repository with explicit licenses for the new code/data and a versioned release.

Operational note: a read-only discovery attempt incorrectly treated `cca --help` as a profile selection. Its newly created profile was moved to ignored local quarantine without following its symlinks. No prompt was supplied; it is not a benchmark/model run and its configuration will not be published.
