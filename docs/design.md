# Frozen temporal evidence bridge: experiment design

Date: 2026-11-02. This is a local research proof of concept, authorized for implementation. No weights are trained. No existing repository is changed, committed, or pushed. The working branch is `research/temporal-capsules-2026-11-02` in this standalone repository. Rollback consists of removing this new directory; the Hugging Face download cache may remain.

## Question and competing approaches

Can explicit, compact temporal evidence improve a frozen small language model's answers at lower prompt cost, and does a frozen time-series foundation model add predictive value beyond a cheap numerical forecaster?

1. Raw numbers in the LLM context: no additional model, simplest, but high token cost and weak numerical inductive bias.
2. Train a TSFM-to-LLM soft-token projector: fewer tokens, can carry hidden features, but needs paired data/training and is already close to TEST, TS-Reasoner and PocketTSLM.
3. Forecast tool calls with full numeric results: zero training, clear interfaces, but added call/response tokens and prior art already supplies forecast quantiles to LLMs.
4. **Chosen experiment: an explicit-clock, query-independent temporal capsule.** Compress historical values and forecast quantiles with a budgeted joint piecewise-linear codec; disclose the measured reconstruction error and distinguish observations from projections. This is cheap, inspectable, and works with frozen existing models. The exact quality/cost result is the research hypothesis; neither fusion nor piecewise-linear compression is claimed new.
5. Rolling predictive-surprise event encoding: potentially useful for unexpected events, but more TSFM passes and a different benchmark. Deferred until the basic fusion earns its cost.

The central candidate improvement is joint adaptive knot allocation across a forecast's median and marginal quantile curves, with an explicit elapsed-time axis and a measured interpolation-error certificate. The certificate is about compressed numeric evidence at sampled times, not forecast truth, calibrated uncertainty, path probability, or broad temporal intelligence. This connects familiar components; the literature review does not establish globally novel architecture.

## Architecture and boundaries

Reuse Rachitt's `/Users/rachitt/mlx-tsfm` at the recorded local commit, and a locally cached Qwen3-4B abliterated 4-bit checkpoint. The latter is an available small-model experimental subject, not representative of all frontier LLMs. Use its actual tokenizer and disable thinking in both scoring and generation.

Input is a finite regularly sampled univariate history, its sampling interval and a horizon. Capsules are built without the question, options, hidden future, answer or generator parameters. Observed questions receive only historical evidence. Forecast questions additionally receive predicted median and 10th/90th quantiles or a cheap model's residual intervals, explicitly marked as projections. The CSV reader verifies hourly timestamps; the array API requires the caller to supply regularly sampled data and rejects invalid intervals. Native TimesFM deciles only. A crossing envelope expands to include all three original values while preserving the point prediction; repairs are recorded and raw outputs retained. Clock values round-trip exactly, and the certificate uses the displayed timestamps and rounded values.

The adaptive codec greedily inserts the sampled time with largest vertical interpolation error across the displayed curves, until its knot budget is reached. It then measures actual reconstruction error after numeric rounding. A tokenizer budget is enforced on the entire evidence block. Uniform-knot and temporal-statistics alternatives are development candidates/controls. A budget cap is not an a priori distortion guarantee.

The statistical forecaster selects among persistence, seasonal persistence and linear extrapolation using only a trailing validation block within each input history. It produces an honest simple baseline; neither future labels nor benchmark scores choose its per-case model. A no-LLM direct decision reference is scored from each forecast on compatible questions.

## Evaluation

Use fresh synthetic series with temporal counterfactuals for observed history tasks, plus non-overlapping chronological blocks from ETTh1 for a small forecast integration task. Hold out seeds and some generator morphologies; time-reversal and time-dilation partners remain in one split/cluster. Fixed context 128, horizon 24. ETTh1 pretraining overlap cannot be ruled out; this is not an uncontaminated benchmark claim or a reproduction of an official leaderboard.

Core arms: raw history, budgeted uniform history, temporal statistics, adaptive history plus cheap forecast, adaptive history plus TimesFM forecast, and mismatched TimesFM forecast. A question-only control exposes answer priors. For observed questions the last two proper-forecast arms share the same historical capsule by construction: any improvement is the codec's, not evidence that TimesFM taught the LLM temporal order. A full-forecast reference can diagnose compression losses if needed.

Development-only search: fixed candidates combining uniform/adaptive codecs and evidence budgets (initially 256/384/512 tokenizer tokens). Select by development accuracy with a predeclared token-cost tie break; preserve every candidate result. Freeze selection, source fingerprints and case identifiers before scoring held-out cases. Do not optimize on final outcomes. If a bug invalidates a result, retain it as invalidated and explain the rerun.

Evaluation uses conditional A/B/C/D token likelihood with identical questions/options and prompt construction across arms. This measures constrained multiple-choice interpretation, not unconstrained answer quality. Also run real free-generation examples. Record every prompt, answer, probability, token count, synchronized model timing, forecast, numeric error and configuration. Report per-task accuracy, paired cluster-bootstrap differences, tokens, warm p50/p95 and first-load/compile separately. Do not interpret conditional option probabilities as calibration.

Pre-run review rulings: evidence budgets exclude the common question/chat template; report full input tokens separately. Source, model weights/tokenizer, package versions, external editable TSFM sources and exact checkpoint revision are frozen. ETTh1 intervals bootstrap guarded windows of one underlying OT series and remain exploratory. The codec can allocate historical knots differently depending on the forecast, so the forecaster comparison is a whole-pipeline ablation. Timing with precomputed forecasts is measured; uncached timing is an explicitly labeled sum of that measurement and separately timed forecast work. Real forecast blocks are 18 development and 36 held-out, with 18/36 synthetic observed cases respectively.

Success is separated: (a) useful temporal evidence if it improves held-out history interpretation versus raw/uniform at lower input cost, with paired uncertainty; (b) useful TSFM fusion if it improves forecast QA versus the same cheap-forecast capsule and loses that gain under mismatched predictions; (c) economic value if measured additional forecast cost is small versus the saved LLM work. With a small pilot, confidence intervals or negative results may be inconclusive. No outcome is declared in advance.

## Implementation and verification plan

- `backends.py` and tests: finite shapes, native quantiles, synchronized TimesFM inference, actual Qwen forced-choice/generation, durable smoke CLI. Terra owns these files.
- `benchmark.py` and tests: reproducible exact labels, public/private separation, split disjointness, timestamp interventions and deterministic option shuffling. Luna owns these files.
- `core.py` and tests: temporal codec, reconstruction bound after rounding, token budget, uniform/statistical controls, classical forecast selection. Root owns these files.
- `experiment.py`: source download/provenance, explicit run IDs, immutable case/selection artifacts, development search, final evaluation, bootstrap/cost summaries and demos. Root owns it.
- `docs/literature.md`: dated primary-source survey and novelty critique. Research Terra owns it.
- `README.md`, `docs/results.md`: exact commands, measured outcomes, failure analysis, model/data limitations and next research gate.

Formal tests cover finite/empty/rank-invalid inputs, no future leakage, explicit clock, counterfactual labels, reconstruction error, budget failure and option-scoring semantics. Execute the real CLI on actual model weights, not one-off test scripts. Run lint, the full project test suite, retained artifact integrity checks and an independent review before claiming completion. Do not commit or push.
