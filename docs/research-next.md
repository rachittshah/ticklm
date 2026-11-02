# Three falsifiable follow-ups for the frozen temporal bridge

Date: 2026-11-02. These are proposed experiments, not implemented results or global novelty claims. They assume a question-independent capsule, frozen model weights and a fixed actual-token budget. They target different failures: ambiguity from compression, missing unexpected events, and paying for a forecaster that does not help.

## 1. Answer-certifying, progressive history compression

**Hypothesis.** At the same token budget, allocating a little space to certified uncertainty about the *observed samples* can produce more correct answers at a fixed error rate than spending every token on interpolated knots. This can help even if TimesFM adds zero predictive value.

For sampled series x and piecewise-linear reconstruction x_hat, compute a segment-specific epsilon_j = max |x_t - x_hat_t| over sampled t in segment j **after serialization and rounding**. The capsule includes the knot values and epsilon_j rounded conservatively upward. Thus L_t = x_hat_t - epsilon_j and U_t = x_hat_t + epsilon_j enclose each observed sample in exact arithmetic; implementation must account for its rounding. Given a later question comparing two sampled instants a,b, L_a > U_b certifies x_a > x_b. For an extreme-order task, one can conservatively bound the set of possible peak and trough positions; say that the order is supported only when every consistent position pair implies the same order. For categorical questions whose labels come from hidden synthetic generating parameters, this sample-level certificate alone is not valid.

An anytime codec can build a *question-independent* sequence of candidate additions, such as a new knot or a tighter segment bound, greedily maximizing estimated reduction in ambiguity over a fixed, public development distribution of generic comparisons. Formally, for public query family Q0 and token cost T, select the next addition a to maximize

    [ E_{q~Q0} ambiguity(C, q) - E_{q~Q0} ambiguity(C + a, q) ] / [ T(C + a) - T(C) ],

with deterministic ties and T measured by the actual tokenizer. This optimizes a proxy, not a guarantee of global optimum, and C never sees the held-out question or label. Freeze Q0 and weights on development data. Never drop certification fields silently to meet the budget.

**Small test today.** Use the existing observed-history tasks and current adaptive knots as baseline, plus uniform knots and raw values at matched token budgets. First run the current fixed four-choice likelihood evaluation unchanged for an apples-to-apples score. Then run a *separate*, explicitly defined selective task with an abstain option (and report both coverage and accuracy conditional on answering); do not compare those two metrics as if they used the same choice set. Compute a deterministic certificate-side reference without LLM. Report risk-coverage curves, unsupported decisive answers, reconstruction max error, latency, actual tokens and paired cluster-bootstrap intervals. For time reversals, test whether support and answers change consistently.

**Label boundary.** The implemented benchmark now derives observed labels directly from the visible samples, superseding an earlier metadata-based design. Certification above encloses observed samples, not latent event centers or the unseen clean signal. A certified numeric comparison is not by itself a guarantee that an LLM will follow it correctly; a global value-error bound also does not automatically certify every bend or duration question.

**Failure condition.** At equal token cost, supported-answer accuracy or risk-coverage is no better than ordinary knots/statistics; alternatively the bound is too loose to answer useful questions, or the LLM ignores it. Do not broaden claims from a small pilot. **Prior art:** Ramer/Douglas–Peucker, 2001 time-series segmentation and later error-bounded piecewise-linear compression already establish adaptive encoding and sampled error bounds; see [literature.md](literature.md). The hypothesis combines existing tools with query-family support and selective LLM evaluation; we have not established this combination as unique.

## 2. A predictive-innovation clock beside geometric knots

**Hypothesis.** A few timestamped *unexpected-change* events extracted by a cheap one-step predictor can disambiguate trend reversals, event order and duration that a visually similar point-budget reconstruction blurs. Its value should be separable from merely adding extra raw samples.

At each time t fit or update a fixed cheap predictor using only x_<t, compute residual r_t = x_t - xhat_{t|<t}, and normalize by a robust scale s_t computed strictly from earlier residuals. For example, z_t = r_t / max(s_t, floor). Form disjoint events using a fixed threshold tau and merge rule, and serialize (start, end, signed largest residual and accumulated magnitude) in time order; allocate budget between events and geometric knots. Warm-up residuals and zero scales need explicit behavior. A purely descriptive alternative is to retain signed first differences without any forecasting model.

Pre-register on development cases the parameters and a workload-weighted objective such as

    J = mean held-out task accuracy - lambda * (prompt tokens + alpha * measured encoder milliseconds),

with lambda/alpha chosen before final evaluation; report the raw components too. The event stream is query-independent and cannot use a future held-out question, target answer or latent synthetic parameters. It describes residuals under a specific model, not causes or objectively anomalous events. A TSFM-based innovation stream would require repeated prefix forecasts and extra latency; that variant is a later comparison, not assumed cheap.

**Small test today.** Existing event_order, trend, turn_timing and duration tasks provide a first test. At exactly the same evidence-token budget compare geometric knots alone, residual events alone, knots+events, uniform samples and knots+an equal-token random residual control; repeat with a second cheap predictor. Report each family separately, signed change under paired time reversal, and count extra predictor cost. Match numeric precision and number of timestamps. If current labels measure a latent event rather than observed data, report this explicitly: raw residuals may reflect noise, not the target.

**Failure condition.** Any gains vanish under equal-token random/first-difference controls, appear only on a development generator family, fail on reversal or require enough prefix-model compute to erase total latency gains. Some task families are insensitive to surprise and may lose useful amplitude information. **Prior art:** change-point detection, residual monitoring and time-series agent tooling are mature; [TimeSage-MT](https://arxiv.org/abs/2606.01498), 2026-05-31, is an agentic TS-analysis prior-art benchmark. This is a test of a particular allocation between geometric and predictive evidence, not a new invention of residuals.

## 3. Cost-aware routing with honest prequential accounting

**Hypothesis.** Where a TSFM helps only on some histories, a fixed, previously chosen routing policy can avoid most of its compute while retaining useful forecast-task accuracy. If it never improves on a classical forecaster, the economically correct outcome is to route entirely to the classical model.

For each case, let L_s and L_f be the same held-out loss for the cheap statistical and foundation forecast respectively, and let c_s and c_f be measured total runtime cost including initialization, compression and any routing validation. A policy pi maps only available history and fixed past information to {s,f}. Its intended objective is

    min_pi E[ L_pi + lambda * c_pi ],

with lambda fixed before the final test; compare the expected objective and its loss/cost components against always-s, always-f, a fixed random routing rate and a clairvoyant oracle lower bound. Forecast error and downstream LLM task loss are different objectives and must be reported separately. The oracle sees actual future only for retrospective upper-bound analysis; it is not a deployable route.

Two valid protocols have different economics:

1. **Offline fixed policy:** choose a simple threshold on pre-forecast history features using development histories and rolling-origin backtests; freeze it and invoke the TSFM only when it routes to f. The policy is tuning a rule, not training an adapter. Every feature must be available at the cutoff.
2. **Within-series prequential policy:** spend TSFM calls to compare earlier validation origins with the cheap method, using outcomes already observed by the current origin; choose a model for subsequent forecasts. Include *all* validation calls in cost, even if it routes to s. This can be amortized only when multiple later forecasts reuse the same explicit, immutable series/cutoff/model evidence. Never pick the best model using the test future. With a single request the validation overhead may dominate.

**Small test today.** Existing quarter-of-forecast-peak questions permit an offline retrospective comparison if both forecasters and their full cost are already being evaluated on identical cases. Freeze a simple development-only router, use the identical forecast codec for each forecaster, and run the selected route on held-out cases. Evaluate a cheap-only policy as the primary economic baseline. Keep saved capsule artifacts keyed by explicit series, cutoff, model and configuration; account for cold/warm runs separately. When there are few real chronological cases, state that confidence intervals are wide and report the individual paired outcomes.

**Failure condition.** A held-out policy does not improve the predeclared joint objective over the best fixed cheap/foundation choice, its marginal gains disappear after validation or initialization cost, or it is unstable under a chronological shift. Forecast calibration cannot be inherited from the selected model without checking route-conditional coverage. If the gate chooses cheap almost always and cost falls without accuracy loss, that is a useful negative result for TSFM necessity, not evidence that routing uncovered temporal intelligence. **Prior art:** [Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](https://arxiv.org/abs/2510.06419), 2025-10-07, studies forecasting model portfolios and selection; [LLM as Forecasting Planner](https://arxiv.org/abs/2607.24892), 2026-07-27, uses frozen forecaster proposals with LLM decisions. Neither establishes novelty for routing; both are abstract-level checks here.

## What to do first

The existing experiment and its frozen selection take priority; do not alter its already selected protocol or test set to chase these hypotheses. If there is time for an additional, separately tracked pilot today, mechanism 1 is the closest fit to observed event-order and timing questions and does not depend on TSFM strength. Start with a deterministic sample-level reconstruction certificate and a separately labeled sanity set; LLM interpretation and abstention are follow-on comparisons. Mechanism 2 is feasible with cheap predictors but needs careful event-budget and labeling controls. Mechanism 3 is feasible only after measured forecast outputs exist, and a routing policy needs independent development/held-out data; do not claim a saving from reused precomputed evaluations without charging all the real calls.
