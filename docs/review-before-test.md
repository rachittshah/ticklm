# Scientific review before held-out scoring

Date: 2026-11-02. This review is written before the untouched held-out run. It does not inspect or interpret held-out outcomes. It sets boundaries for reading the result, not a prediction of success.

## What this experiment can test

There are two separate questions:

- **Observed-history evidence:** Does a selected representation help the frozen Qwen3-4B 4-bit model answer the benchmark's four-choice questions about generated histories, compared with raw or uniform history under the stated token budget? The codec is the treatment here. Because the observed cases do not use TimesFM forecasts, this result cannot show a TimesFM contribution.
- **Forecast-assisted decisions:** Does the full TimesFM forecast-capsule pipeline improve the specified future-maximum-bin choice over the cheap-forecast pipeline? This is a system-level forecast-assisted result. The test does not show that the LLM learned temporal cognition, nor does it show that TimesFM alone is better than the cheap forecaster on other forecasting metrics.

The forecast question asks which six-hour bin contains the maximum over the next 24 hourly samples. That is narrow and easy to score; it does not test multi-step event reasoning, calendar knowledge, causal inference, or use of irregular time. The no-LLM direct forecast decision is necessary to show how much of the answer is already present in the predicted curve.

## Scientific caveats

1. **Question and model scope.** Forced-choice token likelihood measures selection among the supplied choices. It is not free-form answer quality, calibrated confidence, or open-ended reasoning. Report each task separately, include the question-only prior control, and label any free-generation examples as illustrative.

2. **Synthetic labels and interventions.** The current benchmark builder derives observed labels from the visible context and states the measured feature in each question. The earlier note in [research-next.md](research-next.md) about labels taken from generator metadata describes an earlier design and is superseded for these cases. Still inspect the frozen prompts and case metadata before scoring: answers and generation parameters must stay out of the public view. Reversal partners and time-dilation partners are related examples, not independent observations; keep them in the same split and statistical cluster. These interventions test order and clock-scale sensitivity, not a general ability to understand time.

3. **Real-data dependence and contamination.** The real task uses 18 development and 36 held-out non-overlapping, guarded windows from the OT channel of one ETTh1 series. Windows from that series can remain correlated, and bootstrap uncertainty across windows does not establish dataset-level generalization. TimesFM3's pretraining overlap with ETTh1 cannot be ruled out from the documented source lists. Treat this arm as an exploratory real-series check, not a contamination-free benchmark or broad out-of-domain result.

4. **Forecast comparison is a pipeline ablation.** Adaptive knot allocation can change the historical representation when the forecast curve changes. Therefore, cheap-forecast versus TimesFM-forecast comparisons measure the full capsule pipeline, including any allocation changes; they do not isolate the forecast model's effect while holding every historical token fixed. State this in the results. The mismatched-forecast arm is a useful alignment control, but it does not eliminate every prompt or representation confound.

5. **Compression and forecast uncertainty.** The interpolation-error certificate describes reconstruction of sampled numeric curves after serialization. It does not certify the correctness of a QA answer or the forecast. TimesFM quantiles are marginal at each horizon; they are not a joint path distribution, a probability of crossing a threshold, or evidence of calibrated coverage. Report repaired quantile crossings and the actual displayed values.

6. **Clock and compute claims.** Hourly timestamps and time dilation test an explicit sampling interval, not calendar or seasonal cognition. Report actual evidence tokens and full prompt tokens separately, since the evidence budget excludes the shared chat template. Separate warm inference, first load/compile, precomputed-forecast timing, and incremental uncached TSFM cost. A small pilot's latency is local to this hardware and configuration.

Development selects among uniform/adaptive codecs at 256/384/512 evidence-token caps by highest total accuracy, breaking ties with lower prompt-token use. With only 36 development cases, this selection is noisy and creates winner's-curse risk; retain and report the full candidate grid. Held-out arms are question-only, raw history, uniform history, statistics, cheap forecast, TimesFM forecast, and mismatched TimesFM forecast. Forecast-arm comparisons are not interchangeable with observed-history results.

The benchmark literature supports these limits: [MMTS-Bench](https://arxiv.org/abs/2602.08588) finds temporal-reasoning scores below feature-analysis scores and reports gains from statistical summaries; [TemporalBench](https://arxiv.org/abs/2602.13272) separates historical understanding from forecasting and finds they can diverge. TimesFM-to-LLM fusion, time-series soft tokens, and on-device time-series reasoning already have prior work, including [TS-Reasoner](https://arxiv.org/abs/2510.03519) and [PocketTSLM](https://arxiv.org/abs/2610.21308). Do not claim a novel fusion architecture or newly acquired model capability from this pilot.

## Checklist before writing the result

- [ ] Freeze and record the selected codec/budget, full development candidate grid, case IDs, source hashes, TSFM checkpoint revision, Qwen weights/tokenizer, code revision, and package versions before held-out scoring.
- [ ] Confirm logged prompts contain only the intended context/evidence, question, and options; no future values, answer labels, generator parameters, split IDs, or scoring metadata.
- [ ] Verify answer parsing and direct forecast scoring against the displayed options; report option-position balance and the question-only prior.
- [ ] Report per-task accuracy and paired uncertainty. Keep synthetic reversal/dilation partners clustered. Describe ETTh1 uncertainty as window-level evidence from one source series.
- [ ] Report actual token counts, compression error, quantile repairs, synchronized warm p50/p95 latency, load/compile cost, and measured incremental TSFM time. Do not infer quantile calibration from this QA task.
- [ ] Separate observed-history codec claims from forecast-assisted system claims. If uncertainty is wide or a criterion is missed, say “inconclusive” or report the negative result; do not tune against held-out cases.

Reasonable claim wording, if supported by the frozen held-out results: “Under this local setup and these specified tasks, the selected capsule changed four-choice accuracy and token/latency cost relative to the tested baselines.” Name the model checkpoint, task family, source series, and sample size. Avoid claims about general temporal intelligence, real-world decisions, or broad forecasting quality.
