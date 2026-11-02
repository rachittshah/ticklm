# TickLM: can we give a language model a live sense of time?

This report covers the original local Qwen-and-TimesFM experiment. The later [Claude Code and Codex trials](frontier-results.md) are recorded separately, including their failed freshness checks.

**TickLM is a working prototype that helps an AI assistant keep track of passing time and changes while it is busy.** It runs locally and does not require retraining either model.

The useful result is straightforward: a small program watches the clock and incoming information, then gives the language model an up-to-date account of what is happening. If the situation changes while the model is writing, the program can stop that old answer from reaching the user.

The model is using a clock supplied to it. It has not learned to keep time independently.

## What we were trying to achieve

An assistant should be able to notice things such as:

- “I have been waiting for five seconds.”
- “That deadline has already passed.”
- “My last measurement is now too old to rely on.”
- “An alarm arrived while I was answering, so my answer needs to change.”

We also wanted to explore whether TimesFM—a model that predicts how measurements change over time—could help the assistant decide when to check again.

These are different jobs. An ordinary clock measures elapsed time directly. A forecasting model is useful only if its predictions help the assistant anticipate a change.

## Models and machine used

The final combined run, **`live-fused-20261102-b`**, ran on an **Apple M4 Max Mac**, using Python 3.12, MLX 0.32.4, and MLX-LM 0.32.2. Both models were used with existing weights; there was no training or fine-tuning for this experiment.

| Component | What we actually ran | Role in this experiment |
|---|---|---|
| Language model | **Qwen3-4B abliterated**, a roughly four-billion-parameter model, converted to MLX's 4-bit format | Read the supplied clock and event snapshot and generate an answer. |
| Time-series model | **Google TimesFM 3.0**, checkpoint `google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2` | Forecast future numerical values from recent measurements. |
| Forecast implementation | [Rachitt's `mlx-tsfm`](https://github.com/rachittshah/mlx-tsfm), using `TimesFM3.forecast_batch` | Run the TimesFM checkpoint locally through MLX; the checkpoint repository's name does not mean we used PyTorch for inference. |
| Clock and controller | Python's `time.monotonic()` and this repository's `TemporalState` and `LiveRuntime` | Calculate elapsed time, collect observations, fire deadlines, prepare model context, and reject expired or superseded answers. |
| Cheap forecast comparison | A straight-line predictor using the last 16 input values, labeled `linear-16` | Provide a low-cost comparison for the suggested time of the next check. |

The language model was loaded from `/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit`. The run's [manifest](../evidence/live-fused-20261102-b/manifest.json) records hashes of its weights, tokenizer and configuration, rather than an upstream language-model revision. The [startup record](../evidence/live-fused-20261102-b/trace.jsonl#L3) records the TimesFM checkpoint revision and weight hash. TimesFM's non-commercial model license applies.

## How we measured the time between A and B

**Python calculated the time difference. Qwen read and reported that calculated difference.** TimesFM was not involved in this calculation.

Point A was the start of the live run, after model warmup. Point B was the instant the controller prepared the context for the elapsed-time question. Both readings came from the same monotonic clock: a clock intended for measuring intervals rather than displaying a date or time of day.

| Reading | Recorded value |
|---|---|
| A: start of the live run, derived from the recorded snapshot | 49066.418777208 seconds on the monotonic clock |
| B: preparation of Qwen's snapshot | 49070.946509750 seconds on the same clock |
| B minus A, calculated by the controller | **4.527732542 seconds** |
| Value written into the prompt, rounded to three decimals | **4.528 seconds** |
| Qwen's response | **`4.528`** |

The relevant prompt excerpt was:

```text
Run elapsed time: 4.528 seconds since run start, measured with a monotonic clock.

How many seconds has this run been active, as of this clock snapshot?
Reply with only the decimal number, no units.
```

This shows Qwen could report the supplied elapsed time. It is not an independent estimate of elapsed time, an arithmetic test, or evidence that the model counts seconds between calls. The close numerical match is expected because the answer was already present in the prompt. Deadline status and measurement freshness were also explicitly supplied; their successful answers test use of that context.

By the time the answer was published, the run had been active for about **4.791 seconds**. The answer still described the earlier snapshot at B. We retain both times so generation delay does not silently turn an old reading into a claim about the present. See the [exact request](../evidence/live-fused-20261102-b/trace.jsonl#L57), [model completion](../evidence/live-fused-20261102-b/trace.jsonl#L63), and [publication record](../evidence/live-fused-20261102-b/trace.jsonl#L64). A is recoverable from the recorded snapshot reading minus its recorded elapsed time.

## What the 201 measurements mean

The live portion of the run lasted **24.691 seconds**. The controller received **201 simulated numerical readings**, made **7 language-model calls**, and requested **1 TimesFM forecast**. Those are three different counts.

Each reading was one number from a simulated sensor, generated approximately every **0.1 seconds** while the test was running. The value rose gradually, varied smoothly, and jumped by eight units when the alarm was introduced. Its units were arbitrary; it was not a physical temperature, load or other device measurement. The exact generating rule was:

```text
value = 50 + 0.25 × elapsed_seconds + 3 × sin(elapsed_seconds / 4)
During ALARM, add 8 to the value.
```

We deliberately stopped producing readings for **three seconds** to test old information. Scheduling overhead also makes the arrival spacing approximate. Therefore, 201 is the actual count received, not an assumed rate multiplied by the run length. Each arrival has its own timestamp and value in the [trace](../evidence/live-fused-20261102-b/trace.jsonl); the generator is in [realtime.py](../realtime.py#L224).

The values were simulated, but the waits, generation, arrivals, deadlines and answer rejection took place as real time passed. A separate integration test used events appended to an actual file during generation.

## What the seven Qwen calls tested

| Call | Information or trigger | Recorded outcome |
|---|---|---|
| 1. Old-context control | A deliberately frozen snapshot from before the three-second deadline expired | `PENDING`. This was a control, not a current answer. |
| 2. Fresh deadline question | A new snapshot after the deadline expired | `EXPIRED`. |
| 3. Elapsed-time question | The clock snapshot described above | `4.528`. |
| 4. Longer answer during a change | Began with NORMAL mode; an alarm arrived after the first generated token | The response still described NORMAL and reached its 200-token limit. The controller rejected it because the situation had changed. |
| 5. Refreshed answer | A new snapshot containing the alarm and expired urgent deadline | `ALARM; EXPIRED`. |
| 6. Old-measurement question | The measurement source was paused and the last reading was over one second old | `STALE`. |
| 7. Forecast-triggered check | TimesFM's suggested timer fired; Qwen received a fresh clock and event snapshot | Reported a snapshot elapsed time of 23.992 seconds, a fresh observation, and ALARM mode. |

The urgent deadline in call 4 was set for **250 milliseconds after the alarm**. The controller delivered it while Qwen was still generating, rather than waiting for the answer to finish. The test also checked that a request could expire merely because time passed, without a new event; that check did not require another language-model call. The [summary's `results` entries](../evidence/live-fused-20261102-b/summary.json) retain the model responses and validity decisions.

## Exactly how TimesFM used the stream

TimesFM predicted **future measurement values**. The controller converted those predictions into a time to check again:

1. **Collect enough recent history.** At forecast submission, 185 readings had arrived; the total reached 201 by the end of the run. The controller used linear interpolation between recent observed readings to make 128 input values at exactly 0.1-second spacing. The first-to-last input interval was 12.7 seconds. It refused to fill any source gap larger than 0.3 seconds; the deliberate three-second pause was outside this input window.
2. **Predict the next values.** TimesFM received those 128 numbers and returned 64 future point estimates, plus lower and upper estimates at the 10th and 90th percentiles. At the controller's 0.1-second cadence, the forecast horizon was 6.4 seconds. TimesFM received a numerical sequence, not the A and B clock readings.
3. **Turn a predicted change into a timer.** A simple rule looked for the first point estimate at least one unit above or below the last input value, about 62.12846. The first such prediction was step 11: **1.1 seconds** after the last input reading. The rule bounded any suggested interval to 0.25–3 seconds. It used the point estimates only; the percentile estimates did not affect scheduling and were not interpreted as event probabilities.
4. **Wake Qwen with current information.** The controller anchored the timer to the last input reading's timestamp. In this run, the forecast finished before the intended wake time, so its computation did not add extra waiting. A forecast that finished too late could delay the check. The timer fired about **0.78 milliseconds** after its scheduled time. Qwen then received a new clock, observation and event snapshot.

The forecast itself was **not inserted into Qwen's prompt**, and no model weights or hidden representations were combined. Here, the connection between the models was the timer: TimesFM influenced **when** Qwen ran; the controller supplied **what Qwen knew** at that time. Incoming events and deadlines did not have to wait for the forecast timer.

Inspect the [forecast input](../evidence/live-fused-20261102-b/trace.jsonl#L224), [scheduled timer](../evidence/live-fused-20261102-b/trace.jsonl#L228), and [timer firing](../evidence/live-fused-20261102-b/trace.jsonl#L239). The input preparation and scheduling rule are in [realtime.py](../realtime.py#L36).

## Durations, deadlines and timeout limits

These numbers refer to different things:

| Quantity | Value in this experiment | Meaning |
|---|---|---|
| Model warmup after worker launch | 1.749 seconds | Recorded warmup wait before the live run's clock started; excluded from the 24.691-second run. |
| Live run duration | 24.691 seconds | Observed duration of the scripted combined test. |
| Initial deadline | 3 seconds after it was set near run start | A timer the controller had to deliver. |
| Urgent deadline | 0.25 seconds after the alarm | A timer that had to fire during generation. |
| Measurement freshness limit | 1 second | Older readings were labeled STALE. |
| Intentional source pause | 3 seconds | A test condition, not a failure or timeout. |
| Model warmup safety limit | 90 seconds | Maximum warmup wait after worker launch; it was not reached. |
| Live test safety limit | 70 seconds | Maximum allowed live test duration; it was not reached. |

**The retained records do not establish a 21-minute timeout or a 21-minute live test.** The final summary does contain a deadline overdue age of approximately **21.69 seconds**: the run ended that long after the initial three-second deadline. That number is time since an already delivered deadline, not the delivery delay or a timeout. The deadline was delivered **4.33 milliseconds after its scheduled time**. See the [run summary](../evidence/live-fused-20261102-b/summary.json) and the safety limits in [realtime.py](../realtime.py#L90).

## What passed

| Test | Result | What it establishes |
|---|---|---|
| Clock and event handling, without TimesFM | 10 of 10 scripted checks passed | The system can use a clock, track changes, reject an old answer, and refresh its response. |
| Clock and event handling, with TimesFM | 12 of 12 scripted checks passed | The forecasting model can run on newly collected data and schedule a later check; these include the ten clock/event checks. |
| Events written to a real file during generation | Passed | The public watcher can receive an event and deadline, reject an outdated answer, and later display a valid one. |
| Automated software tests | 91 passed | The tested clock, timing, input, and answer-validity rules behave as specified. |

These are results from short, scripted tests. “12 of 12” is not a claim of perfect reliability in ordinary conversations.

## Is TimesFM worth the extra work?

**We do not know yet.**

The clock-only version already performs the basic timekeeping job. It does not need a neural model to count seconds, notice that a deadline passed, or record when information arrived.

In the final test, TimesFM took **67.39 milliseconds** to produce its forecast. The cheap comparison took **3.71 milliseconds**, including selection among simple predictors using a held-back portion of the input history; the selected predictor was `linear-16`. TimesFM suggested checking again after **1.1 seconds**; the cheap predictor suggested **1.3 seconds**. We used TimesFM's suggestion, but this one comparison does not tell us which was better.

The TimesFM timing includes the first execution of this input shape in the worker and excludes model loading. It is not a repeated steady-state benchmark. We did not compare future prediction accuracy, energy consumption, missed changes, or total language-model cost under the two scheduling policies. The 7 Qwen calls include scripted probes, so that count is not a measured saving from forecast-based scheduling.

The next research question is whether prediction helps the assistant catch changes sooner or make fewer unnecessary model calls. That needs a comparison on the same streams, with the same limits on missed events and response delay. We have not completed that comparison.

For now, the cheapest demonstrated approach is the clock-and-event program. TimesFM is a working optional addition whose benefit remains unproven.

## What went wrong along the way

The first live attempt handled the clock and incoming events correctly, but the language model gave wrong answers about expired deadlines and old measurements. It struggled with the compact technical fields we supplied.

We made those fields explicit: how long the run had been active, how old the last measurement was, and whether a deadline was pending or expired. We also clarified an ambiguous question about whether a deadline was “due now.” The later runs passed after those changes.

We kept the failed runs. They show that simply attaching timestamps to a model is not enough. How the information is presented matters, and the software must enforce timing rules rather than trusting every generated answer.

The earlier static forecasting study was also a detour: it tested questions about numerical sequences, which did not answer the request for awareness of time as it passes. Its results are retained separately.

## What this does—and does not—mean

The combined system now has a practical way to keep time, receive changes while the model is busy, and check whether an answer is still usable.

The language model itself is unchanged. It receives current time information when called; it does not continuously think or maintain an independent internal clock between calls. We have not established a new scientific breakthrough, broad temporal intelligence, or an advantage over all existing approaches. The [research review](realtime-research.md) documents related work.

There are also practical limits. For file input, the program measures age from when it reads a record. It cannot tell how long that record waited before being read, so the original measurement's age may be unknown. Relative deadlines start when read. Replacing or shortening the input file can cause records to be missed. We have not established reliable operation through machine sleep, production restarts, or arbitrary long-running conversations.

## Read the evidence

The records below include the exact prompts, clock readings, arriving events, model responses, and decisions to display or reject an answer. Each run identifies the code and model files it used.

- [Final clock-only trial](../evidence/live-clock-20261102-d/summary.json)
- [Final trial with TimesFM](../evidence/live-fused-20261102-b/summary.json)
- [Real file-watcher test output](../evidence/live-watch-20261102-b/test-output.log) and [displayed output](../evidence/live-watch-20261102-b/stdout.log)
- [First failed live trial](../evidence/live-clock-20261102-a/summary.json) and [second attempt](../evidence/live-clock-20261102-b/summary.json)
- [Automated test results](../evidence/verification/pytest.txt)
- [Independent review](realtime-final-review.md)
- [Earlier static-study failure analysis](failure-analysis-v1.md)

<details>
<summary>Additional timing details</summary>

In the final TimesFM trial, the two deadlines were handled 4.33 milliseconds and 0.45 milliseconds after their scheduled times. Ninety-five percent of the recorded gaps between checks were 12.82 milliseconds or shorter. These are measurements from one run, not guarantees about the maximum possible delay.

The TimesFM forecast took 67.39 milliseconds, including the first execution of that input shape in that process. The simple predictor took 3.71 milliseconds. The forecast used 128 regularly spaced values constructed from the measurements received during the live run. The program refuses to fill a long gap in missing data merely to make that input look regular.

Displayed answers are labeled with the time of the information they used. The current clock is reported separately. That distinction matters: showing a new timestamp beside an old answer does not make the old answer current.

</details>
