# TickLM with Claude Code and Codex

**Both clients now work with TickLM, and we ran the live experiment through each.** Claude launched through `cca startup`. Both models answered the four graded questions correctly in both of their runs. Neither passed the complete freshness test: some correct answers arrived after their information was already out of date.

This is an integration result and a timing experiment, not a general ranking of the models. There was one clock-only run and one TimesFM-assisted run per client on November 2, 2026. The original [local Qwen experiment](live-results.md) is a separate result.

## What we actually ran

| Part | Actual setup |
|---|---|
| Claude | Claude Code **2.1.316**, launched through **`cca startup --safe`**, with `--safe-mode`; the observed model was **`claude-opus-5-5`**. |
| Codex | Codex CLI **0.165.0**, using its streaming app server and existing local login; the server selected **`gpt-6-astra`**. |
| Effort | **`xhigh` requested for both.** This records the client setting, not independent confirmation of the provider's internal computation. |
| Clock and event loop | The same local Python controller on the Apple M4 Max. |
| Optional forecast model | Local **Google TimesFM 3.0** through MLX, checkpoint revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. |

Each question started a fresh conversation in an isolated working directory. We sent the same scripted user questions and temporal context format; each client still had its own system instructions and transport overhead. Claude's built-in tools and customizations were disabled. Codex disabled relevant execution features and used a read-only sandbox; unexpected tool or approval events fail the adapter. These were prompt-response tests, not coding tasks.

We used the clients' existing authentication. Credentials and account details are not retained in the evidence. Claude's startup profile is a local launcher choice, not a different model.

## The four completed trials

| Run | Time after warmup | Simulated readings received | Model calls | Correct answers to four graded questions | Publishable answers among those four | Overall result |
|---|---:|---:|---:|---:|---:|---|
| [Claude, clock only](../evidence/frontier-claude-clock-20261102-a/summary.json) | 46.54 s | 409 | 6 | 4 / 4 | 3 / 4 | Failed freshness check |
| [Codex, clock only](../evidence/frontier-codex-clock-20261102-a/summary.json) | 71.66 s | 650 | 6 | 4 / 4 | 2 / 4 | Failed freshness check; another reply expired |
| [Claude + TimesFM](../evidence/frontier-claude-fused-20261102-a/summary.json) | 58.17 s | 515 | 7 | 4 / 4 | 3 / 4 | Failed freshness check |
| [Codex + TimesFM](../evidence/frontier-codex-fused-20261102-a/summary.json) | 143.03 s | 1,328 | 7 | 4 / 4 | 2 / 4 | Failed freshness check; another reply expired |

The four graded questions ask for the deadline status, elapsed time, updated alarm state, and whether the snapshot's last reading is stale. The other calls are a deliberately old-context control, a long answer during which we inject an alarm, and, for the TimesFM runs, a forecast-triggered status report. “4 / 4” does not grade every sentence of the longer answers.

As in the original test, Python generated roughly one simulated reading every tenth of a second, with an intentional three-second pause. The input counts differ because the runs took different amounts of real time. The values depend on elapsed time and alarm state, so these were the same generator and protocol, **not identical recorded input sequences**. The different durations are not evidence that TimesFM itself made a client faster or slower.

The original scripted checks passed 9 of 10 without TimesFM and 11 of 12 with it for both clients. The missing check required the stale-reading answer to remain publishable. The final aggregate also requires all four ordinary graded answers to be publishable, rather than treating correct but expired text as success. All four runs report `all_checks_passed: false` and exit with status 1 after preserving their results. That exit status reports the failed test expectation; the client connections themselves worked.

## How the models used the clock

The controller still measured time. For example, in the Claude-plus-TimesFM run:

```text
A: start of the live run
B: time the controller prepared the elapsed-time question

Python's clock calculation, B minus A: 12.021795042 seconds
Value supplied in the prompt:         12.022 seconds
Claude's answer:                      12.022
Time taken to complete that answer:    4.024 seconds
```

Codex's corresponding answer was `20.294`, against a supplied snapshot elapsed time of 20.294241541 seconds. Its answer completed about 3.771 seconds after the snapshot. Both reported the elapsed value already provided to them. This is not evidence that either model independently counts seconds or measures time between prompts.

The snapshots were prepared at different points because earlier client requests took different amounts of time. Each answer refers to its own snapshot, not the later publication instant.

## What happened while they were answering

For the long-answer test, the controller waited until it received **visible answer text**, then changed NORMAL to ALARM and set a deadline for 250 milliseconds later. In all four trials:

1. The alarm arrived between the first visible text and the answer's completion.
2. The controller delivered the urgent deadline before the final visible text fragment arrived from the client.
3. The old answer was rejected.
4. A new request answered `ALARM; EXPIRED`, and that refreshed answer was publishable.

In the TimesFM runs, the urgent deadline was delivered **10.96 milliseconds late with Claude** and **6.69 milliseconds late with Codex**. Those are observations from these runs, not maximum-delay guarantees. The controller continued receiving readings during the longer model calls.

"First visible text" is the time TickLM received a stream event from the local client. The retained traces place each urgent deadline between the first and last visible text fragments of the long answer. These are not timestamps from inside the model or proof of exactly when the provider generated those tokens. Reasoning events and request-start messages do not count as visible answer text.

## Why correct answers were rejected

The test retained the original rules: ordinary answers expire after **six seconds**, and an observation-bound answer becomes invalid if its underlying reading changes. A reading older than **one second** is stale. The 240-second post-warmup test limit merely allowed enough time for all cloud calls; it did not relax these rules.

The freshness question was asked about **1.6 seconds into a three-second pause**. Only about 1.4 seconds remained before readings resumed. Both clients correctly answered `STALE` about their supplied snapshot, but the answer finished after new readings arrived. Publishing it as the current situation would have been wrong, so TickLM rejected it.

Some answers also exceeded the six-second validity window:

| Example | Completion time from request snapshot | What TickLM did |
|---|---:|---|
| Codex clock-only elapsed-time answer | 6.28 s | Rejected the correct number because its request expired. |
| Codex + TimesFM deadline answer | 9.38 s | Rejected `EXPIRED` because its request expired. |
| Codex + TimesFM stale-reading answer | 70.76 s | Rejected it; its request had long expired and the stream had resumed. |
| Claude + TimesFM final status report | 6.47 s | Rejected the forecast-triggered answer because its request expired. |
| Codex + TimesFM final status report | 8.17 s | Rejected the forecast-triggered answer because its request expired. |

The 70.76-second call is an observed outlier, not an established model speed. The measured interval includes client work, network delays, provider waiting and inference; this experiment does not separate their contributions. The final status reports are outside the four-question publication counts in the earlier table, and both were rejected.

## What TimesFM did in these runs

Both integrations exercised the real local forecasting path. The controller constructed 128 regularly spaced values from recent live arrivals, asked TimesFM for 64 future values, and applied the existing one-unit-change rule to choose a bounded time for another check. Each scheduled timer fired and started another language-model call.

| Measurement | Claude + TimesFM | Codex + TimesFM |
|---|---:|---:|
| TimesFM forecast computation | 74.10 ms | 90.72 ms |
| Cheap predictor computation | 3.70 ms | 3.75 ms |
| TimesFM's next-check interval | 1.1 s | 3.0 s |
| Cheap predictor's proposed interval | 1.1 s | 2.4 s |
| Forecast-triggered call completed | Yes | Yes |
| That answer remained publishable | No | No |

The intervals start at the last input reading, rather than at the end of forecast computation. Three seconds is the scheduler's maximum interval; it does not promise that a change will happen then. The cheap predictor was `linear-16` in both runs.

TimesFM affected **when the next call started**. It did not shorten the cloud client's response time, supply the elapsed-time calculation, or place a forecast inside the language-model prompt. We have not demonstrated a forecasting accuracy advantage, a reduction in calls, or lower overall compute cost.

## What we can say about cost

No models were trained. The local clock and event loop did not call a language model on every tick. Clock-only trials used six model calls; TimesFM trials used seven because the script explicitly added a forecast-triggered check. Those call counts are not measured savings against a polling baseline.

The clients do not enforce the runner's requested output-token cap. The long answers therefore differed in length and reported token usage; this is another reason not to treat this as an equal-compute model comparison.

Claude's `input_tokens` field counts newly supplied uncached tokens, not the complete context. For example, its clock-only stale-reading call reported 2 fresh input tokens, 1,647 cache-creation tokens and 2,294 cache-read tokens: **3,943 input tokens in total**. The full counts are in each result's `input_tokens_total`, `cache_creation_input_tokens` and `cache_read_input_tokens`. Codex's reported input counts were roughly 17,000 per request in these runs, including its client-provided context. Provider token fields are retained as reported and are not all directly comparable.

Claude Code reported cost estimates of about **$0.121** for the clock-only trial and **$0.126** for the TimesFM trial. These are CLI estimates, not verified charges to the subscription account. Codex did not supply a dollar estimate; its cost is **unknown, not zero**.

## Run it again

From the repository, using the existing environment and signed-in clients:

```bash
.venv/bin/python realtime.py self-test --provider claude \
  --model claude-opus-5-5 --claude-profile startup --effort xhigh \
  --clock-only --trial-timeout-s 240 --run-dir artifacts/my-claude-clock

.venv/bin/python realtime.py self-test --provider codex \
  --model gpt-6-astra --effort xhigh \
  --clock-only --trial-timeout-s 240 --run-dir artifacts/my-codex-clock
```

Omit `--clock-only` and choose a new run directory to include TimesFM. The `watch` command accepts the same provider options for an actual incoming event file. Model access depends on the signed-in account; the adapter records the observed identity or fails instead of inventing one.

The implementation uses the documented [Claude Code streaming output](https://code.claude.com/docs/en/headless#stream-responses) and [Codex app-server events](https://learn.chatgpt.com/docs/app-server). The adapters are in [claude_backend.py](../src/temporal_bridge/claude_backend.py) and [codex_backend.py](../src/temporal_bridge/codex_backend.py).

## Evidence and what remains

Each linked run directory contains the manifest, full trace, summary, console output and explicit failed-test record. Failures have not been discarded. The initial Codex clock-only run preceded the final publication-aggregate and metadata wording changes; its manifest identifies that source. Its failed overall outcome is unchanged. The later combined runs use the stricter publication aggregate. Source fingerprints belong to individual runs, not an assumed single version across all development trials.

The adapter and controller verification finished with **127 passed, 1 skipped**. The skipped test is the existing opt-in local-MLX watcher test, not a skipped frontier trial. Ruff and `git diff --check` passed. Independent review covered publication grading, visible-text timing, process cleanup and truthful model identity. The controller and provider source hashes used by each run are retained.

The next experiment should measure a useful latency policy: display deterministic current clock/event facts immediately, let the frontier model provide a clearly dated explanation, and use or retry model output only when its underlying facts remain valid. Lower effort and shorter answers are also candidates for measurement. None of those changes has been tested here. The present result establishes that frontier clients can participate in the live loop, and that the clock and validity checks remain necessary while they respond.
