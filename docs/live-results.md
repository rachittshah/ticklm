# Live-time proof of concept: measured result

Date: 2026-11-02. Hardware: Apple M4 Max, 36 GiB unified memory. Local language model: Qwen3-4B abliterated, 4-bit MLX. Optional predictor: Rachitt's MLX TimesFM 3 implementation, pinned checkpoint `43046b85ec22d584a13f8098c2ed39c889e129c2`. Neither model was trained or modified.

**The corrected prototype works in the scripted live trials.** It keeps track of actual elapsed time, observation age, deadlines and incoming changes while the language model generates. It invalidates an obsolete answer and supplies refreshed context for the next one. The TimesFM path also consumes a live-derived history and schedules a later wake.

This is an external runtime giving a frozen model current temporal facts. It is not a clock learned inside the weights, a general temporal-reasoning benchmark, or evidence that TimesFM improves scheduling over cheap rules.

## Final evidence

| Trial | What actually ran | Outcome |
|---|---|---|
| [Clock-only D](../evidence/live-clock-20261102-d/summary.json) | Real wait, refreshed clock, numerical elapsed answer, first-token-triggered alarm, urgent deadline, stale-result rejection, refreshed answer, source pause and expiry | 10/10 scripted checks, 8.85 seconds |
| [Clock + TimesFM B](../evidence/live-fused-20261102-b/summary.json) | Same clock/event trial, then enough live arrivals to construct a regular 128-point history, actual TimesFM inference and execution of its proposed wake | 12/12 scripted checks, 24.69 seconds, 201 sample arrivals, 7 LLM calls |
| [Real file watcher B](../evidence/live-watch-20261102-b/test-output.log) | Public CLI, actual append-only file events during generation, file deadline, rejection without exposing stale text, later valid publication | Real-model integration test passed |

The TimesFM trial's measured details:

- The model answered **4.528 seconds** for a captured elapsed time of **4.527732542 seconds**. This demonstrates reading the supplied clock, not independently estimating time from its tokens.
- The deliberately stale frozen context said the original deadline was `PENDING`; refreshed context produced `EXPIRED`.
- An alarm arrived after the first actual generated token. The urgent deadline fired during that generation. The old answer was rejected for `state_changed`; the refreshed answer was `ALARM; EXPIRED`.
- Deadline dispatch was observed 4.33 ms and 0.45 ms after the two scheduled times. Event-loop interval p95 was 12.82 ms. These are observations from this run, not worst-case guarantees.
- After a genuine pause in sample arrival, the model identified the sample as `STALE`. A separate ticket also expired with no new semantic event.
- The one TimesFM forecast call took **67.39 ms**, including this process's first execution of that shape. The cheap predictor took **3.71 ms**. TimesFM suggested a 1.1-second attention interval; the cheap predictor suggested 1.3 seconds. The TimesFM wake was executed. This single comparison does not show that its wake was better.

The live sensor in the self-test is simulated, with each value computed when its actual wall-clock arrival occurs. There is no preloaded numerical history. The forecast input is **128 regularly resampled values derived from those live samples**, rather than an assertion that every sample arrived at an exact 100 ms interval. A long missing-data gap is rejected before resampling. The separate watcher test uses actual file writes.

Every successful and rejected completion is in the diagnostic JSONL trace, with explicit request identity, snapshot, first-token time, event time and publication decision. User-facing watcher output suppresses rejected text. Narratives carry an as-of-snapshot timestamp; the publication-time clock is a separate, fresh object.

## Development failures are retained

- [Clock A](../evidence/live-clock-20261102-a/summary.json): timer and concurrency mechanics worked, but the model missed all three deadline/freshness answers. Compact fields such as `fired` did not produce reliable interpretations.
- [Clock B](../evidence/live-clock-20261102-b/summary.json): explicit time meanings fixed the elapsed, stale-observation and refreshed-alarm answers; the deadline question still failed. The ambiguous “due now?” wording was replaced with an explicit request for the deadline's status.
- [Clock C](../evidence/live-clock-20261102-c/summary.json) and [fused A](../evidence/live-fused-20261102-a/summary.json): passed their scripted checks. Further review then tightened watcher publication, request-identity copying and source intake handling. D and fused B reran the core behavior with those fixes.

These are iterative debugging trials with different implementations, not independent draws that can be combined into a reliability percentage. Unit tests separately reproduce UTC jumps, deadline crossings between ticks, observation updates, freshness transitions, restarts, expired tickets and mutated ticket identities.

## What is cheapest, and what remains research

Exact timekeeping needs no neural model. The default recommendation for passing time, observation receipt age and deadlines is the clock-only runtime. It invokes the LLM at relevant boundaries rather than spending model inference to count time.

TimesFM is an optional anticipatory component. The implementation proves that a frozen numerical predictor can participate in the live loop without blocking event intake. It does **not** yet establish fewer LLM calls, better event detection, lower overall cost or better answers than a cheap predictor. A meaningful next experiment would compare schedules on identical recorded streams with fixed limits on missed events and response delay, then measure inference calls and total energy/latency. That experiment is not claimed complete here.

The research hypothesis is a validity-limited temporal context: the runtime supplies typed clock facts and predicted dynamics, but a result is usable only while its time and event dependencies remain valid. Related clock-aware and asynchronous systems already exist. The [real-time literature review](realtime-research.md) includes TimeInteract, On the Clock and AgentTime; no global novelty claim is made.

## Practical boundaries

File-source age is measured since controller ingestion, not producer write time or physical measurement time. Records buffered during model warmup have unknown producer age; `after_s` deadlines begin when ingested. Rotation/truncation can discard replacement-file records before detection, and this is logged. The watcher is not a lossless event transport.

The supplied LLM is small and modified; success on these fixed prompts does not imply robust arbitrary conversational time reasoning. The benchmark does not cover machine suspend/resume, production restarts, external timestamp reconciliation or distributed clocks. There are no hard real-time guarantees. TimesFM 3's non-commercial weight terms remain applicable.

The earlier static time-series QA was the wrong target for the user's real-time objective. Its [failure analysis](failure-analysis-v1.md) and [raw evidence](../evidence/static-v1/) remain available for inspection; they are not used to claim live temporal awareness.
