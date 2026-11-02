# Live temporal state: corrected objective

The user explicitly clarified that the target is both passing time and changing events **in real time**. The earlier static time-series QA experiment was a scope mistake. Its implementation and negative evidence remain available, but they do not establish the requested behavior.

The live prototype is an external temporal coprocessor for a frozen local language model. It has four responsibilities:

1. A monotonic clock measures actual elapsed time, age and time remaining. UTC names calendar time but never controls durations.
2. An event loop ingests observations and semantic changes even while the model is generating. Deadlines fire independently of inference.
3. Every inference request carries an explicit run, request and state identity plus temporal validity limits. The runtime rejects obsolete completions; narrative is labeled with its snapshot time, and live clock fields are refreshed at publication.
4. Optionally, a frozen TimesFM forecaster consumes a causally resampled, actually collected stream and proposes the next bounded attention time. Its predictions cannot override deadlines or force observed events to wait.

The time-series model supplies anticipation, not a clock. A clock-only control is mandatory. The mathematical distinction is simple: `elapsed = monotonic_now - start`, while a prediction describes `x(t+h)` from earlier measurements. Neither the foundation model nor language-model weights are trained in this prototype.

## Why this structure

- Injecting only the current date into a prompt supplies no persistent live event handling.
- A tight loop asking the LLM every tick is unnecessarily expensive and still risks stale output during generation.
- Training a temporal adapter could change the model itself, but does not eliminate the need for current observations and a clock. It is a separate research project.
- The selected event loop plus isolated inference process keeps exact timing cheap. The model wakes for semantic events, deadlines and bounded attention intervals. TimesFM may refine an interval only after enough live observations exist.

The candidate research mechanism is a validity-limited temporal context: attach a snapshot's semantic revision, elapsed age and relevant expiry boundaries, then revalidate it at publication. The proposed addition is prediction-guided attention under those exact guards. Related asynchronous and streaming systems already exist; this implementation is not evidence of globally novel architecture or consciousness.

## Live tests, not simulated passage of time

The real-process test waits with the model idle, compares a frozen pre-wait snapshot with a refreshed one, asks for numerical elapsed seconds, changes the state after the first actual generated token, schedules an urgent deadline during the same generation, checks rejection and refresh, pauses a live source and asks about observation age, and expires a request without a new semantic event.

In the TSFM run, no numerical history is preloaded. A simulated sensor produces samples at actual wall-clock arrivals. The controller waits for sufficient recent data, refuses a blackout in the resampling window, forecasts in the worker, and executes a forecast-proposed wake. Values are simulated; waiting, generation, event arrival, deadline firing and scheduling are real. This is a scripted systems trial, not a general reasoning benchmark or physical-sensor validation.

The initial clock-only trial demonstrated runtime concurrency but failed three model answers. That failure remains in `artifacts/live-clock-20261102-a`. Subsequent trials must retain exact grading and report both mechanics and model answers separately. Numerical answers are compared with the captured snapshot value; publication delay is distinct.

## Validity and practical limits

Ordinary measurements advance `data_revision`; semantic events advance `revision`. Requests that depend on an observation bind its data revision and freshness boundary. Other narratives are explicitly as-of-snapshot, not silently described as current. Pending deadline crossings invalidate a ticket even between timer ticks. Wall-clock jumps do not extend deadlines. A restarted engine cannot accept an old ticket.

The file watcher uses arrival time and starts at end-of-file, so historical content is not relabeled as a live event. It detects rotation/truncation and rejects unsupported timestamp fields. External source time, irregular-series modeling and hard real-time operating-system guarantees are out of scope.

A successful result establishes that the **combined system** can maintain and use live temporal state. It does not establish an internal clock in frozen LLM weights, broad temporal cognition, or a benefit from TimesFM over cheap rules. Forecast-based wake quality needs a matched live-trace study before any efficiency claim.
