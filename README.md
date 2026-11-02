# Temporal Bridge — a live sense of time for a local LLM

An event-driven temporal coprocessor for a frozen language model: **passing time, changing observations, freshness, deadlines, and answers that become stale while being generated.** It reuses [Rachitt's MLX TimesFM implementation](https://github.com/rachittshah/mlx-tsfm) for optional prediction-guided attention.

The operating system supplies the clock. The runtime keeps receiving events and fires deadlines while the LLM works in a separate process. The LLM receives current temporal context when it wakes. TimesFM can propose when to wake next; it cannot replace the clock or delay an observed alarm.

```
clock + arriving observations/events
                │
                ▼
 temporal state + deadline supervisor ───► live clock / deadline events
                │
        snapshot with validity limits
                ▼
       isolated frozen language model
                │
       revalidate before publication
          ┌─────┴─────┐
       current       obsolete
       publish       reject + refresh

collected stream → optional frozen TimesFM → bounded next-wake hint
```

**Claim boundary:** this gives the combined system live temporal state. It does not add a clock to frozen model weights, demonstrate consciousness, or establish broad temporal intelligence. The first static forecasting experiment was a scope mistake; it is retained as prior exploration, not evidence of the live goal.

## Run the live prototype

Python 3.12, Apple Silicon, the MLX TimesFM checkout, and a local MLX language-model checkpoint are required for the measured configuration.

```bash
cd /Users/rachitt/labs/temporal-bridge
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]' -e /Users/rachitt/mlx-tsfm

# Actual waiting, mid-generation alarm, deadlines, expiry, and refreshed answers.
.venv/bin/python realtime.py self-test --clock-only --run-dir artifacts/my-clock-trial

# Also collect live samples, forecast them, and execute a predicted wake.
.venv/bin/python realtime.py self-test --run-dir artifacts/my-fused-trial

# Watch the machine's live load, with a real 20-second deadline.
.venv/bin/python realtime.py watch --clock-only --seconds 60 \
  --deadline-seconds 20 --run-dir artifacts/my-watch
```

Use a new run directory every time. The default model is the locally cached Qwen3-4B abliterated 4-bit checkpoint at `/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit`; override with `--model /path/to/model`.

The self-test's sensor values are simulated **at real wall-clock arrivals**. No history is preloaded. Waiting, event arrival, inference, deadline dispatch and wake scheduling are real. The model receives 128 causally resampled points only after enough uninterrupted live history exists. This is a scripted integration trial, not a statistical reasoning benchmark. The final clock-only trial passed 10 checks; the final fused trial passed 12 checks over 24.69 seconds with 201 sample arrivals. See [measured results and retained traces](docs/live-results.md).

## Feed your own live events

Create an empty append-only file, then start the watcher. It starts at end-of-file and uses arrival times; it does not relabel a historical backlog as live data.

```bash
mkdir -p data
touch data/live-events.jsonl
.venv/bin/python realtime.py watch --clock-only --events data/live-events.jsonl \
  --seconds 120 --run-dir artifacts/my-events
```

Append one JSON object per line from another terminal or a sensor producer:

```json
{"type":"sample","value":42.5}
{"type":"event","name":"door","payload":{"open":true}}
{"type":"deadline","id":"follow_up","after_s":10}
```

Omit `--clock-only` to enable TimesFM wake hints once sufficient regularly resampleable history arrives. `--change-threshold` controls which numeric changes create semantic wake events. The default is 1.0 in the stream's units; choose it for your source. The runtime also wakes on a maximum-silence timer. Source timestamps are deliberately rejected: age means time since the controller read a record, not time since a producer wrote or physically measured it. Buffered records' physical age is unknown; relative file deadlines start at ingestion. Rotation/truncation resets to the new end-of-file and can discard replacement-file records. This prototype is not a lossless event transport.

Published output includes the narrative's snapshot time and freshly computed clock fields. Ordinary samples advance a data revision; relevant events advance the semantic revision. Observation-dependent requests can bind the data revision and freshness limit. Every pending deadline crossing is checked at publication, including between supervisor ticks.

## Validation and evidence

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests app.py realtime.py
git diff --check
```

The live trials retain request snapshots, prompts, first-token timestamps, incoming events, deadline dispatches, raw completions, rejections and publication-time clocks. Failed development trials are retained too. Exact categorical grading rejects negation and ambiguous replies; a numeric elapsed-time probe is checked against the captured clock.

- [Live design and limits](docs/realtime-design.md)
- [Measured results and evidence](docs/live-results.md)
- [Current real-time research](docs/realtime-research.md)
- [Independent runtime review](docs/realtime-review.md)
- [Initial live-trial failures and review](docs/live-evidence-review.md)
- [Earlier static study and its failure analysis](docs/failure-analysis-v1.md)

Clock-only behavior is the mandatory low-cost baseline. The prototype demonstrates that TimesFM can be connected to live scheduling; a reduction in inference calls or improvement over cheap scheduling is **not established**. There is no hard real-time operating-system guarantee.

## Models and earlier research

The TSFM source checkout used during development was `mlx-tsfm` commit `c834636e97f98940d95e34601a17de3886bf79d3`. TimesFM weights are pinned to revision `43046b85ec22d584a13f8098c2ed39c889e129c2` of `google/timesfm-3.0-pytorch`. Its [non-commercial weight license](https://huggingface.co/google/timesfm-3.0-pytorch) applies separately from the implementation. Weights and local caches are not committed.

The earlier `temporal_bridge.experiment` CLI remains reproducible from its recorded source revision and manifests. Its [literature review](docs/literature.md) covers static time-series fusion, and [design](docs/design.md) records that earlier study. Those results cannot answer the corrected live-time question.
