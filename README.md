# Temporal Bridge

**Helping an AI assistant keep track of passing time and changes around it.**

The idea is simple: an assistant should notice that a deadline has passed, that its information is getting old, or that something changed while it was writing an answer. It should be able to do this without waiting for someone to send another message.

This prototype adds a small program around a language model. That program watches the clock and incoming information, tells the model when it needs to respond, and checks whether its answer is still usable before showing it.

It runs locally. Neither the language model nor the forecasting model was retrained.

## What this looks like

Imagine an assistant monitoring a door. It starts writing “the door is closed.” Before it finishes, the door opens.

The program records the change while the assistant is still writing. It catches the outdated answer, keeps it out of the user-facing output, and asks the assistant to respond using the new information. Any timers keep running throughout.

That is the kind of behavior this project is trying to make work.

There are three parts:

| Part | Its job |
|---|---|
| The clock and event watcher | Keep track of elapsed time, deadlines, new information, and how old that information is. |
| The language model | Read the current situation and explain it or answer a question. |
| TimesFM, an optional time-series forecasting model | Look at changing measurements and suggest when the assistant should check again. |

**The clock tells us how much time has passed. TimesFM predicts what might happen next.** The forecasting model is not needed to count seconds or enforce a deadline.

## What worked in the live test

We ran a test lasting about **25 seconds**. During that time, the program received **201 measurements** and called the language model **7 times**.

- After a real wait, the model correctly reported the elapsed time and an expired deadline.
- An alarm arrived while the model was writing. The program handled the alarm and a deadline without waiting for the answer to finish.
- The outdated answer was rejected. The next answer reflected the alarm.
- When new measurements stopped arriving, the model correctly identified the last measurement as too old.
- TimesFM made a forecast from measurements collected during the run. The program used that forecast to schedule another check.

The clock-only version passed all 10 checks in its final scripted test. The version with TimesFM passed all 12. A separate test also passed with events written to a real file while the model was generating an answer.

The measurements in the main demonstration were simulated, but they arrived as real time passed. We did not feed in a complete recorded history and pretend it was live. The waits, model responses, alarms, deadlines, and answer rejection actually happened.

Read the [plain-language results report](docs/live-results.md) for the measurements, failures, and limits.

## What we have not proved

We have built a working way to give the **combined system** current information about time. We have not taught the language model to keep its own internal clock.

We have also not shown that TimesFM chooses better times to check than a simple rule or cheap predictor. The clock-only version already handles passing time, deadlines, and incoming changes. TimesFM's added value remains a research question.

These were short, scripted tests. They do not establish reliability across arbitrary conversations, long-running use, or production failures. Earlier attempts gave wrong answers; those failures are retained in the repository.

## Try it

On the existing local setup, run:

```bash
cd /Users/rachitt/labs/temporal-bridge
.venv/bin/python realtime.py self-test --run-dir artifacts/my-live-demo
```

This runs the demonstration with TimesFM. Add `--clock-only` to test the simpler version. Use a new output directory for each run.

<details>
<summary>Installation and model details</summary>

The tested setup uses Python 3.12 on an Apple Silicon Mac, a local language model, and [Rachitt's TimesFM implementation for MLX](https://github.com/rachittshah/mlx-tsfm). MLX is the software used here to run the models on the Mac.

```bash
cd /Users/rachitt/labs/temporal-bridge
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]' -e /Users/rachitt/mlx-tsfm
```

The default language model is Qwen3-4B abliterated, stored in a compressed 4-bit format at `/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit`. To use another compatible local model, add `--model /path/to/model`.

The TimesFM implementation used during development was commit `c834636e97f98940d95e34601a17de3886bf79d3`. Its model files are pinned to revision `43046b85ec22d584a13f8098c2ed39c889e129c2` of `google/timesfm-3.0-pytorch`. The model's [non-commercial license](https://huggingface.co/google/timesfm-3.0-pytorch) applies. Model files and local caches are not stored in this repository.

</details>

<details>
<summary>Watch your own incoming information</summary>

Create an empty file and start the watcher:

```bash
mkdir -p data
touch data/live-events.jsonl
.venv/bin/python realtime.py watch --clock-only --events data/live-events.jsonl \
  --seconds 120 --run-dir artifacts/my-events
```

From another terminal or a sensor program, append one JSON object per line:

```json
{"type":"sample","value":42.5}
{"type":"event","name":"door","payload":{"open":true}}
{"type":"deadline","id":"follow_up","after_s":10}
```

The watcher ignores anything already in the file when it starts. Omit `--clock-only` to enable forecast-based checks after enough suitably spaced measurements arrive. `--change-threshold` sets how large a numerical change must be to trigger attention; its default is 1.0 in your measurement's units.

To watch the Mac's system load instead, leave out `--events`:

```bash
.venv/bin/python realtime.py watch --clock-only --seconds 60 \
  --deadline-seconds 20 --run-dir artifacts/my-watch
```

The displayed answer is labeled with the time of the information it used. A separate clock shows the time when the answer is presented. Rejected answers stay in the diagnostic log and are not displayed as answers.

There are limits to this file-based input. The program knows when it reads a record, not when someone originally wrote or measured it. A delayed record can therefore have an unknown real-world age. Relative deadlines start when the program reads them. Replacing or shortening the input file can also cause records to be missed. This is a prototype, not a guaranteed delivery system.

</details>

<details>
<summary>Tests, research, and full evidence</summary>

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests app.py realtime.py
git diff --check
```

The recorded verification is **91 automated tests passed**. The optional test that runs the actual model and file watcher passed separately. Code checks also passed.

- [Results in plain language](docs/live-results.md)
- [Live-system design](docs/realtime-design.md)
- [Research papers on real-time agents](docs/realtime-research.md)
- [Independent review](docs/realtime-final-review.md)
- [Raw records, including failed attempts](evidence/README.md)

The earlier static forecasting experiment addressed the wrong version of the question. Its [failure analysis](docs/failure-analysis-v1.md) and [recorded results](evidence/static-v1/) are preserved for inspection. They are not evidence of a live sense of time.

</details>
