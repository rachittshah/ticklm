# Retained evidence

These files are verbatim copies of local experiment artifacts. Model weights, downloaded caches and private data are not included.

- `live-clock-20261102-a`, `-b`: failed development trials, deliberately preserved.
- `live-clock-20261102-c`, `-d`: passing clock/event trials; D uses the final core guards and model/source fingerprints.
- `live-fused-20261102-a`, `-b`: passing live TimesFM integration trials; B is the final rerun.
- `live-watch-20261102-a`, `-b`: public-CLI file ingestion tests. B retains stdout and the actual pytest result. A's temporary stdout was already removed by pytest retention; its trace and test output remain.
- `static-v1`: the original, mis-scoped static forecasting/representation experiment, including full prompts, forecasts, selected configuration, cases and held-out outcomes.
- `verification`: final CPU test and lint output.
- `frontier-claude-clock-20261102-a`, `frontier-claude-fused-20261102-a`: actual Claude Code trials through `cca startup`, with observed model `claude-opus-5-5`. All four graded answers were correct; stale observation answers were rejected. The forecast-triggered answer also expired in the fused run. These are failed overall trials, retained with their summaries and error records.
- `frontier-codex-clock-20261102-a`, `frontier-codex-fused-20261102-a`: actual `gpt-6-astra` trials through Codex app-server. All four graded answers were correct; some arrived after expiry or a new observation. Both runs failed overall. The initial clock-only run preceded the final publication-aggregate and metadata changes; its source manifest is authoritative for that attempt.

Read [the frontier comparison](../docs/frontier-results.md) for model identities, client-observed timings, complete input-token accounting, and what the failures mean. `failure.json` in these four run directories records failed scripted checks, not a failed client connection. The `verification` directory's original files describe the local experiment; frontier verification is recorded separately as `frontier-pytest.txt` and `frontier-ruff.txt`.

Each live trace identifies first-token events, actual arrivals, deadline dispatch, raw worker completions, publication or rejection, and snapshot/publication clocks. JSON summaries describe scripted checks, not general benchmark success rates. Manifests identify the source used by that particular run; earlier runs can differ from current source. Read [live-results.md](../docs/live-results.md) for scope and limits.
