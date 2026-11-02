# Running TickLM with Claude Code and Codex

Work branch: `feat/frontier-cli-models`, starting at `cceac7b`.

The goal is to run the existing live clock, event and optional TimesFM experiment through the user's authenticated coding clients. Claude must launch through `cca startup`. Record the actual served model, visible output, timing, usage where available, and failures. This measures the client plus model, including network and startup time, rather than bare model inference.

Three approaches were considered. Direct provider APIs offer control but require a different authentication route. Ordinary CLI result output is simpler but does not establish when text begins arriving. Streaming Claude Code output and Codex app-server events preserve existing authentication and expose visible answer deltas, so those are the chosen interfaces.

Implementation adds separate Claude and Codex adapters and focused protocol tests. The existing worker remains responsible for model execution in a child process; the live controller continues to own clocks, deadlines and answer validity. Both adapters use fresh conversations, isolated working directories and no task tools. Credentials, account details and raw configuration must never enter retained evidence.

Keep the existing timing rules: initial deadline three seconds, urgent deadline 250 milliseconds after the first visible answer text, one-second observation freshness, and the existing answer validity windows. A configurable total trial timeout can accommodate slower cloud clients without extending those validity windows. Record a failure if a cloud answer is already stale when it arrives. First visible text is a client-observed event, not a provider-side token-generation timestamp.

Run a connectivity probe through each adapter, then the same clock-only checks through both. If operational, exercise the TimesFM path with the same frozen local forecast model. Preserve failed trials and summaries; separate answer correctness from whether the controller could publish the answer in time. Do not describe a short scripted result as a general model ranking or a compute-saving result.

Rollback: this branch isolates the work. Return to `research/realtime-temporal-state` for the prior implementation; after merging, revert the frontier integration commits. Existing historical evidence and model weights are not rewritten. CLI integrations do not change global client settings or authentication.

Interface references: [Codex app-server](https://learn.chatgpt.com/docs/app-server) and [Claude Code streaming output](https://code.claude.com/docs/en/headless#stream-responses).
