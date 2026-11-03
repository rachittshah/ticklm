# Native-client microtier expansion

Registered before the expanded models' microtier runs, 3 November 2026. This
supplement evaluates six client/model combinations selected by advertised
availability and a preference for the latest distinct model families, not by
preflight answer quality. Sol and Terra had already received one development
preflight each before this expansion was registered; their preflight answers
were visible and are not blinded. Older catalog entries are excluded from this
bounded expansion to prioritize the latest advertised family variants.

The selected client models are `gpt-6.1-sol`, `gpt-6.1-terra`, and `gpt-6-luna`
through the existing Codex sign-in; and `claude-opus-5-5`, `claude-sonnet-5-5`,
and `claude-haiku-5-5` through `cca startup`. All request `xhigh` effort and 256
output tokens. The client cannot enforce the output-token cap; clients may add
instructions and perform internal retries. Claude client-reported costs are
estimates, not evidence of direct API billing. Codex cost is unavailable. These
are client/model results and must not be merged into the capped model-only
ranking.

Each advertised model receives one development preflight with its exact ID and
effort. A transport-successful response with matching resolved identity admits
it to the microtier regardless of answer correctness. A preflight failure,
unsupported effort, identity mismatch, or inaccessible account excludes that
combination and is retained; there is no silent model, effort, or profile
fallback. The existing verified Sol and Terra preflights satisfy this access
check. No paid direct API is used.

All admitted systems run the same frozen 28-request microtier, containing one
counterfactual pair for each of seven families under raw and derived views.
Its SHA-256 is
`122b912e70dc6c51e39921baeba2d0ae91967cbccd2e2e87ec7f4f8b363b8c83`.
Execution uses immutable source commit
`8421a60346096e11debce5aa4b47baa4bac74c3b`; no test cases are selected or
altered based on model outputs. Retain raw requests, errors, partial results,
and early-stop status. Run no more than two expansion processes concurrently.

Microtier results are exploratory, have only one pair per family and view, and
are reported separately from the 112-request screen. They do not establish a
stable leaderboard, broad coverage, or model ranking. Accessibility, rate limits,
and client behavior may change over time.
