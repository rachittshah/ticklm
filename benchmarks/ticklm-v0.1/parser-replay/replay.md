# Post-hoc live parser replay

Post-hoc after Qwen format failures: offline parser counterfactual at original recorded publication times, assuming zero extra parser time. No new inference or live intervention.

| System | Parser | Guard | Correct s | Wrong s | Undecided s | Invalid | Failed | Snapshot correct | Snapshot wrong | Valid |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-06b-bf16-mlx | strict | oracle-assisted reference | 0.000 | 0.000 | 90.000 | 343 | 0 | 0 | 0 | 0 |
| qwen3-06b-bf16-mlx | strict | freshness_only | 0.000 | 0.000 | 90.000 | 343 | 0 | 0 | 0 | 0 |
| qwen3-06b-bf16-mlx | strict | unguarded | 0.000 | 0.000 | 90.000 | 343 | 0 | 0 | 0 | 0 |
| qwen3-06b-bf16-mlx | alternative | oracle-assisted reference | 56.048 | 0.000 | 33.952 | 4 | 0 | 226 | 113 | 339 |
| qwen3-06b-bf16-mlx | alternative | freshness_only | 56.048 | 28.768 | 5.184 | 4 | 0 | 226 | 113 | 339 |
| qwen3-06b-bf16-mlx | alternative | unguarded | 58.500 | 29.269 | 2.231 | 4 | 0 | 226 | 113 | 339 |
| gpt6-astra-codex-xhigh | strict | oracle-assisted reference | 13.594 | 0.000 | 76.406 | 0 | 1 | 18 | 0 | 18 |
| gpt6-astra-codex-xhigh | strict | freshness_only | 13.594 | 0.000 | 76.406 | 0 | 1 | 18 | 0 | 18 |
| gpt6-astra-codex-xhigh | strict | unguarded | 35.940 | 40.032 | 14.028 | 0 | 1 | 18 | 0 | 18 |
| gpt6-astra-codex-xhigh | alternative | oracle-assisted reference | 13.594 | 0.000 | 76.406 | 0 | 1 | 18 | 0 | 18 |
| gpt6-astra-codex-xhigh | alternative | freshness_only | 13.594 | 0.000 | 76.406 | 0 | 1 | 18 | 0 | 18 |
| gpt6-astra-codex-xhigh | alternative | unguarded | 35.940 | 40.032 | 14.028 | 0 | 1 | 18 | 0 | 18 |
| claude-fable-51-cca-startup-xhigh | strict | oracle-assisted reference | 0.000 | 0.000 | 90.000 | 4 | 0 | 2 | 0 | 2 |
| claude-fable-51-cca-startup-xhigh | strict | freshness_only | 0.000 | 0.000 | 90.000 | 4 | 0 | 2 | 0 | 2 |
| claude-fable-51-cca-startup-xhigh | strict | unguarded | 0.164 | 10.500 | 79.336 | 4 | 0 | 2 | 0 | 2 |
| claude-fable-51-cca-startup-xhigh | alternative | oracle-assisted reference | 0.000 | 0.000 | 90.000 | 4 | 0 | 2 | 0 | 2 |
| claude-fable-51-cca-startup-xhigh | alternative | freshness_only | 0.000 | 0.000 | 90.000 | 4 | 0 | 2 | 0 | 2 |
| claude-fable-51-cca-startup-xhigh | alternative | unguarded | 0.164 | 10.500 | 79.336 | 4 | 0 | 2 | 0 | 2 |
| smollm3-3b-4bit-mlx | strict | oracle-assisted reference | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |
| smollm3-3b-4bit-mlx | strict | freshness_only | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |
| smollm3-3b-4bit-mlx | strict | unguarded | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |
| smollm3-3b-4bit-mlx | alternative | oracle-assisted reference | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |
| smollm3-3b-4bit-mlx | alternative | freshness_only | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |
| smollm3-3b-4bit-mlx | alternative | unguarded | 0.000 | 0.000 | 90.000 | 38 | 0 | 0 | 0 | 0 |

The alternative accepts only the frozen parser or an exact matching option line. Timing, dispatches, completions, state changes and publication timestamps remain recorded values.

Oracle-assisted decisions use ground truth and are a reference, not a deployable guard.
