# TickLM-Bench v0.1 — measured results

Generated 2026-11-03T10:15:42.336456+00:00 directly from retained request records.

This is an exploratory synthetic benchmark. It measures supplied temporal facts and rules, not an internal clock or a general state-of-the-art ranking. Raw and arithmetic-assisted inputs, sample sizes, and native clients appear separately.

Quality comparisons require the same case-file SHA, per-view scorer case-set SHA, current strict scoring code, and complete, structurally valid coverage. Track and harness are shown but do not by themselves prevent a shared-data quality comparison. Policy-aligned groups add track, request count, adapter type, and requested generation settings; model weights are not part of that identity.

Macro accuracy is shown beside its family-stratified bootstrap interval; micro accuracy is separate. Degenerate intervals such as [0%, 0%] for all-invalid outputs or [100%, 100%] for all-correct outputs do not establish zero reasoning ability or broad generalization. No option-order variants were run, so option-order invariance is not measured.

## Shared frozen-data quality group 9389b2b1ef8fbc19 — raw

Dataset SHA `869a458b245e4d8c1aae519eccf802600140be2f694db42deae447ae724648d5`; case-set SHA `b386f78418d44d6e3c64c1fd776739f812dd0d6aa0ff3457c9005261ce979cd5`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 11 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| claude-fable-51-cca-startup-xhigh | native-client / claude_code | claude-fable-5-1 | 56 / 56 | 0 | 0 | 56/56 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 28/28 | `9389b2b1ef8fbc19` | `8ddad770c0ac9570` | yes | no |
| gpt6-astra-codex-xhigh | native-client / codex | gpt-6-astra | 56 / 49 | 0 | 7 | 49/49 | 100.0% | 100.0% | 0 | 21/21 | `9389b2b1ef8fbc19` | `7425da9d86e6d642` | no | no |
| gemma-3-1b-it-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/gemma-3-1b-it-4bit-2d44e83d) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| lfm2.5-1.2b-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/lfm2.5-1.2b-instruct-4bit-dee2f8a2) | 56 / 56 | 0 | 0 | 20/56 | 35.7% [28.6%, 42.9%] | 35.7% | 0 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| llama-3.2-3b-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/llama-3.2-3b-instruct-4bit-7f0dc925) | 56 / 56 | 0 | 0 | 24/56 | 42.9% [37.5%, 48.2%] | 42.9% | 2 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| qwen3-06b-q4km-llamacpp | local-http / http | qwen3-0.6b-q4 | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `56a88c8956d45099` | yes | no |
| qwen3-06b-q4km-ollama | local-http / http | qwen3:0.6b | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `03f16012aa55a831` | yes | no |
| phi-4-mini-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/phi-4-mini-instruct-4bit-ac1c269c) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| qwen3-06b-bf16-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| qwen3-4b-abliterated-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| qwen3.5-4b-optiq-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/qwen3.5-4b-optiq-4bit-4d4b423e) | 56 / 56 | 0 | 0 | 32/56 | 57.1% [53.6%, 60.7%] | 57.1% | 0 | 4/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |
| smollm3-3b-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `9389b2b1ef8fbc19` | `7ce9b1a81737a096` | yes | yes |

## Shared frozen-data quality group b2ae655ffa2f0edb — raw

Dataset SHA `814f7a4c586324f3a9bf5e4244a8a361327920f693555f200bba82b96abb1c80`; case-set SHA `029c0cf9917f85a607baa835103e496928a0703184fc56271f7ccc4781eba2d7`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 2 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| qwen3-06b-bf16-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 224 / 224 | 0 | 0 | 0/224 | 0.0% [0.0%, 0.0%] | 0.0% | 224 | 0/112 | `b2ae655ffa2f0edb` | `3f91fb032217c20a` | yes | yes |
| smollm3-3b-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 224 / 224 | 0 | 0 | 0/224 | 0.0% [0.0%, 0.0%] | 0.0% | 224 | 0/112 | `b2ae655ffa2f0edb` | `3f91fb032217c20a` | yes | yes |

## Shared frozen-data quality group c4973d2205b43f88 — derived

Dataset SHA `122b912e70dc6c51e39921baeba2d0ae91967cbccd2e2e87ec7f4f8b363b8c83`; case-set SHA `41f47991c0f3bf7d3e45f9b9b9e4c01f6eafec5099967373c7e40576d2af9b37`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 6 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| claude-haiku-55-cca-startup-xhigh | native-client / claude_code | claude-haiku-5-5 | 14 / 14 | 0 | 0 | 14/14 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 7/7 | `c4973d2205b43f88` | `5c62da13310f12dd` | yes | yes |
| claude-opus-55-cca-startup-xhigh | native-client / claude_code | claude-opus-5-5 | 14 / 14 | 0 | 0 | 14/14 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 7/7 | `c4973d2205b43f88` | `5c62da13310f12dd` | yes | yes |
| claude-sonnet-55-cca-startup-xhigh | native-client / claude_code | claude-sonnet-5-5 | 14 / 14 | 0 | 0 | 13/14 | 92.9% [92.9%, 92.9%] | 92.9% | 1 | 6/7 | `c4973d2205b43f88` | `5c62da13310f12dd` | yes | yes |
| gpt-6-luna-codex-xhigh | native-client / codex | gpt-6-luna | 14 / 14 | 0 | 0 | 13/14 | 92.9% [92.9%, 92.9%] | 92.9% | 0 | 6/7 | `c4973d2205b43f88` | `5d9e72b4583761ae` | yes | yes |
| gpt-6.1-sol-codex-xhigh | native-client / codex | gpt-6.1-sol | 14 / 13 | 0 | 1 | 13/13 | 100.0% | 100.0% | 0 | 6/6 | `c4973d2205b43f88` | `5d9e72b4583761ae` | no | no |
| gpt-6.1-terra-codex-xhigh | native-client / codex | gpt-6.1-terra | 14 / 14 | 0 | 0 | 14/14 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 7/7 | `c4973d2205b43f88` | `5d9e72b4583761ae` | yes | yes |
| qwen3-06b-q4km-opencode-ollama | native-local-client / opencode | UNKNOWN (adapter returned no resolved model) | 14 / 14 | 0 | 0 | 0/14 | 0.0% [0.0%, 0.0%] | 0.0% | 14 | 0/7 | `c4973d2205b43f88` | `61bb328b8b6b339c` | yes | no |

## Shared frozen-data quality group d6f85512a6f0cc8a — raw

Dataset SHA `122b912e70dc6c51e39921baeba2d0ae91967cbccd2e2e87ec7f4f8b363b8c83`; case-set SHA `f05dd43cad19fa9fef880a91230f4ed58387c431a4f322ecf2c032a44f190906`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 4 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| claude-haiku-55-cca-startup-xhigh | native-client / claude_code | claude-haiku-5-5 | 14 / 14 | 0 | 0 | 14/14 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 7/7 | `d6f85512a6f0cc8a` | `cfd8576549ae741c` | yes | yes |
| claude-opus-55-cca-startup-xhigh | native-client / claude_code | claude-opus-5-5 | 14 / 14 | 0 | 0 | 13/14 | 92.9% [92.9%, 92.9%] | 92.9% | 1 | 6/7 | `d6f85512a6f0cc8a` | `cfd8576549ae741c` | yes | yes |
| claude-sonnet-55-cca-startup-xhigh | native-client / claude_code | claude-sonnet-5-5 | 14 / 14 | 0 | 0 | 11/14 | 78.6% [78.6%, 78.6%] | 78.6% | 3 | 5/7 | `d6f85512a6f0cc8a` | `cfd8576549ae741c` | yes | yes |
| gpt-6-luna-codex-xhigh | native-client / codex | gpt-6-luna | 14 / 12 | 0 | 2 | 11/12 | 92.9% | 91.7% | 0 | 4/5 | `d6f85512a6f0cc8a` | `16f976280016d0b8` | no | no |
| gpt-6.1-sol-codex-xhigh | native-client / codex | gpt-6.1-sol | 14 / 14 | 0 | 0 | 14/14 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 7/7 | `d6f85512a6f0cc8a` | `16f976280016d0b8` | yes | no |
| gpt-6.1-terra-codex-xhigh | native-client / codex | gpt-6.1-terra | 14 / 12 | 0 | 2 | 12/12 | 100.0% | 100.0% | 0 | 5/5 | `d6f85512a6f0cc8a` | `16f976280016d0b8` | no | no |
| qwen3-06b-q4km-opencode-ollama | native-local-client / opencode | UNKNOWN (adapter returned no resolved model) | 14 / 13 | 0 | 1 | 0/13 | 0.0% | 0.0% | 13 | 0/6 | `d6f85512a6f0cc8a` | `bf505bf0f47d6b53` | no | no |

## Shared frozen-data quality group de366477e2621a40 — derived

Dataset SHA `869a458b245e4d8c1aae519eccf802600140be2f694db42deae447ae724648d5`; case-set SHA `654b83331214c04529ac3fa94b11b06ff6b1e2239b5d044f6e68e4d784bea16a`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 11 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| claude-fable-51-cca-startup-xhigh | native-client / claude_code | claude-fable-5-1 | 56 / 56 | 0 | 0 | 56/56 | 100.0% [100.0%, 100.0%] | 100.0% | 0 | 28/28 | `de366477e2621a40` | `3da3c53ad6ba7cd3` | yes | no |
| gpt6-astra-codex-xhigh | native-client / codex | gpt-6-astra | 56 / 54 | 0 | 2 | 54/54 | 100.0% | 100.0% | 0 | 26/26 | `de366477e2621a40` | `70eb78f601e44929` | no | no |
| gemma-3-1b-it-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/gemma-3-1b-it-4bit-2d44e83d) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| lfm2.5-1.2b-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/lfm2.5-1.2b-instruct-4bit-dee2f8a2) | 56 / 56 | 0 | 0 | 20/56 | 35.7% [26.8%, 44.6%] | 35.7% | 0 | 1/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| llama-3.2-3b-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/llama-3.2-3b-instruct-4bit-7f0dc925) | 56 / 56 | 0 | 0 | 26/56 | 46.4% [42.9%, 50.0%] | 46.4% | 0 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| qwen3-06b-q4km-llamacpp | local-http / http | qwen3-0.6b-q4 | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `1768e670abc255a5` | yes | no |
| qwen3-06b-q4km-ollama | local-http / http | qwen3:0.6b | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `f6ea6459e18fc37e` | yes | no |
| phi-4-mini-instruct-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/phi-4-mini-instruct-4bit-ac1c269c) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| qwen3-06b-bf16-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| qwen3-4b-abliterated-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| qwen3.5-4b-optiq-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/qwen3.5-4b-optiq-4bit-4d4b423e) | 56 / 56 | 0 | 0 | 33/56 | 58.9% [53.6%, 64.3%] | 58.9% | 0 | 5/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |
| smollm3-3b-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 56 / 56 | 0 | 0 | 0/56 | 0.0% [0.0%, 0.0%] | 0.0% | 56 | 0/28 | `de366477e2621a40` | `70d83a8ee506416e` | yes | yes |

## Shared frozen-data quality group e8e751f1949f89f4 — derived

Dataset SHA `814f7a4c586324f3a9bf5e4244a8a361327920f693555f200bba82b96abb1c80`; case-set SHA `fc21a3c4175c1b2ffd74f40d0f1765696af51daefc0e157a785ac606517049aa`; scorer SHA `5fc5bafff536a9f3fe15f85bba038e477b8d60b762f6b418fa0b97da9a315c55`. 2 run(s) are complete members of this exact quality group.

| System | Track / adapter | Model identity | Expected / completed | Missing | Failed | Correct / completed | Macro accuracy [95% CI] | Micro accuracy | Invalid | Both-correct pairs | Quality group | Policy group | Quality comparable | Policy aligned |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| qwen3-06b-bf16-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 224 / 224 | 0 | 0 | 0/224 | 0.0% [0.0%, 0.0%] | 0.0% | 224 | 0/112 | `e8e751f1949f89f4` | `f3905d94f4f79f00` | yes | yes |
| smollm3-3b-4bit-mlx | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 224 / 224 | 0 | 0 | 0/224 | 0.0% [0.0%, 0.0%] | 0.0% | 224 | 0/112 | `e8e751f1949f89f4` | `f3905d94f4f79f00` | yes | yes |

## Post-hoc exact-option diagnostic

Added after the first Qwen 0.6B run exposed complete-option answers; it was not preregistered. Applied uniformly to every system, this diagnostic accepts the frozen strict parser or a response consisting entirely of one exact option line rendered in that case prompt. It never extracts a letter from prose and does not replace the primary strict score.

| Run | View | Correct / completed | Invalid |
|---|---|---:|---:|
| claude-fable-51-screen | derived | 56/56 | 0 |
| claude-fable-51-screen | raw | 56/56 | 0 |
| claude-haiku55-micro | derived | 14/14 | 0 |
| claude-haiku55-micro | raw | 14/14 | 0 |
| claude-opus55-micro | derived | 14/14 | 0 |
| claude-opus55-micro | raw | 13/14 | 1 |
| claude-sonnet55-micro | derived | 13/14 | 1 |
| claude-sonnet55-micro | raw | 11/14 | 3 |
| codex-gpt6-astra-screen | derived | 54/54 | 0 |
| codex-gpt6-astra-screen | raw | 49/49 | 0 |
| codex-luna-micro | derived | 13/14 | 0 |
| codex-luna-micro | raw | 11/12 | 0 |
| codex-sol-micro | derived | 13/13 | 0 |
| codex-sol-micro | raw | 14/14 | 0 |
| codex-terra-micro | derived | 14/14 | 0 |
| codex-terra-micro | raw | 12/12 | 0 |
| gemma3-screen | derived | 0/56 | 56 |
| gemma3-screen | raw | 0/56 | 56 |
| lfm25-screen | derived | 20/56 | 0 |
| lfm25-screen | raw | 20/56 | 0 |
| llama32-screen | derived | 26/56 | 0 |
| llama32-screen | raw | 25/56 | 0 |
| llamacpp-qwen3-06b-screen | derived | 9/56 | 28 |
| llamacpp-qwen3-06b-screen | raw | 10/56 | 29 |
| ollama-qwen3-06b-screen | derived | 9/56 | 28 |
| ollama-qwen3-06b-screen | raw | 10/56 | 29 |
| opencode-ollama-micro | derived | 2/14 | 8 |
| opencode-ollama-micro | raw | 3/13 | 7 |
| phi4-screen | derived | 24/56 | 1 |
| phi4-screen | raw | 23/56 | 4 |
| qwen3-06b-full | derived | 74/224 | 28 |
| qwen3-06b-full | raw | 73/224 | 38 |
| qwen3-06b-screen | derived | 17/56 | 5 |
| qwen3-06b-screen | raw | 18/56 | 7 |
| qwen3-4b-screen | derived | 15/56 | 29 |
| qwen3-4b-screen | raw | 6/56 | 40 |
| qwen35-screen | derived | 33/56 | 0 |
| qwen35-screen | raw | 32/56 | 0 |
| smollm3-full | derived | 0/224 | 224 |
| smollm3-full | raw | 4/224 | 220 |
| smollm3-screen | derived | 0/56 | 56 |
| smollm3-screen | raw | 2/56 | 54 |

## Requested policy, model identity, and runtime

Resolved model identity is taken from returned provider metadata. Native runs without a resolved identity are marked UNKNOWN; a requested name alone is not proof of the served model. “Requested max tokens” is not claimed as an enforced cap unless the recorded adapter capability says so.

| Run | Track / adapter | Requested model | Resolved model identity | Requested max tokens / hard cap | Effort requested | Thinking requested / supported | Seed requested / supported | Streaming / supported | p50 / p95 seconds | First request seconds | Input / output tokens |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| claude-fable-51-screen | native-client / claude_code | claude-fable-5-1 | claude-fable-5-1 | 256 / no | xhigh | — / unknown | None / no | yes | 14.355 / 18.974 | 15.354 | 1309389/19060 |
| claude-haiku55-micro | native-client / claude_code | claude-haiku-5-5 | claude-haiku-5-5 | 256 / no | xhigh | — / unknown | None / no | yes | 3.522 / 4.185 | 3.982 | 126970/4586 |
| claude-opus55-micro | native-client / claude_code | claude-opus-5-5 | claude-opus-5-5 | 256 / no | xhigh | — / unknown | None / no | yes | 7.245 / 9.117 | 5.698 | 105911/1075 |
| claude-sonnet55-micro | native-client / claude_code | claude-sonnet-5-5 | claude-sonnet-5-5 | 256 / no | xhigh | — / unknown | None / no | yes | 7.283 / 9.047 | 9.226 | 103887/980 |
| codex-gpt6-astra-screen | native-client / codex | gpt-6-astra | gpt-6-astra | 256 / no | xhigh | — / unknown | None / no | yes | 5.666 / 11.209 | 11.027 | 1696059 (+9 unknown)/1602 (+9 unknown) |
| codex-luna-micro | native-client / codex | gpt-6-luna | gpt-6-luna | 256 / no | xhigh | — / unknown | None / no | yes | 4.015 / 12.170 | 5.317 | 446609 (+2 unknown)/347 (+2 unknown) |
| codex-sol-micro | native-client / codex | gpt-6.1-sol | gpt-6.1-sol | 256 / no | xhigh | — / unknown | None / no | yes | 7.102 / 74.953 | 8.731 | 460767 (+1 unknown)/726 (+1 unknown) |
| codex-terra-micro | native-client / codex | gpt-6.1-terra | gpt-6.1-terra | 256 / no | xhigh | — / unknown | None / no | yes | 5.186 / 9.264 | 8.666 | 481000 (+2 unknown)/286 (+2 unknown) |
| gemma3-screen | direct-generation / mlx | mlx-community/gemma-3-1b-it-4bit | LOCAL CHECKPOINT (models/benchmark-v1/gemma-3-1b-it-4bit-2d44e83d) | 256 / yes | — | False / unknown | None / no | yes | 0.927 / 1.027 | 2.466 | 17818/27954 |
| lfm25-screen | direct-generation / mlx | mlx-community/LFM2.5-1.2B-Instruct-4bit | LOCAL CHECKPOINT (models/benchmark-v1/lfm2.5-1.2b-instruct-4bit-dee2f8a2) | 256 / yes | — | False / unknown | None / no | yes | 0.052 / 0.060 | 0.224 | 16836/224 |
| llama32-screen | direct-generation / mlx | mlx-community/Llama-3.2-3B-Instruct-4bit | LOCAL CHECKPOINT (models/benchmark-v1/llama-3.2-3b-instruct-4bit-7f0dc925) | 256 / yes | — | False / unknown | None / no | yes | 0.183 / 0.214 | 0.763 | 19440/314 |
| llamacpp-qwen3-06b-screen | local-http / http | qwen3-0.6b-q4 | qwen3-0.6b-q4 | 256 / yes | — | False / unknown | None / unknown | yes | 0.040 / 0.351 | 0.043 | 0 (+112 unknown)/0 (+112 unknown) |
| ollama-qwen3-06b-screen | local-http / http | qwen3:0.6b | qwen3:0.6b | 256 / yes | — | False / unknown | None / unknown | yes | 0.157 / 1.139 | 0.045 | 0 (+112 unknown)/0 (+112 unknown) |
| opencode-ollama-micro | native-local-client / opencode | ollama/qwen3:0.6b | UNKNOWN (adapter returned no resolved model) | 256 / no | — | — / unknown | None / no | no | 29.278 / 90.208 | 34.462 | 15354 (+26 unknown)/1490 (+26 unknown) |
| phi4-screen | direct-generation / mlx | mlx-community/Phi-4-mini-instruct-4bit | LOCAL CHECKPOINT (models/benchmark-v1/phi-4-mini-instruct-4bit-ac1c269c) | 256 / yes | — | False / unknown | None / no | yes | 0.325 / 0.780 | 0.826 | 15688/1078 |
| qwen3-06b-full | direct-generation / mlx | Qwen/Qwen3-0.6B | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 256 / yes | — | False / unknown | None / no | yes | 0.078 / 0.313 | 0.538 | 71606/4766 |
| qwen3-06b-screen | direct-generation / mlx | Qwen/Qwen3-0.6B | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 256 / yes | — | False / unknown | None / no | yes | 0.059 / 0.082 | 1.617 | 17930/1081 |
| qwen3-4b-screen | direct-generation / mlx | huihui-ai/Qwen3-4B-abliterated | LOCAL CHECKPOINT (/Users/rachitt/labs/qwen3-4b-abliterated-mlx/models/qwen3-4b-abliterated-4bit) | 256 / yes | — | False / unknown | None / no | yes | 1.728 / 4.075 | 2.264 | 17930/11162 |
| qwen35-screen | direct-generation / mlx | mlx-community/Qwen3.5-4B-OptiQ-4bit | LOCAL CHECKPOINT (models/benchmark-v1/qwen3.5-4b-optiq-4bit-4d4b423e) | 256 / yes | — | False / unknown | None / no | yes | 0.546 / 1.600 | 1.363 | 18322/224 |
| smollm3-full | direct-generation / mlx | mlx-community/SmolLM3-3B-4bit | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 256 / yes | — | False / unknown | None / no | yes | 1.354 / 3.523 | 1.246 | 92544/56484 |
| smollm3-screen | direct-generation / mlx | mlx-community/SmolLM3-3B-4bit | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 256 / yes | — | False / unknown | None / no | yes | 0.812 / 1.798 | 2.341 | 23136/13405 |

## Results by family

| Run | View | Family | Correct / completed | Invalid |
|---|---|---|---:|---:|
| claude-fable-51-screen | derived | calendar | 8/8 | 0 |
| claude-fable-51-screen | derived | chronology | 8/8 | 0 |
| claude-fable-51-screen | derived | duration | 8/8 | 0 |
| claude-fable-51-screen | derived | event_window | 8/8 | 0 |
| claude-fable-51-screen | derived | interval | 8/8 | 0 |
| claude-fable-51-screen | derived | policy | 8/8 | 0 |
| claude-fable-51-screen | derived | revision | 8/8 | 0 |
| claude-fable-51-screen | raw | calendar | 8/8 | 0 |
| claude-fable-51-screen | raw | chronology | 8/8 | 0 |
| claude-fable-51-screen | raw | duration | 8/8 | 0 |
| claude-fable-51-screen | raw | event_window | 8/8 | 0 |
| claude-fable-51-screen | raw | interval | 8/8 | 0 |
| claude-fable-51-screen | raw | policy | 8/8 | 0 |
| claude-fable-51-screen | raw | revision | 8/8 | 0 |
| claude-haiku55-micro | derived | calendar | 2/2 | 0 |
| claude-haiku55-micro | derived | chronology | 2/2 | 0 |
| claude-haiku55-micro | derived | duration | 2/2 | 0 |
| claude-haiku55-micro | derived | event_window | 2/2 | 0 |
| claude-haiku55-micro | derived | interval | 2/2 | 0 |
| claude-haiku55-micro | derived | policy | 2/2 | 0 |
| claude-haiku55-micro | derived | revision | 2/2 | 0 |
| claude-haiku55-micro | raw | calendar | 2/2 | 0 |
| claude-haiku55-micro | raw | chronology | 2/2 | 0 |
| claude-haiku55-micro | raw | duration | 2/2 | 0 |
| claude-haiku55-micro | raw | event_window | 2/2 | 0 |
| claude-haiku55-micro | raw | interval | 2/2 | 0 |
| claude-haiku55-micro | raw | policy | 2/2 | 0 |
| claude-haiku55-micro | raw | revision | 2/2 | 0 |
| claude-opus55-micro | derived | calendar | 2/2 | 0 |
| claude-opus55-micro | derived | chronology | 2/2 | 0 |
| claude-opus55-micro | derived | duration | 2/2 | 0 |
| claude-opus55-micro | derived | event_window | 2/2 | 0 |
| claude-opus55-micro | derived | interval | 2/2 | 0 |
| claude-opus55-micro | derived | policy | 2/2 | 0 |
| claude-opus55-micro | derived | revision | 2/2 | 0 |
| claude-opus55-micro | raw | calendar | 2/2 | 0 |
| claude-opus55-micro | raw | chronology | 2/2 | 0 |
| claude-opus55-micro | raw | duration | 1/2 | 1 |
| claude-opus55-micro | raw | event_window | 2/2 | 0 |
| claude-opus55-micro | raw | interval | 2/2 | 0 |
| claude-opus55-micro | raw | policy | 2/2 | 0 |
| claude-opus55-micro | raw | revision | 2/2 | 0 |
| claude-sonnet55-micro | derived | calendar | 2/2 | 0 |
| claude-sonnet55-micro | derived | chronology | 1/2 | 1 |
| claude-sonnet55-micro | derived | duration | 2/2 | 0 |
| claude-sonnet55-micro | derived | event_window | 2/2 | 0 |
| claude-sonnet55-micro | derived | interval | 2/2 | 0 |
| claude-sonnet55-micro | derived | policy | 2/2 | 0 |
| claude-sonnet55-micro | derived | revision | 2/2 | 0 |
| claude-sonnet55-micro | raw | calendar | 2/2 | 0 |
| claude-sonnet55-micro | raw | chronology | 2/2 | 0 |
| claude-sonnet55-micro | raw | duration | 2/2 | 0 |
| claude-sonnet55-micro | raw | event_window | 2/2 | 0 |
| claude-sonnet55-micro | raw | interval | 1/2 | 1 |
| claude-sonnet55-micro | raw | policy | 0/2 | 2 |
| claude-sonnet55-micro | raw | revision | 2/2 | 0 |
| codex-gpt6-astra-screen | derived | calendar | 8/8 | 0 |
| codex-gpt6-astra-screen | derived | chronology | 7/7 | 0 |
| codex-gpt6-astra-screen | derived | duration | 8/8 | 0 |
| codex-gpt6-astra-screen | derived | event_window | 8/8 | 0 |
| codex-gpt6-astra-screen | derived | interval | 8/8 | 0 |
| codex-gpt6-astra-screen | derived | policy | 7/7 | 0 |
| codex-gpt6-astra-screen | derived | revision | 8/8 | 0 |
| codex-gpt6-astra-screen | raw | calendar | 6/6 | 0 |
| codex-gpt6-astra-screen | raw | chronology | 8/8 | 0 |
| codex-gpt6-astra-screen | raw | duration | 7/7 | 0 |
| codex-gpt6-astra-screen | raw | event_window | 7/7 | 0 |
| codex-gpt6-astra-screen | raw | interval | 5/5 | 0 |
| codex-gpt6-astra-screen | raw | policy | 8/8 | 0 |
| codex-gpt6-astra-screen | raw | revision | 8/8 | 0 |
| codex-luna-micro | derived | calendar | 1/2 | 0 |
| codex-luna-micro | derived | chronology | 2/2 | 0 |
| codex-luna-micro | derived | duration | 2/2 | 0 |
| codex-luna-micro | derived | event_window | 2/2 | 0 |
| codex-luna-micro | derived | interval | 2/2 | 0 |
| codex-luna-micro | derived | policy | 2/2 | 0 |
| codex-luna-micro | derived | revision | 2/2 | 0 |
| codex-luna-micro | raw | calendar | 1/2 | 0 |
| codex-luna-micro | raw | chronology | 1/1 | 0 |
| codex-luna-micro | raw | duration | 2/2 | 0 |
| codex-luna-micro | raw | event_window | 2/2 | 0 |
| codex-luna-micro | raw | interval | 2/2 | 0 |
| codex-luna-micro | raw | policy | 2/2 | 0 |
| codex-luna-micro | raw | revision | 1/1 | 0 |
| codex-sol-micro | derived | calendar | 2/2 | 0 |
| codex-sol-micro | derived | chronology | 2/2 | 0 |
| codex-sol-micro | derived | duration | 2/2 | 0 |
| codex-sol-micro | derived | event_window | 2/2 | 0 |
| codex-sol-micro | derived | interval | 1/1 | 0 |
| codex-sol-micro | derived | policy | 2/2 | 0 |
| codex-sol-micro | derived | revision | 2/2 | 0 |
| codex-sol-micro | raw | calendar | 2/2 | 0 |
| codex-sol-micro | raw | chronology | 2/2 | 0 |
| codex-sol-micro | raw | duration | 2/2 | 0 |
| codex-sol-micro | raw | event_window | 2/2 | 0 |
| codex-sol-micro | raw | interval | 2/2 | 0 |
| codex-sol-micro | raw | policy | 2/2 | 0 |
| codex-sol-micro | raw | revision | 2/2 | 0 |
| codex-terra-micro | derived | calendar | 2/2 | 0 |
| codex-terra-micro | derived | chronology | 2/2 | 0 |
| codex-terra-micro | derived | duration | 2/2 | 0 |
| codex-terra-micro | derived | event_window | 2/2 | 0 |
| codex-terra-micro | derived | interval | 2/2 | 0 |
| codex-terra-micro | derived | policy | 2/2 | 0 |
| codex-terra-micro | derived | revision | 2/2 | 0 |
| codex-terra-micro | raw | calendar | 2/2 | 0 |
| codex-terra-micro | raw | chronology | 1/1 | 0 |
| codex-terra-micro | raw | duration | 2/2 | 0 |
| codex-terra-micro | raw | event_window | 2/2 | 0 |
| codex-terra-micro | raw | interval | 2/2 | 0 |
| codex-terra-micro | raw | policy | 2/2 | 0 |
| codex-terra-micro | raw | revision | 1/1 | 0 |
| gemma3-screen | derived | calendar | 0/8 | 8 |
| gemma3-screen | derived | chronology | 0/8 | 8 |
| gemma3-screen | derived | duration | 0/8 | 8 |
| gemma3-screen | derived | event_window | 0/8 | 8 |
| gemma3-screen | derived | interval | 0/8 | 8 |
| gemma3-screen | derived | policy | 0/8 | 8 |
| gemma3-screen | derived | revision | 0/8 | 8 |
| gemma3-screen | raw | calendar | 0/8 | 8 |
| gemma3-screen | raw | chronology | 0/8 | 8 |
| gemma3-screen | raw | duration | 0/8 | 8 |
| gemma3-screen | raw | event_window | 0/8 | 8 |
| gemma3-screen | raw | interval | 0/8 | 8 |
| gemma3-screen | raw | policy | 0/8 | 8 |
| gemma3-screen | raw | revision | 0/8 | 8 |
| lfm25-screen | derived | calendar | 4/8 | 0 |
| lfm25-screen | derived | chronology | 2/8 | 0 |
| lfm25-screen | derived | duration | 3/8 | 0 |
| lfm25-screen | derived | event_window | 3/8 | 0 |
| lfm25-screen | derived | interval | 3/8 | 0 |
| lfm25-screen | derived | policy | 3/8 | 0 |
| lfm25-screen | derived | revision | 2/8 | 0 |
| lfm25-screen | raw | calendar | 4/8 | 0 |
| lfm25-screen | raw | chronology | 2/8 | 0 |
| lfm25-screen | raw | duration | 4/8 | 0 |
| lfm25-screen | raw | event_window | 3/8 | 0 |
| lfm25-screen | raw | interval | 2/8 | 0 |
| lfm25-screen | raw | policy | 3/8 | 0 |
| lfm25-screen | raw | revision | 2/8 | 0 |
| llama32-screen | derived | calendar | 4/8 | 0 |
| llama32-screen | derived | chronology | 3/8 | 0 |
| llama32-screen | derived | duration | 4/8 | 0 |
| llama32-screen | derived | event_window | 4/8 | 0 |
| llama32-screen | derived | interval | 4/8 | 0 |
| llama32-screen | derived | policy | 4/8 | 0 |
| llama32-screen | derived | revision | 3/8 | 0 |
| llama32-screen | raw | calendar | 4/8 | 0 |
| llama32-screen | raw | chronology | 2/8 | 0 |
| llama32-screen | raw | duration | 4/8 | 0 |
| llama32-screen | raw | event_window | 3/8 | 2 |
| llama32-screen | raw | interval | 4/8 | 0 |
| llama32-screen | raw | policy | 4/8 | 0 |
| llama32-screen | raw | revision | 3/8 | 0 |
| llamacpp-qwen3-06b-screen | derived | calendar | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | chronology | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | duration | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | event_window | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | interval | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | policy | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | derived | revision | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | calendar | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | chronology | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | duration | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | event_window | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | interval | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | policy | 0/8 | 8 |
| llamacpp-qwen3-06b-screen | raw | revision | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | calendar | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | chronology | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | duration | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | event_window | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | interval | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | policy | 0/8 | 8 |
| ollama-qwen3-06b-screen | derived | revision | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | calendar | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | chronology | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | duration | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | event_window | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | interval | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | policy | 0/8 | 8 |
| ollama-qwen3-06b-screen | raw | revision | 0/8 | 8 |
| opencode-ollama-micro | derived | calendar | 0/2 | 2 |
| opencode-ollama-micro | derived | chronology | 0/2 | 2 |
| opencode-ollama-micro | derived | duration | 0/2 | 2 |
| opencode-ollama-micro | derived | event_window | 0/2 | 2 |
| opencode-ollama-micro | derived | interval | 0/2 | 2 |
| opencode-ollama-micro | derived | policy | 0/2 | 2 |
| opencode-ollama-micro | derived | revision | 0/2 | 2 |
| opencode-ollama-micro | raw | calendar | 0/1 | 1 |
| opencode-ollama-micro | raw | chronology | 0/2 | 2 |
| opencode-ollama-micro | raw | duration | 0/2 | 2 |
| opencode-ollama-micro | raw | event_window | 0/2 | 2 |
| opencode-ollama-micro | raw | interval | 0/2 | 2 |
| opencode-ollama-micro | raw | policy | 0/2 | 2 |
| opencode-ollama-micro | raw | revision | 0/2 | 2 |
| phi4-screen | derived | calendar | 0/8 | 8 |
| phi4-screen | derived | chronology | 0/8 | 8 |
| phi4-screen | derived | duration | 0/8 | 8 |
| phi4-screen | derived | event_window | 0/8 | 8 |
| phi4-screen | derived | interval | 0/8 | 8 |
| phi4-screen | derived | policy | 0/8 | 8 |
| phi4-screen | derived | revision | 0/8 | 8 |
| phi4-screen | raw | calendar | 0/8 | 8 |
| phi4-screen | raw | chronology | 0/8 | 8 |
| phi4-screen | raw | duration | 0/8 | 8 |
| phi4-screen | raw | event_window | 0/8 | 8 |
| phi4-screen | raw | interval | 0/8 | 8 |
| phi4-screen | raw | policy | 0/8 | 8 |
| phi4-screen | raw | revision | 0/8 | 8 |
| qwen3-06b-full | derived | calendar | 0/32 | 32 |
| qwen3-06b-full | derived | chronology | 0/32 | 32 |
| qwen3-06b-full | derived | duration | 0/32 | 32 |
| qwen3-06b-full | derived | event_window | 0/32 | 32 |
| qwen3-06b-full | derived | interval | 0/32 | 32 |
| qwen3-06b-full | derived | policy | 0/32 | 32 |
| qwen3-06b-full | derived | revision | 0/32 | 32 |
| qwen3-06b-full | raw | calendar | 0/32 | 32 |
| qwen3-06b-full | raw | chronology | 0/32 | 32 |
| qwen3-06b-full | raw | duration | 0/32 | 32 |
| qwen3-06b-full | raw | event_window | 0/32 | 32 |
| qwen3-06b-full | raw | interval | 0/32 | 32 |
| qwen3-06b-full | raw | policy | 0/32 | 32 |
| qwen3-06b-full | raw | revision | 0/32 | 32 |
| qwen3-06b-screen | derived | calendar | 0/8 | 8 |
| qwen3-06b-screen | derived | chronology | 0/8 | 8 |
| qwen3-06b-screen | derived | duration | 0/8 | 8 |
| qwen3-06b-screen | derived | event_window | 0/8 | 8 |
| qwen3-06b-screen | derived | interval | 0/8 | 8 |
| qwen3-06b-screen | derived | policy | 0/8 | 8 |
| qwen3-06b-screen | derived | revision | 0/8 | 8 |
| qwen3-06b-screen | raw | calendar | 0/8 | 8 |
| qwen3-06b-screen | raw | chronology | 0/8 | 8 |
| qwen3-06b-screen | raw | duration | 0/8 | 8 |
| qwen3-06b-screen | raw | event_window | 0/8 | 8 |
| qwen3-06b-screen | raw | interval | 0/8 | 8 |
| qwen3-06b-screen | raw | policy | 0/8 | 8 |
| qwen3-06b-screen | raw | revision | 0/8 | 8 |
| qwen3-4b-screen | derived | calendar | 0/8 | 8 |
| qwen3-4b-screen | derived | chronology | 0/8 | 8 |
| qwen3-4b-screen | derived | duration | 0/8 | 8 |
| qwen3-4b-screen | derived | event_window | 0/8 | 8 |
| qwen3-4b-screen | derived | interval | 0/8 | 8 |
| qwen3-4b-screen | derived | policy | 0/8 | 8 |
| qwen3-4b-screen | derived | revision | 0/8 | 8 |
| qwen3-4b-screen | raw | calendar | 0/8 | 8 |
| qwen3-4b-screen | raw | chronology | 0/8 | 8 |
| qwen3-4b-screen | raw | duration | 0/8 | 8 |
| qwen3-4b-screen | raw | event_window | 0/8 | 8 |
| qwen3-4b-screen | raw | interval | 0/8 | 8 |
| qwen3-4b-screen | raw | policy | 0/8 | 8 |
| qwen3-4b-screen | raw | revision | 0/8 | 8 |
| qwen35-screen | derived | calendar | 4/8 | 0 |
| qwen35-screen | derived | chronology | 7/8 | 0 |
| qwen35-screen | derived | duration | 5/8 | 0 |
| qwen35-screen | derived | event_window | 4/8 | 0 |
| qwen35-screen | derived | interval | 5/8 | 0 |
| qwen35-screen | derived | policy | 4/8 | 0 |
| qwen35-screen | derived | revision | 4/8 | 0 |
| qwen35-screen | raw | calendar | 4/8 | 0 |
| qwen35-screen | raw | chronology | 7/8 | 0 |
| qwen35-screen | raw | duration | 4/8 | 0 |
| qwen35-screen | raw | event_window | 4/8 | 0 |
| qwen35-screen | raw | interval | 5/8 | 0 |
| qwen35-screen | raw | policy | 4/8 | 0 |
| qwen35-screen | raw | revision | 4/8 | 0 |
| smollm3-full | derived | calendar | 0/32 | 32 |
| smollm3-full | derived | chronology | 0/32 | 32 |
| smollm3-full | derived | duration | 0/32 | 32 |
| smollm3-full | derived | event_window | 0/32 | 32 |
| smollm3-full | derived | interval | 0/32 | 32 |
| smollm3-full | derived | policy | 0/32 | 32 |
| smollm3-full | derived | revision | 0/32 | 32 |
| smollm3-full | raw | calendar | 0/32 | 32 |
| smollm3-full | raw | chronology | 0/32 | 32 |
| smollm3-full | raw | duration | 0/32 | 32 |
| smollm3-full | raw | event_window | 0/32 | 32 |
| smollm3-full | raw | interval | 0/32 | 32 |
| smollm3-full | raw | policy | 0/32 | 32 |
| smollm3-full | raw | revision | 0/32 | 32 |
| smollm3-screen | derived | calendar | 0/8 | 8 |
| smollm3-screen | derived | chronology | 0/8 | 8 |
| smollm3-screen | derived | duration | 0/8 | 8 |
| smollm3-screen | derived | event_window | 0/8 | 8 |
| smollm3-screen | derived | interval | 0/8 | 8 |
| smollm3-screen | derived | policy | 0/8 | 8 |
| smollm3-screen | derived | revision | 0/8 | 8 |
| smollm3-screen | raw | calendar | 0/8 | 8 |
| smollm3-screen | raw | chronology | 0/8 | 8 |
| smollm3-screen | raw | duration | 0/8 | 8 |
| smollm3-screen | raw | event_window | 0/8 | 8 |
| smollm3-screen | raw | interval | 0/8 | 8 |
| smollm3-screen | raw | policy | 0/8 | 8 |
| smollm3-screen | raw | revision | 0/8 | 8 |

## Live simulation

These rows use time-weighted in-force outcomes from saved live trajectories. Snapshot correctness and invalidation before publication are request-level diagnostics, distinct from in-force seconds. A freshness-only guard does not check semantic correctness; it only checks the state version. The oracle-guarded row uses ground-truth correctness and is a reference only, never a deployable controller.

| Run | Status | Track / adapter | Model identity | Trajectories completed / expected; requests | Snapshot correct / valid | Stale before publication / valid | Policy | Correct / wrong / undecided seconds (horizon) |
|---|---|---|---|---:|---:|---:|---|---:|
| claude-fable-51 | complete | native-client / claude_code | claude-fable-5-1 | 3/3; 6 | 2/2 | 2/2 | Unguarded publication | 0.164 / 10.500 / 79.336 (90.000) |
| claude-fable-51 | complete | native-client / claude_code | claude-fable-5-1 | 3/3; 6 | 2/2 | 2/2 | Freshness-only guard (state version only) | 0.000 / 0.000 / 90.000 (90.000) |
| claude-fable-51 | complete | native-client / claude_code | claude-fable-5-1 | 3/3; 6 | 2/2 | 2/2 | Oracle-guarded reference (not deployable) | 0.000 / 0.000 / 90.000 (90.000) |
| codex-gpt6-astra | complete | native-client / codex | gpt-6-astra | 3/3; 19 | 18/18 | 13/18 | Unguarded publication | 35.940 / 40.032 / 14.028 (90.000) |
| codex-gpt6-astra | complete | native-client / codex | gpt-6-astra | 3/3; 19 | 18/18 | 13/18 | Freshness-only guard (state version only) | 13.594 / 0.000 / 76.406 (90.000) |
| codex-gpt6-astra | complete | native-client / codex | gpt-6-astra | 3/3; 19 | 18/18 | 13/18 | Oracle-guarded reference (not deployable) | 13.594 / 0.000 / 76.406 (90.000) |
| qwen3-06b | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 3/3; 343 | 0/0 | 0/0 | Unguarded publication | 0.000 / 0.000 / 90.000 (90.000) |
| qwen3-06b | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 3/3; 343 | 0/0 | 0/0 | Freshness-only guard (state version only) | 0.000 / 0.000 / 90.000 (90.000) |
| qwen3-06b | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/qwen3-0.6b-bf16-c1899de) | 3/3; 343 | 0/0 | 0/0 | Oracle-guarded reference (not deployable) | 0.000 / 0.000 / 90.000 (90.000) |
| smollm3 | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 3/3; 38 | 0/0 | 0/0 | Unguarded publication | 0.000 / 0.000 / 90.000 (90.000) |
| smollm3 | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 3/3; 38 | 0/0 | 0/0 | Freshness-only guard (state version only) | 0.000 / 0.000 / 90.000 (90.000) |
| smollm3 | complete | direct-generation / mlx | LOCAL CHECKPOINT (models/benchmark-v1/smollm3-3b-4bit-d3a7e059) | 3/3; 38 | 0/0 | 0/0 | Oracle-guarded reference (not deployable) | 0.000 / 0.000 / 90.000 (90.000) |
