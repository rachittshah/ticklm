# Qwen3 temporal weight-surgery journey

**Date:** 3 November 2026  
**Branch:** `experiment/temporal-weight-surgery`  
**Updated through:** 2026-11-03 06:22 UTC  
**Status:** The train-only curriculum run is active. The held-out test set has not been evaluated.

```mermaid
timeline
    title Weight-surgery development timeline — 3 Nov 2026
    06:22 UTC : Fable review and data red-team corrections recorded
              : Base dev measured; LoRA-4 and all-layer LoRA completed
              : Frozen linear probe completed; curriculum run active
              : Held-out test remains unopened
```

This log records the work and evidence so far. The raw artifacts, including failed attempts and adapter checkpoints, remain under the locally ignored `artifacts/weight-surgery-20261103/` directory. The compact run index is [journey-index.json](../evidence/weight-surgery-20261103/journey-index.json). No model weights have been uploaded; no upload is recorded.

## Research review and checkpoint identity

Before training, a Claude review was run with requested and resolved model `claude-fable-5-1` at `xhigh` effort. The saved JSON records `elapsed_s: 204.97264645798714` (204.97 seconds), plus an estimated cost of `$1.00581875`. The prompt requested a response of at most 650 words. Separately, wrapper metadata records `requested_max_tokens: 8` and `max_tokens_enforcement: not_enforced`; this was a technical wrapper setting, not a prompt instruction, and it did not constrain the response. This is a recorded model review, not an independent research verification. It emphasized paired counterfactual scoring, random-label controls, retention checks, and keeping any result scoped to these synthetic rules. Its full prompt and response remain in `artifacts/weight-surgery-20261103/fable-plan-prompt.txt` and `fable-plan-review.json`.

The model path is `models/qwen3-0.6b-bf16-c1899de`; run manifests record revision `c1899de289a04d12100db370d81485cdf75e47ca`. The revision is caller-supplied in the manifest; local checkpoint and tokenizer file hashes are also recorded. MLX reports 596,049,920 loaded model parameters. The checkpoint file headers contain 751,632,384 stored tensor elements; this stored-element count includes tied or duplicated tensors and is not the active parameter count.

## Data corrections and baseline

Before the baseline, the synthetic benchmark was red-teamed and corrected to vary fractional margins on both sides of decision boundaries, include equality only for a subset of positives, separate relevant events from mismatched or earlier events, and balance reset presence across sustained-condition outcomes. Pair members keep the same choices and differ in one input. These cases provide rule-following controls; they do not establish that an LLM is needed for the underlying decisions.

The source hash for `surgery_data.py` changed after baseline when public rendering and robustness-control helpers were added. Saved manifests show the development data itself stayed byte-identical across baseline, both LoRA runs, the frozen probe, and the active curriculum run: `32e346074349a11f3db710adfc672c162e14f21c653c6eee1868126c0f518656`. The train dataset for the first two LoRA runs and frozen probe is `44e947dc2d6792c5bbe0b91e686bcc95e9f047a63bae97c337b625466ce502e8`. The curriculum run's train dataset has a new hash, `e053a80c29c82772c3e98a9a4113ad4a873d32c3f9889614ac4bd2036fa803ec`; its dev hash is unchanged. No model evaluation on the test split has occurred.

The unmodified base scored **37.890625%** on 256 development cases; both-cases-correct pair accuracy was **0.78125%**. The deterministic reference scored **100%**, as expected for these generated rules. The recorded base command was an `evaluate` run on `dev` with 32 pairs per family, seed 1103, and batch size 4; its full effective arguments and hashes are in `base-dev/run.json`. A separate base-generation run on three fixed prompts is saved under `base-generation/` for later comparison.

## Candidate 1: four-layer LoRA

The `lora4-choice-s1103` run used rank-8 LoRA on query/value projections in the last four layers, restricted four-choice cross-entropy, learning rate `3e-4`, batch size 4, seed 1103, and 160 steps over 1,024 train cases. It exposed 163,840 trainable parameters, **0.02749%** of the 596,049,920 loaded parameters. Development checkpoints at steps 32, 64, 96, 128, and 160 are all retained.

The development-selected checkpoint suggestion is step 96: **41.40625%** item accuracy, with pair-both-correct accuracy still **0.78125%**, unchanged from base. The retained non-temporal development smoke set scored 41.67% at that checkpoint. Adapter reload parity passed with a maximum logit difference of zero. This does not support a temporal-reasoning claim: the small item-accuracy movement did not improve the paired counterfactual endpoint. Selection remains a suggestion to review with retention and generation evidence.

## Candidate 2: all-layer LoRA with full-vocabulary loss

The `lora28-full-s1103` run completed 600 steps with rank 8 on query/value projections in all 28 layers, full-vocabulary cross-entropy, learning rate `3e-4`, batch size 8, and training-retention anchors every five steps. It trained 1,146,880 parameters (**0.1924%** of the loaded parameter count). The CLI's best-development suggestion is checkpoint 240: **51.171875%** item accuracy, **2.34375%** pair-both-correct, and 62.5% retention accuracy. At that checkpoint, deadline, silence, and sustained-condition accuracy were each 50%; relevance accuracy was 54.6875%. The final step-600 retention score was 72.9167%, but it was not the best temporal checkpoint. Reload parity passed.

These scores are consistent with weak choice-format or broad priors: the time-focused families remain at 50%, and pair consistency is low. They do not justify a temporal-reasoning claim. The automatic checkpoint selection is only a suggestion; retention and free-generation comparisons still need review.

The exact command observed for this run was:

```sh
.venv/bin/python -m temporal_bridge.surgery train --model models/qwen3-0.6b-bf16-c1899de --revision c1899de289a04d12100db370d81485cdf75e47ca --output artifacts/weight-surgery-20261103/lora28-full-s1103 --steps 600 --batch-size 8 --eval-batch-size 4 --learning-rate 0.0003 --rank 8 --layers 28 --seed 1103 --pairs-per-family 128 --dev-pairs-per-family 32 --loss full --anchor-every 5
```

For the earlier completed runs, exact effective arguments are in their `run.json` files; the original shell lines are not retained in the run logs.

## Frozen linear-probe control

The frozen-base hidden-state probe used the same 1,024 training cases and 256 development cases. It reached **44.53125%** item accuracy and **3.125%** pair-both-correct accuracy on dev. This is a readout control, not a fine-tuned model. Its pair score is above both base (**0.78125%**) and the best all-layer LoRA suggestion (**2.34375%**), though it remains low. The probe result shows that a frozen representation supports some readout on this dataset; it does not establish that temporal reasoning improved through weight adaptation.

## Candidate 3: train-only time curriculum with paired minibatches

This development run is active as `lora28-curriculum-s1103`. It uses the all-layer rank-8, full-vocabulary setup, 1,000 steps, batch size 8, learning rate `3e-4`, anchors every five steps, plus `--curriculum --paired-batches`. The curriculum applies positive time scale/offset transformations to training inputs and batches paired counterfactuals together. The original dev hash remains unchanged; test data remains unopened. The saved `run.json` records the full arguments and data/source hashes. At this writing the run has not produced a selected checkpoint, so there is no result to report.

The exact command observed for this run is:

```sh
.venv/bin/python -m temporal_bridge.surgery train --model models/qwen3-0.6b-bf16-c1899de --revision c1899de289a04d12100db370d81485cdf75e47ca --output artifacts/weight-surgery-20261103/lora28-curriculum-s1103 --steps 1000 --batch-size 8 --eval-batch-size 4 --learning-rate 0.0003 --rank 8 --layers 28 --seed 1103 --pairs-per-family 128 --dev-pairs-per-family 32 --loss full --anchor-every 5 --curriculum --paired-batches
```

## Validation and remaining limits

The first saved full-suite transcript contains two `AttributeError` failures in probe tests (`fit_probe` and `extract_probe_features`) because those tests landed while the older module version was loaded. The failed attempt remains at `verification-pytest.txt`. The later full suite passed **188 tests with 1 skipped**, and Ruff reported **All checks passed!**; the saved outputs are `verification-pytest-v3.txt` and `verification-ruff-v3.txt`. Code checkpoint `c4b41bb` was committed after that validation.

Memory values written by MLX are process high-water marks, not isolated model-inference memory: `get_peak_memory()` counts from process start or its last reset ([MLX documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)). Free-generation comparison of base and adapted weights, final checkpoint choice, held-out test evaluation, and the live-clock replay remain outstanding. Any eventual claim must stay limited to measured behavior on these synthetic rules and this setup.
