# Temporal surgery data

`temporal_bridge.surgery_data` builds reproducible synthetic, four-choice decisions. `build_cases(split, pairs_per_family=32, seed=1103)` returns two cases for each pair in each of four families; the default is 256 cases per split. `split` is `train`, `dev`, or `test`. Each case includes an ID, pair ID, split, family, prompt, answer letter, action ID, variant, raw state, and a mapping of letters to action IDs. `deterministic_answer(case)` computes the choice from the family, raw state, and choices; it does not read the supplied answer or action. The choice descriptions and policy are included in the prompt; identifiers, variant numbers, split names, and computed labels are not.

Each pair has identical choices and changes exactly one input, causing the correct action to change. Equality counts as enough time for silence and sustained conditions and as on time for deadlines. Both sides of the decision boundary use varying fractional margins; equality occurs in only a subset of positive cases. Times include fractional seconds. Sustained-condition cases include resets in equal numbers for both outcomes. A reset restarts the duration immediately, with the condition continuously true again; sustained pairs change either the observation time or the reset time.

Relevance cases specify a current plan's ID and issue time and an event's target ID, type, and time. A revision or cancellation applies only to the matching plan and only at or after the issue time. Earlier or mismatched events leave the current plan in effect; a matching revision calls for adopting the replacement and a matching cancellation calls for stopping. Relevance pairs change either event time or target ID, keeping all other inputs fixed. Answer letters are balanced within each relevance event and outcome at the default size. Events targeting another plan serve as unrelated events.

The generator seeds an independent random stream for each split, family, and pair index; the same arguments reproduce the same cases, and extending a split leaves earlier pairs unchanged. Answer positions are exactly balanced within each family whenever the number of pairs is divisible by four, including the default. With other counts they are approximately balanced. Each pair keeps its order of options, while assignment of the first or second variant is randomized. A seed changes the samples but the same seed and split are required for an identical benchmark.

Train uses three distinct body templates per family; development and test each use a different held-out body template. Headers are also distinct. Base times occupy disjoint ranges near 10–30, 1000–1020, and 10000–10020 seconds for train, dev, and test; thresholds use distinct ranges near 3–4.75, 13–14.75, and 23–24.75 seconds. Tests compare body text across splits without the headers, and compare templates after normalizing numeric values and common event/reset variants. This measures transfer to larger numbers and limited rewording of the same rules. It does **not** test unseen temporal rules, other event categories, real-world clocks, or broad temporal competence. Plan IDs and the two event types are reused across splits. Episodes can repeat within a split, so the effective number of distinct problems can be lower than the case count. Choice order also varies; all scores must preserve the provided prompt and answer together.

`render_prompt(split, family, state, choices, template_index=0)` exposes the same renderer used by the benchmark. For train, `template_index` selects one of three templates modulo three; dev and test each use their held-out template. The caller supplies the raw state and A–D mapping of action IDs.

`build_robustness_cases(split='test', pairs_per_family=32, seed=1103)` returns three **separate** control lists; it does not alter `build_cases` or replace the primary test set:

- `option_rotation` moves every answer option one position forward, including D to A, and recalculates the gold letter.
- `time_translation` adds 128 seconds to every non-null absolute timestamp, leaving durations, thresholds, choices, and the answer unchanged.
- `in_range_numbers` samples with an independent, recorded seed in the train numeric domain, then renders with the requested split's wording. At the default test split this separates number range from the changed body wording. For dev and test, its question bodies do not exactly match train bodies. Because the finite numeric domain is shared, individual raw states can repeat across independently seeded episodes; this is not a guarantee of unseen train-range states.

Each control has its own IDs and pair IDs. `parent_id` and `sourcecase` name the generated source case; `source_split` and `source_seed` identify its generator stream, and translation records `time_offset`. A source ID alone does not identify a seed. Compare controls separately against their corresponding source cases, and report them as additional evaluations rather than folding them into the primary test score. In-range examples are independent episodes, so their parent cases come from the recorded train-range seed, not from the primary test set.

`build_curriculum_cases(pairs_per_family=128, seed=1103)` is an optional, separate training set derived only from `build_cases('train', ...)`. It returns 1,024 examples in 512 pairs by default. Each pair receives a deterministic common positive scale selected from 0.25, 0.5, 1, 2, 4, 8, and 16, and a common offset from 0 through 100,000 seconds. Offsets are stratified across zero, 1–99, 100–999, 1,000–9,999, 10,000–99,999, and 100,000 seconds, with a shuffled bin order and a sampled value within each bin. All absolute timestamps are multiplied by the scale and then shifted by the offset; thresholds and durations are multiplied by the scale only. Event types, plan IDs, choice orders, and correct answers are unchanged. Prompts are rerendered with the three training templates. Each case records `scale`, `time_offset`, source ID, source split, and source seed. The generator rejects a transformed case if its deterministic answer differs from the source.

The curriculum broadens the training number distribution but retains the synthetic training rules, templates, and source-pair structures. It does not change the fixed development or test sets, and it does not establish model generalization without a separate evaluation. Its exact prompts are checked against those held-out sets; this check does not imply that all numerical patterns or equivalent causal states are unique across splits.

`build_retention_cases(split='test')` returns 48 split-disjoint multiple-choice questions: eight vocabulary, grammar, and basic-fact questions plus forty elementary arithmetic questions. Train questions are available for preservation anchors and are disjoint from dev and test questions. Labels are balanced and choices are shuffled. This is a narrow smoke check, strongly weighted toward arithmetic, and cannot establish general language retention. A separate generated-answer check is still needed as specified in the experiment plan.

Example:

```python
from temporal_bridge.surgery_data import build_cases, build_retention_cases, deterministic_answer

cases = build_cases('test')
assert all(deterministic_answer(case) == case['answer'] for case in cases)
retention = build_retention_cases('test')
```

The deterministic reference is a ceiling for these synthetic rules, not evidence that an LLM is required. Compare base and adapted models on identical cases, retain pair-level predictions, select settings using development data, and reserve the test split for the final evaluation.
