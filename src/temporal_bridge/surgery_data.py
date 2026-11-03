"""Reproducible paired temporal decisions and narrow retention smoke checks."""

from __future__ import annotations

import random

_SPLITS = {'train': 0, 'dev': 1, 'test': 2}
_ACTIONS = {
    'silence': {'alert': 'Raise an alert now', 'wait': 'Keep waiting',
                'restart': 'Restart the clock now', 'cancel': 'Cancel monitoring'},
    'deadline': {'start': 'Start the task now', 'defer': 'Do not start the task',
                 'shorten': 'Assume the task takes no time', 'late': 'Start after the deadline'},
    'relevance': {'continue': 'Continue the existing plan', 'replan': 'Adopt the replacement plan',
                  'stop': 'Stop the cancelled plan', 'wait': 'Wait for another event'},
    'sustained': {'act': 'Act now', 'monitor': 'Keep monitoring',
                  'reset': 'Reset the timer now', 'stop': 'Stop monitoring permanently'},
}
_HEADERS = {
    'train': 'Choose the one action required by the rule. Reply with its letter only.',
    'dev': 'Apply the stated policy to this snapshot. Give only A, B, C, or D.',
    'test': 'At the snapshot, which action follows? Answer with a single choice letter.',
}


def _action(family, state):
    if family == 'silence':
        return 'alert' if state['now'] - state['last_event'] >= state['threshold'] else 'wait'
    if family == 'deadline':
        return 'start' if state['now'] + state['duration'] <= state['deadline'] else 'defer'
    if family == 'relevance':
        if state['target_id'] != state['plan_id'] or state['event_at'] < state['plan_at']:
            return 'continue'
        return {'revision': 'replan', 'cancel': 'stop'}[state['event']]
    if family == 'sustained':
        start = state['reset'] if state['reset'] is not None else state['started']
        return 'act' if state['now'] - start >= state['threshold'] else 'monitor'
    raise ValueError(f'Unknown family: {family}')


def deterministic_answer(case):
    """Return the choice letter calculated from a temporal case's raw state."""
    action = _action(case['family'], case['state'])
    for letter, candidate in case['choices'].items():
        if candidate == action:
            return letter
    raise ValueError(f'Missing action choice: {action}')


_BODY_TEMPLATES = {
    'silence': [
        "Alert if silence lasts at least {threshold} seconds; otherwise wait. Last event: {last_event} seconds. Now: {now} seconds.",
        "The last event occurred at {last_event} seconds and the current time is {now} seconds. Wait unless silence reaches {threshold} seconds; then alert.",
        "A sensor reports time {now} seconds, with its previous event at {last_event} seconds. Raise an alert at {threshold} seconds of silence or more, else wait.",
        "Silence policy: wait below {threshold} seconds, alert at or above it. At time {now} seconds, the preceding event was at {last_event} seconds.",
        "For a signal last seen at {last_event} seconds, decide at {now} seconds. Alert once {threshold} seconds of silence have passed, including equality; otherwise wait.",
    ],
    'deadline': [
        "Start only if finishing by the deadline is possible, including equality; otherwise do not start. Now: {now} seconds; duration: {duration} seconds; deadline: {deadline} seconds.",
        "At {now} seconds a task would take {duration} seconds. Its deadline is {deadline} seconds. Start if it finishes no later than the deadline, else do not start.",
        "The task deadline is {deadline} seconds. Starting at {now} seconds takes {duration} seconds. Permit starting if completion is at or before the deadline; otherwise do not start.",
        "A task needs {duration} seconds and must finish by {deadline} seconds. At time {now} seconds, start only when it can finish at the deadline or earlier; else do not start.",
        "At the snapshot {now} seconds, a job needs {duration} seconds. Its cutoff is {deadline} seconds. If starting now finishes on time, including exactly at cutoff, start; otherwise do not start.",
    ],
    'relevance': [
        "An event affects a plan only when its target matches that plan and its time is at or after plan issue. Otherwise continue. An applicable revision means adopt replacement; an applicable cancellation means stop. Plan {plan_id} issued at {plan_at} seconds; event {event} for {target_id} at {event_at} seconds.",
        "Current plan {plan_id} was issued at {plan_at} seconds. A {event} targeting {target_id} occurred at {event_at} seconds. Ignore events before issue or targeting another plan; otherwise adopt a revision or stop on cancellation.",
        "A {event} for plan {target_id} has timestamp {event_at} seconds. The current plan is {plan_id}, issued at {plan_at} seconds. Continue if target differs or event predates issue; otherwise adopt a revision and stop for cancellation. Equality is applicable.",
        "Keep the current plan if an event targets another plan or precedes issue. At or after issue, a matching revision replaces the plan and a matching cancellation stops it. Event: {event}, target {target_id}, time {event_at} seconds. Current plan: {plan_id}, issued {plan_at} seconds.",
        "Plan {plan_id} began at {plan_at} seconds. An event at {event_at} seconds says {event} for plan {target_id}. Events apply only to the matching plan and at or after its issue. Continue if not applicable; otherwise adopt a revision or stop for cancellation.",
    ],
    'sustained': [
        "Act after at least {threshold} seconds continuously true; otherwise monitor. Reset restarts the duration and the condition immediately holds again. Began: {started} seconds; latest reset: {reset}; now: {now} seconds.",
        "The condition started at {started} seconds; its latest reset is {reset}; now is {now} seconds. Reset restarts the true interval immediately. Monitor below {threshold} seconds of continuous truth and act at or above it.",
        "At {now} seconds, a condition has been true since {started} seconds, except its latest reset was {reset}. A reset immediately restarts truth. Act when the current uninterrupted interval reaches {threshold} seconds, else monitor.",
        "A condition starts at {started} seconds, and its latest reset is {reset}. After reset it immediately holds again. At {now} seconds, act if the uninterrupted interval is at least {threshold} seconds; otherwise monitor.",
        "Monitor until {threshold} seconds of continuous truth, then act, including at equality. The condition began at {started} seconds, latest reset was {reset}, and the snapshot is {now} seconds. Each reset restarts truth immediately.",
    ],
}


def _prompt(split, family, state, choices, index):
    template_index = index % 3 if split == 'train' else (3 if split == 'dev' else 4)
    values = dict(state)
    if family == 'sustained':
        values['reset'] = 'none' if state['reset'] is None else str(state['reset']) + ' seconds'
    body = _BODY_TEMPLATES[family][template_index].format(**values)
    options = '\n'.join(f'{letter}. {_ACTIONS[family][action]}' for letter, action in choices.items())
    return f'{_HEADERS[split]}\n{body}\n{options}'


def _pair_states(family, index, split, rng):
    # Split-specific domains and quarter-second values make equality exact in binary.
    base = (10, 1000, 10000)[_SPLITS[split]] + rng.randrange(0, 80) / 4 + 0.25
    threshold = (3, 13, 23)[_SPLITS[split]] + rng.randrange(0, 8) / 4
    below = rng.choice([0.25, 0.5, 0.75, 1.0])
    above = 0 if index % 4 == 0 else rng.choice([0.25, 0.5, 0.75, 1.0])
    if family == 'silence':
        a = {'now': base + threshold - below, 'last_event': base, 'threshold': threshold}
        b = {**a, 'now': base + threshold + above}
    elif family == 'deadline':
        a = {'now': base, 'duration': threshold, 'deadline': base + threshold - below}
        b = {**a, 'deadline': base + threshold + above}
    elif family == 'relevance':
        event = ('cancel', 'revision')[(index // 2) % 2]
        a = {'event': event, 'plan_id': 'P', 'target_id': 'P',
             'plan_at': base, 'event_at': base - below}
        if index % 2:
            a = {**a, 'event_at': base + above, 'target_id': 'Q'}
            b = {**a, 'target_id': 'P'}
        else:
            b = {**a, 'event_at': base + above}
    elif index % 4 == 2:
        a = {'now': base + threshold + above, 'started': base - 3, 'reset': base + above + below,
             'threshold': threshold}
        b = {**a, 'reset': base}
    else:
        reset = base if index % 4 == 1 else None
        started = base - 3 if reset is not None else base
        a = {'now': base + threshold - below, 'started': started, 'reset': reset,
             'threshold': threshold}
        b = {**a, 'now': base + threshold + above}
    return a, b


def build_cases(split, pairs_per_family=32, seed=1103):
    """Build counterfactual pairs; all choices are balanced when pair count is divisible by four."""
    if split not in _SPLITS or not isinstance(pairs_per_family, int) or pairs_per_family <= 0:
        raise ValueError('Expected train/dev/test and a positive pair count')
    cases = []
    for family, actions in _ACTIONS.items():
        for i in range(pairs_per_family):
            rng = random.Random(f'{seed}:{split}:{family}:{i}:temporal-v1')
            states = list(_pair_states(family, i, split, rng))
            a, b = (_action(family, state) for state in states)
            # Balance each event/action stratum independently, with a fresh order per block.
            group = (i // 4) * 2 + i % 2 if family == 'relevance' else i
            stratum = (i // 2) % 2 if family == 'relevance' else 0
            positions = list(range(4))
            random.Random(f'{seed}:{split}:{family}:{stratum}:{group // 4}:positions-v1').shuffle(positions)
            pos_a = positions[group % 4]
            pos_b = positions[(group + 1 + (group // 4) % 3) % 4]
            rest = [action for action in actions if action not in (a, b)]
            rng.shuffle(rest)
            ordered = [None] * 4
            ordered[pos_a], ordered[pos_b] = a, b
            for index in range(4):
                if ordered[index] is None:
                    ordered[index] = rest.pop()
            choices = dict(zip('ABCD', ordered, strict=True))
            if rng.randrange(2):
                states.reverse()
            pair_id = f'{split}-{family}-{i:05d}'
            for variant, state in enumerate(states):
                case = {'id': f'{pair_id}-{variant}', 'pair_id': pair_id, 'split': split,
                        'family': family, 'variant': variant, 'state': state,
                        'choices': choices.copy(), 'correct_action': _action(family, state)}
                case['prompt'] = _prompt(split, family, state, choices, i)
                case['answer'] = deterministic_answer(case)
                cases.append(case)
    return cases



def render_prompt(split, family, state, choices, template_index=0):
    """Render a decision using the same split-specific wording as the benchmark."""
    if split not in _SPLITS:
        raise ValueError('Expected train/dev/test')
    return _prompt(split, family, state, choices, template_index)


def build_robustness_cases(split='test', pairs_per_family=32, seed=1103):
    """Build separate choice, clock-translation, and train-range control datasets."""
    primary = build_cases(split, pairs_per_family, seed)
    range_seed = f'{seed}:in-range-control-v1'
    range_source = build_cases('train', pairs_per_family, range_seed)
    controls = {'option_rotation': [], 'time_translation': [], 'in_range_numbers': []}
    absolute_fields = {'now', 'last_event', 'deadline', 'plan_at', 'event_at', 'started', 'reset'}
    for name, cases in (('option_rotation', primary), ('time_translation', primary),
                        ('in_range_numbers', range_source)):
        for source in cases:
            state = source['state'].copy()
            choices = source['choices'].copy()
            if name == 'option_rotation':
                choices = {letter: source['choices']['ABCD'[(i - 1) % 4]]
                           for i, letter in enumerate('ABCD')}
            elif name == 'time_translation':
                state = {key: value + 128 if key in absolute_fields and value is not None else value
                         for key, value in state.items()}
            case = {**source, 'id': f'control-{name}-{split}-{source["id"]}',
                    'pair_id': f'control-{name}-{split}-{source["pair_id"]}',
                    'split': split, 'state': state, 'choices': choices, 'control': name,
                    'parent_id': source['id'], 'sourcecase': source['id'],
                    'source_split': source['split'],
                    'source_seed': range_seed if name == 'in_range_numbers' else seed}
            if name == 'time_translation':
                case['time_offset'] = 128
            index = int(source['pair_id'].rsplit('-', 1)[1])
            case['prompt'] = render_prompt(split, case['family'], state, choices, index)
            case['answer'] = deterministic_answer(case)
            controls[name].append(case)
    return controls



def build_curriculum_cases(pairs_per_family=128, seed=1103):
    """Affinely transform train pairs while preserving their temporal relationships."""
    source_cases = build_cases('train', pairs_per_family, seed)
    absolute_fields = {'now', 'last_event', 'deadline', 'plan_at', 'event_at', 'started', 'reset'}
    duration_fields = {'threshold', 'duration'}
    scales = (0.25, 0.5, 1, 2, 4, 8, 16)
    cases = []
    for source in source_cases:
        index = int(source['pair_id'].rsplit('-', 1)[1])
        rng = random.Random(f'{seed}:{source["pair_id"]}:curriculum-v1')
        scale = rng.choice(scales)
        offset_bins = [(0, 0), (1, 99), (100, 999), (1000, 9999),
                       (10000, 99999), (100000, 100000)]
        random.Random(f'{seed}:{source["family"]}:{index // 6}:curriculum-bins').shuffle(offset_bins)
        low, high = offset_bins[index % 6]
        offset = rng.randint(low, high)
        state = {}
        for key, value in source['state'].items():
            if value is not None and key in absolute_fields:
                state[key] = scale * value + offset
            elif value is not None and key in duration_fields:
                state[key] = scale * value
            else:
                state[key] = value
        choices = source['choices'].copy()
        case = {**source, 'id': f'curriculum-{source["id"]}',
                'pair_id': f'curriculum-{source["pair_id"]}',
                'state': state, 'choices': choices, 'control': 'curriculum',
                'parent_id': source['id'], 'sourcecase': source['id'],
                'source_split': 'train', 'source_seed': seed,
                'scale': scale, 'time_offset': offset}
        case['prompt'] = render_prompt('train', case['family'], state, choices, index)
        if deterministic_answer(case) != source['answer']:
            raise ValueError(f'Curriculum transform changed the answer for {source["id"]}')
        cases.append(case)
    return cases


_FACTS = {
    'train': [
        ('Which word is a synonym of happy?', 'glad', 'angry', 'slow', 'empty'),
        ('Which word is opposite to cold?', 'hot', 'blue', 'soft', 'late'),
        ('Which is a mammal?', 'whale', 'salmon', 'trout', 'shark'),
        ('Which shape has three sides?', 'triangle', 'square', 'circle', 'pentagon'),
        ('What is the plural of child?', 'children', 'childs', 'childes', 'childrens'),
        ('Which is a fruit?', 'apple', 'carrot', 'onion', 'potato'),
        ('Which word is a verb?', 'run', 'table', 'blue', 'quiet'),
        ('Which material is attracted to a magnet?', 'iron', 'wood', 'glass', 'paper'),
    ],
    'dev': [
        ('Which word is a synonym of tiny?', 'small', 'vast', 'heavy', 'late'),
        ('Which word is opposite to early?', 'late', 'soft', 'green', 'near'),
        ('Which animal is a bird?', 'sparrow', 'lizard', 'frog', 'salmon'),
        ('Which shape has four equal sides?', 'square', 'triangle', 'circle', 'oval'),
        ('What is the plural of mouse?', 'mice', 'mouses', 'mices', 'mousees'),
        ('Which is a vegetable?', 'carrot', 'peach', 'plum', 'pear'),
        ('Which word is an adjective?', 'bright', 'quickly', 'run', 'table'),
        ('Which object is transparent?', 'clear glass', 'brick', 'wood', 'stone'),
    ],
    'test': [
        ('Which word is a synonym of rapid?', 'fast', 'quiet', 'tall', 'dry'),
        ('Which word is opposite to shallow?', 'deep', 'red', 'thin', 'bright'),
        ('Which animal is a reptile?', 'lizard', 'robin', 'whale', 'frog'),
        ('Which shape has five sides?', 'pentagon', 'triangle', 'square', 'circle'),
        ('What is the plural of tooth?', 'teeth', 'tooths', 'toothes', 'teeths'),
        ('Which is a citrus fruit?', 'lemon', 'grape', 'apple', 'banana'),
        ('Which word is an adverb?', 'quietly', 'quiet', 'chair', 'walk'),
        ('Which substance is liquid at room temperature?', 'water', 'granite', 'iron', 'wood'),
    ],
}


def build_retention_cases(split='test'):
    """Return a small, split-disjoint collection of language and arithmetic smoke checks."""
    if split not in _SPLITS:
        raise ValueError('Expected train/dev/test')
    questions = list(_FACTS[split])
    for n in range(1 + 20 * _SPLITS[split], 9 + 20 * _SPLITS[split]):
        questions.extend([
            (f'What is {n} plus 7?', str(n + 7), str(n + 6), str(n + 8), str(n + 9)),
            (f'What is {n + 12} minus 5?', str(n + 7), str(n + 6), str(n + 8), str(n + 9)),
            (f'What is twice {n}?', str(n * 2), str(n * 2 + 1), str(n * 2 + 2), str(n * 2 + 3)),
            (f'What is {n} plus 10?', str(n + 10), str(n + 9), str(n + 11), str(n + 12)),
            (f'What is {n + 15} minus 9?', str(n + 6), str(n + 5), str(n + 7), str(n + 8)),
        ])
    rng = random.Random(f'{split}:retention-v1')
    cases = []
    for i, (question, answer, *distractors) in enumerate(questions):
        rng.shuffle(distractors)
        distractors.insert(i % 4, answer)
        choices = dict(zip('ABCD', distractors, strict=True))
        options = '\n'.join(f'{letter}. {value}' for letter, value in choices.items())
        cases.append({'id': f'{split}-retention-{i:05d}', 'split': split,
                      'family': 'retention', 'prompt': f'Choose one answer; reply with its letter only.\n{question}\n{options}',
                      'choices': choices, 'answer': 'ABCD'[i % 4]})
    return cases
