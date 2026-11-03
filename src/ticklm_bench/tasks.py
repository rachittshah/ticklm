"""Independent paired temporal tasks with primitive-arithmetic access conditions.

The oracle functions return semantic action IDs, not option letters. Derived views
provide arithmetic intermediate values only; they do not provide comparisons or answers.
"""

from __future__ import annotations

import itertools
import random
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

ORACLE_VERSION = 'ticklm-oracle-v1'
_ACTIONS = {
    'duration': {'alert': 'Raise an alert', 'wait': 'Keep waiting',
                 'restart': 'Restart the timer', 'stop': 'Stop monitoring'},
    'policy': {'alert': 'Raise an alert', 'wait': 'Keep waiting',
               'restart': 'Restart the timer', 'stop': 'Stop monitoring'},
    'interval': {'overlap': 'The intervals overlap', 'separate': 'The intervals do not overlap',
                 'unknown': 'The information is insufficient', 'invalid': 'An interval is invalid'},
    'revision': {'continue': 'Continue the current plan', 'replace': 'Adopt the replacement',
                 'stop': 'Stop the current plan', 'wait': 'Wait for another event'},
    'event_window': {'accept': 'Include the event', 'late': 'Exclude it as late',
                     'outside': 'Exclude it as outside the event window', 'pending': 'Keep it pending'},
    'calendar': {'previous_day': 'The previous calendar day', 'same_day': 'The same calendar day',
                 'next_day': 'The next calendar day', 'two_days_later': 'Two calendar days later'},
    'chronology': {'before': 'A is before C', 'after': 'A is after C',
                   'unknown': 'Their order is not determined', 'inconsistent': 'The statements are inconsistent'},
}


def _fraction(value):
    return Fraction(str(value))


def _chronology_closure(state):
    reach = {node: set() for node in 'ABC'}
    for key in ('r1', 'r2'):
        left, right = state[key].split('<')
        reach[left].add(right)
    for via in 'ABC':
        for left in 'ABC':
            if via in reach[left]:
                reach[left].update(reach[via])
    if any(node in reach[node] for node in 'ABC'):
        return 'inconsistent'
    if 'C' in reach['A']:
        return 'before'
    if 'A' in reach['C']:
        return 'after'
    return 'unknown'


def deterministic_oracle(case):
    """Return the action implied by the family and prompt-visible state."""
    family, s = case['family'], case['state']
    if family == 'duration':
        return 'alert' if _fraction(s['now']) - _fraction(s['start']) >= _fraction(s['limit']) else 'wait'
    if family == 'policy':
        elapsed = _fraction(s['now']) - _fraction(s['start'])
        lower, upper = _fraction(s['lower']), _fraction(s['upper'])
        condition = {'less': elapsed < lower, 'at_least': elapsed >= lower,
                     'inside': lower <= elapsed <= upper, 'outside': elapsed < lower or elapsed > upper}[s['policy']]
        return 'alert' if condition else 'wait'
    if family == 'interval':
        a, b, c, d = map(_fraction, (s['a_start'], s['a_end'], s['b_start'], s['b_end']))
        if a > b or c > d:
            return 'invalid'
        return 'overlap' if (max(a, c) <= min(b, d) if s['closure'] == 'closed' else
                             max(a, c) < min(b, d)) else 'separate'
    if family == 'revision':
        if s['target'] != s['plan'] or _fraction(s['event_at']) < _fraction(s['plan_at']):
            return 'continue'
        return {'cancel': 'stop', 'replace': 'replace'}[s['event']]
    if family == 'event_window':
        event, start, end = map(_fraction, (s['event_time'], s['start'], s['end']))
        if not start <= event < end:
            return 'outside'
        return 'late' if _fraction(s['arrival']) > _fraction(s['close']) else 'accept'
    if family == 'calendar':
        day = date.fromisoformat(s['date']).toordinal()
        instant = day * 1440 + s['minute'] + s['add'] + s['offset']
        return {-1: 'previous_day', 0: 'same_day', 1: 'next_day', 2: 'two_days_later'}[instant // 1440 - day]
    if family == 'chronology':
        return _chronology_closure(s)
    raise ValueError(f'Unknown family: {family}')


def audit_oracle(case):
    """Independently audit decisions using decimals, endpoint checks, or enumeration."""
    family, s = case['family'], case['state']
    def dec(key):
        return Decimal(str(s[key]))
    if family in ('duration', 'policy'):
        elapsed = dec('now') - dec('start')
        if family == 'duration':
            condition = elapsed >= dec('limit')
        elif s['policy'] == 'less':
            condition = elapsed < dec('lower')
        elif s['policy'] == 'at_least':
            condition = not elapsed < dec('lower')
        elif s['policy'] == 'inside':
            condition = not (elapsed < dec('lower') or elapsed > dec('upper'))
        else:
            condition = not (dec('lower') <= elapsed <= dec('upper'))
        return 'alert' if condition else 'wait'
    if family == 'interval':
        a, b, c, d = (dec(key) for key in ('a_start', 'a_end', 'b_start', 'b_end'))
        if a > b or c > d:
            return 'invalid'
        if s['closure'] == 'closed':
            overlap = any(a <= point <= b and c <= point <= d for point in (a, b, c, d))
        else:
            overlap = a < b and c < d and a < d and c < b
        return 'overlap' if overlap else 'separate'
    if family == 'revision':
        applies = s['plan'] == s['target'] and dec('event_at') >= dec('plan_at')
        return ('stop' if s['event'] == 'cancel' else 'replace') if applies else 'continue'
    if family == 'event_window':
        if dec('start') <= dec('event_time') < dec('end'):
            return 'accept' if dec('arrival') <= dec('close') else 'late'
        return 'outside'
    if family == 'calendar':
        utc = datetime.fromisoformat(s['date']).replace(tzinfo=UTC)
        instant = utc + timedelta(minutes=s['minute'] + s['add'])
        local = instant.astimezone(timezone(timedelta(minutes=s['offset'])))
        diff = (local.date() - utc.date()).days
        return {-1: 'previous_day', 0: 'same_day', 1: 'next_day', 2: 'two_days_later'}[diff]
    if family == 'chronology':
        orders = []
        relations = [s[key].split('<') for key in ('r1', 'r2')]
        for order in itertools.permutations('ABC'):
            if all(order.index(left) < order.index(right) for left, right in relations):
                orders.append(order.index('A') < order.index('C'))
        if not orders:
            return 'inconsistent'
        if all(orders):
            return 'before'
        if not any(orders):
            return 'after'
        return 'unknown'
    raise ValueError(f'Unknown family: {family}')


def _states(family, split, index, rng):
    base = (100 if split == 'dev' else 10000) + rng.randrange(120) / 4
    if family == 'duration':
        limit = rng.randrange(8, 24) / 4
        first = {'start': base, 'now': base + limit - 0.25, 'limit': limit}
        second = {**first, 'now': base + limit + (index % 2) * 0.25}
    elif family == 'policy':
        lower = rng.randrange(8, 20) / 4
        upper = lower + 2
        mode = ('less', 'at_least', 'inside', 'outside')[index % 4]
        elapsed = (lower - 0.25, lower) if mode in ('less', 'at_least') or index % 8 < 4 else (upper, upper + 0.25)
        first = {'start': base, 'now': base + elapsed[0], 'lower': lower, 'upper': upper, 'policy': mode}
        second = {**first, 'now': base + elapsed[1]}
    elif family == 'interval':
        closure = 'closed' if index % 2 == 0 else 'half-open'
        a_end = base + 3
        first = {'a_start': base, 'a_end': a_end, 'b_start': a_end if closure == 'closed' else a_end - 0.25,
                 'b_end': a_end + 2, 'closure': closure}
        second = {**first, 'b_start': a_end + 0.25 if closure == 'closed' else a_end}
    elif family == 'revision':
        event = ('cancel', 'replace')[(index // 2) % 2]
        first = {'plan': 'P', 'target': 'P', 'plan_at': base, 'event_at': base - 0.25,
                 'now': base + 2, 'event': event}
        if index % 2:
            first = {**first, 'event_at': base + 0.25, 'target': 'Q'}
            second = {**first, 'target': 'P'}
        else:
            second = {**first, 'event_at': base}
    elif family == 'event_window':
        first = {'start': base, 'end': base + 4, 'event_time': base - 0.25,
                 'arrival': base + 6, 'close': base + 6}
        if index % 2:
            first = {**first, 'event_time': base + 3.75}
            second = {**first, 'arrival': base + 6.25}
        else:
            second = {**first, 'event_time': base}
    elif family == 'calendar':
        year = 2024 if split == 'dev' else 2032
        d = (date(year, 2, 28) + timedelta(days=index % 3) if index % 4 == 0 else
             date(year, 1, 1) + timedelta(days=rng.randrange(366)))
        if index % 2:
            first = {'date': d.isoformat(), 'minute': 30, 'add': 0, 'offset': -60}
            second = {**first, 'offset': 0}
        else:
            first = {'date': d.isoformat(), 'minute': 1380, 'add': 30, 'offset': 0}
            second = {**first, 'offset': 60}
    else:
        pairs = [('B<C', 'C<B'), ('B<C', 'B<A'), ('C<B', 'B<A')]
        left, right = pairs[index % 3]
        names = (('launch', 'review', 'delivery', 'inspection', 'handoff', 'meeting', 'approval', 'filing')
                 if split == 'dev' else
                 ('arrival', 'rehearsal', 'release', 'check', 'repair', 'packing', 'dispatch', 'audit'))
        event_a, event_b, event_c = rng.sample(names, 3)
        first = {'r1': 'A<B', 'r2': left, 'a': event_a, 'b': event_b, 'c': event_c}
        second = {**first, 'r2': right}
    return first, second


def _body(family, s):
    if family == 'duration':
        return (f"Alert when elapsed time since start is at least {s['limit']} seconds; otherwise wait. "
                f"Start: {s['start']} seconds. Now: {s['now']} seconds.")
    if family == 'policy':
        policies = {'less': 'strictly less than the lower bound', 'at_least': 'at least the lower bound',
                    'inside': 'between the lower and upper bounds inclusive',
                    'outside': 'below the lower bound or above the upper bound'}
        return (f"Alert when elapsed time is {policies[s['policy']]}; otherwise wait. "
                f"Start: {s['start']} seconds; now: {s['now']} seconds; "
                f"lower bound: {s['lower']} seconds; upper bound: {s['upper']} seconds.")
    if family == 'interval':
        endpoints = 'both included' if s['closure'] == 'closed' else 'start included, end excluded'
        return (f"Determine whether intervals A and B overlap. Both use {endpoints}. "
                f"A starts {s['a_start']} and ends {s['a_end']}; "
                f"B starts {s['b_start']} and ends {s['b_end']}.")
    if family == 'revision':
        return ("A matching event at or after plan issue applies: cancellation stops the plan and "
                "replacement adopts the replacement. Earlier events or events for another plan are ignored. "
                f"Current plan {s['plan']} issued at {s['plan_at']} seconds; event {s['event']} "
                f"targets {s['target']} at {s['event_at']} seconds; now {s['now']} seconds.")
    if family == 'event_window':
        return ("Include an event if its event time is in the window (start inclusive, end exclusive) "
                "and its arrival is at or before close. Events outside the window are excluded as outside; "
                "events inside but arriving after close are excluded as late. "
                f"Window start {s['start']}, end {s['end']}; event time {s['event_time']}; "
                f"arrival {s['arrival']}; close {s['close']}; all times in seconds.")
    if family == 'calendar':
        return ("All dates use the Gregorian calendar. Starting at the stated UTC date and minute of day "
                "(midnight is minute 0), add the duration, then apply the explicit fixed offset from UTC. "
                "Compare the resulting local calendar date with the original UTC date. "
                f"UTC date {s['date']}; minute {s['minute']}; add {s['add']} minutes; "
                f"local UTC offset {s['offset']} minutes. No daylight-saving changes apply.")
    return (f"Event A is {s['a']}; B is {s['b']}; C is {s['c']}. "
            "Each statement X<Y means X happened strictly before Y. Determine the relation between A and C. "
            "If more than one order is possible choose undetermined; if no ordering satisfies the statements "
            f"choose inconsistent. Statements: {s['r1']}; {s['r2']}.")


def _derived(family, s):
    if family in ('duration', 'policy'):
        return f"Elapsed since start: {s['now'] - s['start']} seconds."
    if family == 'interval':
        return f"Interval A length: {s['a_end'] - s['a_start']}; interval B length: {s['b_end'] - s['b_start']}."
    if family == 'revision':
        return f"Plan age: {s['now'] - s['plan_at']} seconds; event age: {s['now'] - s['event_at']} seconds."
    if family == 'event_window':
        return f"Window duration: {s['end'] - s['start']} seconds."
    if family == 'calendar':
        return f"UTC minute count after addition from the original midnight: {s['minute'] + s['add']}."
    return 'Number of named events: 3; number of supplied statements: 2.'


def build_suite(split='dev', pairs_per_family=16, seed=1103):
    """Build independently generated raw/derived cases in complete paired episodes."""
    if split not in ('dev', 'test') or not isinstance(pairs_per_family, int) or pairs_per_family <= 0:
        raise ValueError('Expected dev/test and positive pairs_per_family')
    cases = []
    for family, actions in _ACTIONS.items():
        for index in range(pairs_per_family):
            rng = random.Random(f'{seed}:{split}:{family}:{index}:ticklm-v1')
            states = list(_states(family, split, index, rng))
            gold = [deterministic_oracle({'family': family, 'state': state}) for state in states]
            if gold[0] == gold[1]:
                raise ValueError(f'Pair does not flip: {family} {index}')
            positions = list(range(4))
            random.Random(f'{seed}:{split}:{family}:{index // 4}:choices-v1').shuffle(positions)
            pos_first = positions[index % 4]
            pos_second = positions[(index + 1 + index // 4 % 3) % 4]
            ordered = [None] * 4
            ordered[pos_first], ordered[pos_second] = gold
            remaining = [action for action in actions if action not in gold]
            rng.shuffle(remaining)
            for slot in range(4):
                if ordered[slot] is None:
                    ordered[slot] = remaining.pop()
            choices = dict(zip('ABCD', ordered, strict=True))
            if rng.randrange(2):
                states.reverse()
            episode_id = f'{split}-{family}-{index:05d}'
            options = '\n'.join(f'{letter}. {actions[action]}' for letter, action in choices.items())
            for variant, state in enumerate(states):
                action = deterministic_oracle({'family': family, 'state': state})
                answer = next(letter for letter, candidate in choices.items() if candidate == action)
                rule_id = f'rule-{family}-{state.get("policy", state.get("closure", "standard"))}'
                template_id = f'tpl-{split}-{index % 2}'
                for view in ('raw', 'derived'):
                    header = ('Choose the best answer; return its letter only.' if index % 2 == 0 else
                              'Apply the stated rules and answer with one letter.')
                    body = _body(family, state)
                    prefix = f"Arithmetic information: {_derived(family, state)}\n" if view == 'derived' else ''
                    cases.append({'id': f'{episode_id}-{variant}-{view}', 'episode_id': episode_id,
                                  'pair_id': episode_id, 'family': family, 'variant': variant,
                                  'view': view, 'split': split, 'prompt': f'{header}\n{prefix}{body}\n{options}',
                                  'choices': choices.copy(), 'answer': answer, 'oracle_version': ORACLE_VERSION,
                                  'state': state.copy(), 'rule_id': rule_id, 'template_id': template_id})
    return cases
