"""Add external arithmetic measurements to prompts without making decisions.

The clock-features view provides calculated durations, so it measures a model with
an external arithmetic aid rather than a change to its intrinsic temporal ability.
"""

from copy import deepcopy


def with_feature_view(cases, view):
    """Deep-copy cases and optionally prefix primitive clock measurements."""
    if view not in ('raw', 'clock-features'):
        raise ValueError(f'Unknown feature view: {view}')
    result = deepcopy(cases)
    if view == 'raw':
        return result
    for case in result:
        family = case['family']
        state = case['state']
        if family == 'silence':
            elapsed = state['now'] - state['last_event']
            features = (f"Elapsed since last event: {elapsed} seconds; "
                        f"silence threshold: {state['threshold']} seconds.")
        elif family == 'deadline':
            remaining = state['deadline'] - state['now']
            features = (f"Time remaining until deadline: {remaining} seconds; "
                        f"required duration: {state['duration']} seconds.")
        elif family == 'sustained':
            start = state['reset'] if state['reset'] is not None else state['started']
            elapsed = state['now'] - start
            features = (f"Elapsed since latest reset or start: {elapsed} seconds; "
                        f"sustained threshold: {state['threshold']} seconds.")
        elif family == 'relevance':
            relative = state['event_at'] - state['plan_at']
            features = (f"Event time relative to plan issue: {relative} seconds; "
                        f"plan ID: {state['plan_id']}; target ID: {state['target_id']}; "
                        f"event: {state['event']}.")
        else:
            raise ValueError(f'Unknown family: {family}')
        case['prompt'] = f'Clock features: {features}\n{case["prompt"]}'
    return result
