"""Post-hoc diagnostic added after the first Qwen 0.6B run; not preregistered.

Use only as a secondary parser sensitivity check, never as a replacement for
the frozen strict score.
"""

import re

_OPTION_LINE = re.compile(r'([ABCD])\. (.+)')


def option_text_answer(case, text):
    """Return a label only when all response text exactly matches a rendered option line.

    Parse the visible prompt, not the gold label or internal choice metadata. This
    diagnostic deliberately rejects prose and letter-only answers; primary scoring
    remains the responsibility of the frozen strict parser.
    """
    if not isinstance(case, dict) or not isinstance(case.get('prompt'), str):
        return None
    if not isinstance(text, str):
        return None

    prompt_lines = case['prompt'].strip().splitlines()
    if len(prompt_lines) < 4:
        return None

    rendered_options = {}
    for line in prompt_lines[-4:]:
        match = _OPTION_LINE.fullmatch(line.strip())
        if match is None:
            return None
        label, option_text = match.groups()
        if not option_text.strip() or label in rendered_options:
            return None
        rendered_options[line.strip()] = label

    if set(rendered_options.values()) != set('ABCD'):
        return None
    return rendered_options.get(text.strip())
