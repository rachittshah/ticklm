from ticklm_bench.diagnostics import option_text_answer


def _case(answer='A'):
    return {
        'answer': answer,
        'prompt': (
            'Choose the action for this event.\n'
            'A. Keep it pending\n'
            'B. Exclude it as late\n'
            'C. Exclude it as outside the event window\n'
            'D. Include the event'
        ),
    }


def test_maps_only_an_exact_rendered_option_line_without_using_gold():
    case = _case(answer='A')

    assert option_text_answer(case, 'D. Include the event') == 'D'


def test_rejects_letter_only_and_unlabeled_option_text():
    case = _case()

    assert option_text_answer(case, 'D') is None
    assert option_text_answer(case, 'Include the event') is None


def test_rejects_free_prose_and_extra_lines_even_when_they_contain_an_option():
    case = _case()

    assert option_text_answer(case, 'D. Include the event because it arrived on time.') is None
    assert option_text_answer(case, 'D. Include the event\nThis is my answer.') is None


def test_reads_only_the_last_four_prompt_lines_and_fails_closed_on_malformed_options():
    case = _case()
    case['prompt'] = 'Earlier text: A. Include the event\n' + case['prompt']
    assert option_text_answer(case, 'A. Keep it pending') == 'A'

    case['prompt'] = case['prompt'].replace('D. Include the event', 'C. Include the event')
    assert option_text_answer(case, 'C. Include the event') is None


def test_non_string_and_missing_inputs_are_ignored():
    assert option_text_answer(None, 'D. Include the event') is None
    assert option_text_answer(_case(), None) is None
    assert option_text_answer({'prompt': None}, 'D. Include the event') is None
