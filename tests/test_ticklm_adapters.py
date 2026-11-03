import json
from types import SimpleNamespace

import pytest

from ticklm_bench import adapters

MESSAGES = [{'role': 'user', 'content': 'Choose A or B.'}]


class FakeResponse:
    def __init__(self, events):
        self.lines = iter([b'data: ' + json.dumps(e).encode() + b'\n' for e in events] + [b'data: [DONE]\n'])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return self.lines


def test_http_stream_separates_reasoning_and_validates_resolved_alias(monkeypatch):
    events = [
        {'model': 'vendor/model-2026', 'choices': [{'delta': {'reasoning': 'private thought'}}]},
        {'model': 'vendor/model-2026', 'choices': [{'delta': {'content': 'A'}, 'finish_reason': 'stop'}]},
        {'model': 'vendor/model-2026', 'choices': [], 'usage': {'prompt_tokens': 11,
         'completion_tokens': 4, 'total_tokens': 15, 'cost': .001}},
    ]
    requests = []
    def open_fake(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(events)
    monkeypatch.setattr(adapters.urllib.request, 'urlopen', open_fake)
    config = {'type': 'http', 'model': 'vendor/model', 'endpoint': 'https://example.test/v1',
              'api_key': 'secret-key', 'allowed_resolved_models': ['vendor/model-2026'],
              'hard_token_cap': True, 'extra_body': {'reasoning': {'enabled': False},
                                                    'provider': {'allow_fallbacks': False,
                                                                 'max_price': {'prompt': .1}}}}
    result = adapters.create_adapter(config).generate(MESSAGES, max_tokens=10, timeout_s=5, request_id='req-1')
    assert result['status'] == 'ok' and result['text'] == 'A'
    assert result['reasoning'] == 'private thought'
    assert result['resolved_model'] == 'vendor/model-2026'
    assert result['usage'] == {'input_tokens': 11, 'output_tokens': 4, 'total_tokens': 15, 'cost_usd': .001}
    assert result['raw_safe_metadata']['request_policy']['reasoning'] == {'enabled': False}
    assert result['raw_safe_metadata']['request_policy']['provider']['max_price'] == {'prompt': .1}
    assert json.loads(requests[0][0].data)['max_tokens'] == 10
    assert 'secret-key' not in json.dumps(result)


def test_http_identity_mismatch_is_unscorable_and_redacts_exception(monkeypatch):
    def wrong(*args, **kwargs):
        return FakeResponse([{'model': 'unexpected', 'choices': [{'delta': {'content': 'A'},
                                                                      'finish_reason': 'stop'}]}])
    monkeypatch.setattr(adapters.urllib.request, 'urlopen', wrong)
    adapter = adapters.create_adapter({'type': 'http', 'model': 'x', 'endpoint': 'https://example.test/v1',
                                       'api_key': 'secret-key'})
    result = adapter.generate(MESSAGES, max_tokens=10, timeout_s=5, request_id='req-2')
    assert result['status'] == 'identity_mismatch'
    def broken(*args, **kwargs):
        raise RuntimeError('server echoed secret-key')
    monkeypatch.setattr(adapters.urllib.request, 'urlopen', broken)
    assert 'secret-key' not in json.dumps(adapter.generate(MESSAGES, max_tokens=10, timeout_s=5, request_id='req-3'))


def test_http_preserves_partial_text_on_transport_failure(monkeypatch):
    class Interrupted(FakeResponse):
        def __iter__(self):
            yield b'data: {"model":"x","choices":[{"delta":{"content":"A"}}]}\n'
            raise OSError('connection failed')
    monkeypatch.setattr(adapters.urllib.request, 'urlopen', lambda *args, **kwargs: Interrupted([]))
    result = adapters.create_adapter({'type': 'http', 'model': 'x', 'endpoint': 'https://example.test/v1'}).generate(
        MESSAGES, max_tokens=10, timeout_s=5, request_id='req-4')
    assert result['status'] == 'error' and result['text'] == 'A' and result['raw_text'] == 'A'


def test_mlx_lazy_load_stream_and_template_capability(monkeypatch):
    calls = []
    class Tokenizer:
        chat_template = '{% if enable_thinking %}yes{% endif %}'
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            return [1, 2]
    monkeypatch.setattr(adapters, '_load_mlx', lambda config: (calls.append('load') or object(), Tokenizer()))
    monkeypatch.setattr(adapters, '_mlx_stream', lambda model, tokenizer, tokens, max_tokens: iter([
        SimpleNamespace(text='A', generation_tokens=1, prompt_tokens=2, finish_reason='stop')]))
    adapter = adapters.create_adapter({'type': 'mlx', 'model': '/local/checkpoint', 'model_id': 'model@rev',
                                       'revision': 'rev', 'thinking': False})
    for request_id in ('one', 'two'):
        result = adapter.generate(MESSAGES, max_tokens=10, timeout_s=5, request_id=request_id)
        assert result['text'] == 'A' and result['usage']['output_tokens'] == 1
        assert result['capabilities']['hard_token_cap'] is True
        assert result['raw_safe_metadata']['thinking_template_support'] is True
        assert result['resolved_model'] is None
        assert result['raw_safe_metadata']['identity_validation'] == 'caller_supplied_local_identity'
    assert calls == ['load']


def test_thinking_tags_separated_from_visible_answer():
    assert adapters._separate_thinking('<think>reason</think>A') == ('A', 'reason')
    assert adapters._separate_thinking('<think>reason') == ('', 'reason')
    assert adapters._visible_so_far('<thi') == ''
    assert adapters._visible_so_far('<think>reason</think>A') == 'A'


def test_native_event_result_uncapped_and_fresh_messages(monkeypatch):
    class Native:
        def generate(self, prompt, max_tokens, emit):
            emit('first_token', elapsed_s=.2)
            emit('delta', text='A')
            emit('result', text='A', model='claude-id', input_tokens=8, output_tokens=1,
                 finish_reason='completed', estimated_cost_usd=.002)
    monkeypatch.setattr(adapters, '_new_native', lambda config, timeout: Native())
    adapter = adapters.create_adapter({'type': 'claude_code', 'model': 'opus',
                                       'allowed_resolved_models': ['claude-id']})
    result = adapter.generate(MESSAGES, max_tokens=10, timeout_s=5, request_id='r')
    assert result['text'] == 'A' and result['first_visible_s'] == .2
    assert result['capabilities']['hard_token_cap'] is False
    assert result['usage']['cost_usd'] == .002
    assert adapter.generate([{'role': 'system', 'content': 'x'}, *MESSAGES], max_tokens=2,
                            timeout_s=5, request_id='q')['status'] == 'unsupported_messages'


@pytest.mark.parametrize('bad', [[], [{'role': 'user', 'content': ''}], [{'role': 'tool', 'content': 'x'}]])
def test_invalid_messages_raise(bad):
    adapter = adapters.create_adapter({'type': 'http', 'model': 'x', 'endpoint': 'http://localhost:42/v1'})
    with pytest.raises(ValueError):
        adapter.generate(bad, max_tokens=2, timeout_s=1, request_id='r')


def test_dev_preflight_saves_result_and_refuses_overwrite(tmp_path, monkeypatch):
    from ticklm_bench import preflight

    class Fake:
        def __init__(self):
            self.metadata = {'provider': 'fake'}
        def generate(self, messages, **kwargs):
            assert messages[0]['role'] == 'user'
            return {'status': 'ok', 'text': 'A', 'usage': {}, 'request': kwargs['request_id']}
        def close(self):
            pass

    monkeypatch.setattr(preflight, 'create_adapter', lambda config: Fake())
    out = tmp_path / 'preflight.json'
    argv = ['--type', 'mlx', '--model', 'unused', '--output', str(out)]
    preflight.main(argv)
    result = json.loads(out.read_text())
    assert result['case']['split'] == 'dev' and result['result']['text'] == 'A'
    with pytest.raises(FileExistsError):
        preflight.main(argv)


def test_preflight_http_rejects_nonlocal_endpoints(tmp_path):
    from ticklm_bench import preflight
    with pytest.raises(SystemExit):
        preflight.main(['--type', 'http', '--model', 'x', '--endpoint', 'https://example.com/v1',
                        '--output', str(tmp_path / 'out.json')])
    assert not (tmp_path / 'out.json').exists()


def test_preflight_http_passes_loopback_endpoint(tmp_path, monkeypatch):
    from ticklm_bench import preflight
    configs = []
    class Fake:
        def __init__(self):
            self.metadata = {}
        def generate(self, messages, **kwargs):
            return {'status': 'ok', 'text': 'A'}
        def close(self):
            pass
    monkeypatch.setattr(preflight, 'create_adapter', lambda config: configs.append(config) or Fake())
    preflight.main(['--type', 'http', '--model', 'x', '--endpoint', 'http://127.0.0.1:18084/v1',
                    '--output', str(tmp_path / 'out.json')])
    assert configs[0]['endpoint'] == 'http://127.0.0.1:18084/v1'


def test_opencode_config_allows_only_loopback_provider_and_denies_tools():
    adapter = adapters.create_adapter({'type': 'opencode', 'model': 'ollama/qwen3:0.6b'})
    try:
        config = adapter._opencode_config()
        assert config['providers']['ollama']['settings']['baseURL'] == 'http://127.0.0.1:11434/v1'
        assert config['policies'][:2] == [
            {'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
            {'action': 'provider.use', 'resource': 'ollama', 'effect': 'allow'}]
        assert {'action': 'tool.use', 'resource': '*', 'effect': 'deny'} in config['policies']
        assert adapter.capabilities['hard_token_cap'] is False
    finally:
        adapter.close()
    with pytest.raises(ValueError):
        adapters.create_adapter({'type': 'opencode', 'model': 'openai/gpt-6'})
    with pytest.raises(ValueError):
        adapters.create_adapter({'type': 'opencode', 'model': 'ollama/qwen3:0.6b', 'api_key': 'unused'})


def test_opencode_json_events_preserve_final_and_reject_tool_use():
    events = [{'type': 'text', 'part': {'type': 'text', 'text': 'A'}},
              {'type': 'step_finish', 'part': {'reason': 'stop', 'tokens': {'input': 20, 'output': 2}}}]
    result = adapters._parse_opencode_events('\n'.join(json.dumps(x) for x in events))
    assert result['text'] == 'A' and result['finish_reason'] == 'stop'
    assert result['usage']['input_tokens'] == 20 and result['usage']['output_tokens'] == 2
    with pytest.raises(ValueError, match='tool'):
        adapters._parse_opencode_events(json.dumps({'type': 'tool_use', 'part': {}}))


def test_opencode_isolation_rejects_ambient_config_source(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (tmp_path / 'opencode.json').write_text('{}')
    with pytest.raises(RuntimeError, match='ambient'):
        adapters._assert_no_ambient_opencode_config(work, managed_paths=[])


def test_opencode_isolation_reuses_same_private_directories(monkeypatch):
    seen = []
    def run(command, *, cwd, env, timeout):
        seen.append((cwd, dict(env)))
        root = cwd.parent
        paths = '\n'.join(f'{key} {root / key / "opencode"}' for key in ('config', 'data', 'cache', 'state'))
        return 0, paths, ''
    monkeypatch.setattr(adapters, '_run_opencode', run)
    adapter = adapters.create_adapter({'type': 'opencode', 'model': 'ollama/qwen3:0.6b'})
    try:
        assert adapter.audit_opencode()['xdg_paths_inside_isolation']
        assert adapter.audit_opencode()['xdg_paths_inside_isolation']
        assert seen[0][0] == seen[1][0]
        assert seen[0][1]['XDG_CACHE_HOME'] == seen[1][1]['XDG_CACHE_HOME']
        assert 'OPENAI_API_KEY' not in seen[0][1]
        assert 'ANTHROPIC_API_KEY' not in seen[0][1]
    finally:
        adapter.close()
