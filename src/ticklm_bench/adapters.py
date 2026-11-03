"""Bounded model adapters; clients with unenforced caps remain a separate track."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_OLLAMA_MODEL = 'ollama/qwen3:0.6b'
_OLLAMA_DIGEST = '3bae9c93586b27bedaa979979733c2b0edd1d0defc745e9638f2161192a0ccf0'


def _load_mlx(config):
    from mlx_lm import load
    return load(config['model'], revision=config.get('revision'))


def _mlx_stream(model, tokenizer, tokens, max_tokens):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    return stream_generate(model, tokenizer, prompt=tokens, max_tokens=max_tokens,
                           sampler=make_sampler(temp=0.0))


def _new_native(config, timeout):
    if config['type'] == 'claude_code':
        from temporal_bridge.claude_backend import ClaudeCodeBackend
        return ClaudeCodeBackend(model=config['model'], effort=config.get('effort', 'high'),
                                 profile=config.get('profile', 'startup'), timeout_s=timeout)
    from temporal_bridge.codex_backend import CodexBackend
    return CodexBackend(model=config['model'], effort=config.get('effort', 'high'), timeout_s=timeout)


def _validate(messages, max_tokens, timeout_s, request_id):
    if (not isinstance(messages, list) or not messages or
            any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
                or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages)):
        raise ValueError('messages must contain nonempty system, user, or assistant text')
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError('max_tokens must be a positive integer')
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError('timeout_s must be positive and finite')
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', request_id):
        raise ValueError('request_id must be a safe explicit identifier')


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, '[REDACTED]') if secret else value
    if isinstance(value, dict):
        return {k: _redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secret) for v in value]
    return value


def _separate_thinking(text):
    reasoning = []
    def remove(match):
        reasoning.append(match.group(1))
        return ''
    visible = re.sub(r'<think>(.*?)</think>', remove, text, flags=re.DOTALL)
    if '<think>' in visible:
        visible, _, partial = visible.partition('<think>')
        reasoning.append(partial)
    return visible, '\n'.join(reasoning)


def _visible_so_far(text):
    visible, _ = _separate_thinking(text)
    # A tag may be split across streaming chunks; do not count its prefix as output.
    for length in range(min(len('<think>') - 1, len(visible)), 0, -1):
        if visible.endswith('<think>'[:length]):
            return visible[:-length]
    return visible


def _safe_request_policy(body):
    policy = {'temperature': body['temperature'], 'max_tokens': body['max_tokens']}
    if 'seed' in body:
        policy['seed'] = body['seed']
    reasoning = body.get('reasoning')
    if isinstance(reasoning, dict):
        policy['reasoning'] = {key: reasoning[key] for key in ('enabled', 'effort', 'max_tokens', 'exclude')
                               if key in reasoning and isinstance(reasoning[key], (str, int, float, bool))}
    provider = body.get('provider')
    if isinstance(provider, dict):
        selected = {key: provider[key] for key in ('order', 'only', 'ignore', 'allow_fallbacks',
                                                    'require_parameters', 'quantizations', 'data_collection')
                    if key in provider}
        price = provider.get('max_price')
        if isinstance(price, dict):
            selected['max_price'] = {key: price[key] for key in ('prompt', 'completion', 'image', 'request')
                                     if key in price and _number(price[key]) is not None}
        policy['provider'] = selected
    return policy


def _parse_opencode_events(output):
    result = {'text': '', 'reasoning': '', 'finish_reason': None,
              'usage': {'input_tokens': None, 'output_tokens': None, 'total_tokens': None, 'cost_usd': None},
              'event_types': []}
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get('type')
        part = event.get('part') or {}
        if not isinstance(part, dict):
            raise TypeError('invalid opencode event')
        if isinstance(kind, str) and kind not in result['event_types']:
            result['event_types'].append(kind)
        if kind in ('tool_use', 'tool', 'tool_call') or part.get('type') in ('tool', 'tool_use', 'tool_call'):
            raise ValueError('tool event in isolated model-only run')
        if kind == 'error':
            raise ValueError('opencode error event')
        if kind == 'text' and isinstance(part.get('text'), str):
            result['text'] += part['text']
        if kind == 'reasoning' and isinstance(part.get('text'), str):
            result['reasoning'] += part['text']
        if kind in ('step_finish', 'step-finish'):
            result['finish_reason'] = part.get('reason')
            tokens = part.get('tokens') or {}
            if isinstance(tokens, dict):
                result['usage']['input_tokens'] = _number(tokens.get('input'))
                result['usage']['output_tokens'] = _number(tokens.get('output'))
                a, b = result['usage']['input_tokens'], result['usage']['output_tokens']
                if a is not None and b is not None:
                    result['usage']['total_tokens'] = a + b
    return result


def _run_opencode(command, *, cwd, env, timeout):
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        raise TimeoutError from exc
    return proc.returncode, stdout, stderr


def _assert_no_ambient_opencode_config(work, managed_paths):
    for directory in (work, *work.parents):
        for relative in ('opencode.json', 'opencode.jsonc', '.opencode/opencode.json', '.opencode/opencode.jsonc'):
            if (directory / relative).exists():
                raise RuntimeError('ambient opencode project configuration exists')
    if any(Path(path).exists() for path in managed_paths):
        raise RuntimeError('ambient opencode managed configuration exists')


class Adapter:
    def __init__(self, config):
        if not isinstance(config, dict) or config.get('type') not in {'mlx', 'http', 'claude_code', 'codex', 'opencode'}:
            raise ValueError('unknown adapter type')
        if not isinstance(config.get('model'), str) or not config['model']:
            raise ValueError('model is required')
        allowed = config.get('allowed_resolved_models', [])
        if not isinstance(allowed, list) or any(not isinstance(x, str) or not x for x in allowed):
            raise ValueError('allowed resolved models must be explicit strings')
        self._secret = config.get('api_key')
        if self._secret is not None and (not isinstance(self._secret, str) or not self._secret):
            raise ValueError('api key must be nonempty when supplied')
        self._config = {k: v for k, v in config.items() if k != 'api_key'}
        self._model = None
        self._tokenizer = None
        self._native = None
        self._native_timeout = None
        self._opencode_temp = None
        kind = config['type']
        self.capabilities = {'hard_token_cap': True if kind == 'mlx' else
                             (False if kind in ('claude_code', 'codex', 'opencode') else config.get('hard_token_cap')),
                             'sampling_seed': False if kind in ('mlx', 'claude_code', 'codex', 'opencode') else config.get('sampling_seed'),
                             'streaming': kind != 'opencode',
                             'thinking': config.get('thinking_supported')}
        self.metadata = {'type': kind, 'provider': config.get('provider', kind),
                         'requested_model': config.get('model_id', config['model']),
                         'revision': config.get('revision'), 'quantization': config.get('quantization'),
                         'capabilities': dict(self.capabilities)}
        if kind == 'http':
            url = config.get('endpoint')
            if not isinstance(url, str):
                raise ValueError('HTTP endpoint required')
            parsed = urllib.parse.urlsplit(url)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or
                    parsed.password or parsed.query or parsed.fragment or
                    (parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'))):
                raise ValueError('endpoint must be HTTPS or loopback HTTP without credentials or query')
            self._endpoint = url.rstrip('/')
            self.metadata['endpoint'] = _redact(self._endpoint, self._secret)
        if kind == 'opencode':
            if (config['model'] != _OLLAMA_MODEL or self._secret is not None or
                    config.get('model_id', _OLLAMA_MODEL) != _OLLAMA_MODEL or
                    config.get('endpoint', 'http://127.0.0.1:11434/v1') != 'http://127.0.0.1:11434/v1'):
                raise ValueError('opencode is restricted to the pinned local Ollama model and endpoint')
            self.metadata['endpoint'] = 'http://127.0.0.1:11434/v1'

    def _base(self, request_id):
        return {'text': '', 'raw_text': '', 'reasoning': '', 'requested_model': self.metadata['requested_model'],
                'resolved_model': None, 'elapsed_s': None, 'first_visible_s': None,
                'usage': {'input_tokens': None, 'output_tokens': None, 'total_tokens': None, 'cost_usd': None},
                'finish_reason': None, 'status': 'error', 'error': None,
                'capabilities': dict(self.capabilities), 'provider': self.metadata['provider'],
                'raw_safe_metadata': {'request_id': request_id}}

    def _identity(self, result):
        resolved = result['resolved_model']
        allowed = [self.metadata['requested_model'], *self._config.get('allowed_resolved_models', [])]
        if resolved is None:
            result['raw_safe_metadata']['identity_validation'] = 'unavailable'
        elif resolved not in allowed:
            result['status'] = 'identity_mismatch'
            result['error'] = {'code': 'resolved_model_not_allowed'}
            result['raw_safe_metadata']['identity_validation'] = 'mismatch'
        else:
            result['raw_safe_metadata']['identity_validation'] = 'matched'

    def generate(self, messages, *, max_tokens, timeout_s, request_id):
        _validate(messages, max_tokens, timeout_s, request_id)
        started = time.monotonic()
        result = self._base(request_id)
        result['raw_safe_metadata']['messages_sha256'] = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        result['raw_safe_metadata']['requested_max_tokens'] = max_tokens
        result['raw_safe_metadata']['requested_thinking'] = self._config.get('thinking')
        result['raw_safe_metadata']['retry_policy'] = ('native_client_internal_unknown' if self._config['type']
                                                       in ('claude_code', 'codex') else 'none_by_adapter')
        try:
            if self._config['type'] == 'mlx':
                self._generate_mlx(result, messages, max_tokens, timeout_s, started)
            elif self._config['type'] == 'http':
                self._generate_http(result, messages, max_tokens, timeout_s, started, request_id)
            elif self._config['type'] == 'opencode':
                self._generate_opencode(result, messages, timeout_s, started)
            else:
                self._generate_native(result, messages, max_tokens, timeout_s, started)
            if result['status'] == 'ok' and self._config['type'] not in ('mlx', 'opencode'):
                self._identity(result)
        except TimeoutError:
            result['status'], result['error'] = 'timeout', {'code': 'timeout'}
        except urllib.error.HTTPError as exc:
            result['status'], result['error'] = 'http_error', {'code': 'http_error', 'http_status': exc.code}
        except Exception as exc:  # noqa: BLE001 - provider messages may contain credentials
            # Provider and OS exceptions can contain credentials; retain only the type.
            result['status'], result['error'] = 'error', {'code': type(exc).__name__}
        result['elapsed_s'] = time.monotonic() - started
        return _redact(result, self._secret)

    def _generate_mlx(self, result, messages, max_tokens, timeout_s, started):
        if self._model is None:
            self._model, self._tokenizer = _load_mlx(self._config)
        template = getattr(self._tokenizer, 'chat_template', None)
        template_text = template if isinstance(template, str) else None
        support = 'enable_thinking' in template_text if template_text else None
        kwargs = {'tokenize': True, 'add_generation_prompt': True}
        if support and self._config.get('thinking') is not None:
            kwargs['enable_thinking'] = self._config['thinking']
        tokens = self._tokenizer.apply_chat_template(messages, **kwargs)
        result['raw_safe_metadata'].update({'chat_template_sha256': hashlib.sha256(template_text.encode()).hexdigest()
                                            if template_text else None,
                                            'thinking_template_support': support,
                                            'generation_temperature': 0.0,
                                            'timeout_enforcement': 'checked_between_stream_events'})
        text = []
        for event in _mlx_stream(self._model, self._tokenizer, tokens, max_tokens):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError
            text.append(event.text)
            result['raw_text'] = ''.join(text)
            result['text'], result['reasoning'] = _separate_thinking(result['raw_text'])
            if result['first_visible_s'] is None and _visible_so_far(result['raw_text']):
                result['first_visible_s'] = time.monotonic() - started
            result['usage']['input_tokens'] = _number(event.prompt_tokens)
            result['usage']['output_tokens'] = _number(event.generation_tokens)
            result['finish_reason'] = event.finish_reason or result['finish_reason']
        result['raw_text'] = ''.join(text)
        result['text'], result['reasoning'] = _separate_thinking(result['raw_text'])
        if result['usage']['input_tokens'] is not None and result['usage']['output_tokens'] is not None:
            result['usage']['total_tokens'] = result['usage']['input_tokens'] + result['usage']['output_tokens']
        result['raw_safe_metadata']['identity_validation'] = 'caller_supplied_local_identity'
        result['raw_safe_metadata']['local_model_path'] = self._config['model']
        result['status'] = 'ok' if result['finish_reason'] else 'incomplete_stream'

    def _generate_http(self, result, messages, max_tokens, timeout_s, started, request_id):
        body = dict(self._config.get('extra_body', {}))
        if any(k in body for k in ('model', 'messages', 'max_tokens', 'stream', 'temperature')):
            raise ValueError('extra_body cannot override required request fields')
        body.update({'model': self._config['model'], 'messages': messages, 'max_tokens': max_tokens,
                     'temperature': 0, 'stream': True})
        if self._config.get('seed') is not None:
            body['seed'] = self._config['seed']
        if self._config.get('stream_options') is not None:
            body['stream_options'] = self._config['stream_options']
        result['raw_safe_metadata']['request_policy'] = _safe_request_policy(body)
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'X-Request-ID': request_id}
        if self._secret:
            headers['Authorization'] = 'Bearer ' + self._secret
        endpoint = self._endpoint if self._endpoint.endswith('/chat/completions') else self._endpoint + '/chat/completions'
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers, method='POST')
        result['raw_safe_metadata']['timeout_enforcement'] = 'socket_timeout_and_checked_between_events'
        result['raw_safe_metadata']['generation_temperature'] = 0
        text, reasoning, models, done = [], [], set(), False
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for line in response:
                if time.monotonic() - started > timeout_s:
                    raise TimeoutError
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    result['status'], result['error'] = 'provider_error', {'code': 'provider_error'}
                    return
                if isinstance(event.get('model'), str):
                    models.add(event['model'])
                for choice in event.get('choices') or []:
                    if choice.get('index', 0) != 0:
                        continue
                    delta = choice.get('delta') or {}
                    segment = delta.get('content')
                    if isinstance(segment, str) and segment:
                        text.append(segment)
                        result['raw_text'] = ''.join(text)
                        result['text'], tagged = _separate_thinking(result['raw_text'])
                        result['reasoning'] = ''.join(reasoning) + tagged
                        if result['first_visible_s'] is None and _visible_so_far(result['raw_text']):
                            result['first_visible_s'] = time.monotonic() - started
                    for key in ('reasoning', 'reasoning_content'):
                        if isinstance(delta.get(key), str):
                            reasoning.append(delta[key])
                            result['reasoning'] = ''.join(reasoning) + _separate_thinking(result['raw_text'])[1]
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                if isinstance(event.get('usage'), dict):
                    usage = event['usage']
                    result['usage'] = {'input_tokens': _number(usage.get('prompt_tokens')),
                                       'output_tokens': _number(usage.get('completion_tokens')),
                                       'total_tokens': _number(usage.get('total_tokens')),
                                       'cost_usd': _number(usage.get('cost'))}
        result['raw_text'] = ''.join(text)
        result['text'], tagged = _separate_thinking(result['raw_text'])
        result['reasoning'] = ''.join(reasoning) + tagged
        result['raw_safe_metadata']['stream_done'] = done
        result['resolved_model'] = next(iter(models)) if len(models) == 1 else None
        if len(models) > 1:
            result['status'], result['error'] = 'identity_mismatch', {'code': 'multiple_resolved_models'}
        else:
            result['status'] = 'ok' if result['finish_reason'] and done else 'incomplete_stream'

    def _generate_native(self, result, messages, max_tokens, timeout_s, started):
        if len(messages) != 1 or messages[0]['role'] != 'user':
            result['status'], result['error'] = 'unsupported_messages', {'code': 'native_requires_one_user_message'}
            return
        if self._native is None or self._native_timeout != timeout_s:
            if self._native is not None:
                self.close()
            self._native = _new_native(self._config, timeout_s)
            self._native_timeout = timeout_s
        final = {}
        def emit(event, **data):
            if event == 'first_token' and result['first_visible_s'] is None:
                result['first_visible_s'] = _number(data.get('elapsed_s'))
                if result['first_visible_s'] is None:
                    result['first_visible_s'] = time.monotonic() - started
            if event == 'delta' and isinstance(data.get('text'), str):
                result['raw_text'] += data['text']
                result['text'], result['reasoning'] = _separate_thinking(result['raw_text'])
            if event == 'result':
                final.update(data)
        self._native.generate(messages[0]['content'], max_tokens, emit)
        result['text'] = final.get('text', '')
        result['raw_text'] = result['text']
        result['resolved_model'] = final.get('resolved_model') or final.get('model')
        result['finish_reason'] = final.get('finish_reason')
        result['usage']['input_tokens'] = _number(final.get('input_tokens_total', final.get('input_tokens')))
        result['usage']['output_tokens'] = _number(final.get('output_tokens'))
        result['usage']['cost_usd'] = _number(final.get('estimated_cost_usd'))
        if result['usage']['input_tokens'] is not None and result['usage']['output_tokens'] is not None:
            result['usage']['total_tokens'] = result['usage']['input_tokens'] + result['usage']['output_tokens']
        result['raw_safe_metadata']['cost_is_estimate'] = final.get('cost_is_estimate')
        result['raw_safe_metadata']['timeout_enforcement'] = 'native_client'
        result['status'] = 'ok' if isinstance(result['text'], str) and result['text'] else 'incomplete_stream'

    def _opencode_config(self):
        return {'model': _OLLAMA_MODEL, 'update': 'disable',
                'providers': {'ollama': {'settings': {'baseURL': 'http://127.0.0.1:11434/v1'}}},
                'policies': [{'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
                             {'action': 'provider.use', 'resource': 'ollama', 'effect': 'allow'},
                             {'action': 'tool.use', 'resource': '*', 'effect': 'deny'},
                             {'action': 'mcp.use', 'resource': '*', 'effect': 'deny'},
                             {'action': 'skill.use', 'resource': '*', 'effect': 'deny'}]}

    def audit_opencode(self):
        """Create a private configuration and check OpenCode honors its XDG paths."""
        if self._config['type'] != 'opencode':
            raise ValueError('audit is only available for opencode')
        if self._opencode_temp is None:
            self._opencode_temp = tempfile.TemporaryDirectory(prefix='ticklm-opencode-')
            root = Path(self._opencode_temp.name)
            for name in ('config/opencode', 'data', 'cache', 'state', 'work'):
                (root / name).mkdir(parents=True, exist_ok=True)
            (root / 'config/opencode/opencode.json').write_text(json.dumps(self._opencode_config()))
        root = Path(self._opencode_temp.name).resolve()
        env = {name: os.environ[name] for name in ('PATH', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL')
               if name in os.environ}
        env.update({f'XDG_{key.upper()}_HOME': str(root / key) for key in ('config', 'data', 'cache', 'state')})
        code, stdout, _ = _run_opencode(['opencode', 'debug', 'paths'], cwd=root / 'work', env=env, timeout=10)
        if code:
            raise RuntimeError('opencode path audit failed')
        paths = dict(line.split(None, 1) for line in stdout.splitlines() if len(line.split(None, 1)) == 2)
        for key in ('config', 'data', 'cache', 'state'):
            if key not in paths or not Path(paths[key]).resolve().is_relative_to(root):
                raise RuntimeError('opencode did not honor isolated XDG paths')
        managed = ['/Library/Application Support/opencode',
                   '/Library/Managed Preferences/ai.opencode.managed.plist',
                   f'/Library/Managed Preferences/{os.environ.get("USER", "")}/ai.opencode.managed.plist']
        _assert_no_ambient_opencode_config(root / 'work', managed)
        return {'xdg_paths_inside_isolation': True, 'ambient_project_and_managed_config_absent': True,
                'config_sha256': hashlib.sha256(
            json.dumps(self._opencode_config(), sort_keys=True).encode()).hexdigest(),
                'provider': 'ollama', 'endpoint': 'http://127.0.0.1:11434/v1',
                'other_providers_deny_policy_configured': True, 'tools_deny_policy_configured': True}

    def _generate_opencode(self, result, messages, timeout_s, started):
        if len(messages) != 1 or messages[0]['role'] != 'user':
            result['status'], result['error'] = 'unsupported_messages', {'code': 'native_requires_one_user_message'}
            return
        audit = self.audit_opencode()
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=min(timeout_s, 5)) as response:
            inventory = json.load(response)
        models = inventory.get('models', [])
        matches = [m for m in models if m.get('name') == 'qwen3:0.6b']
        if len(matches) != 1 or matches[0].get('digest') != _OLLAMA_DIGEST:
            raise ValueError('local Ollama model digest mismatch')
        root = Path(self._opencode_temp.name).resolve()
        env = {name: os.environ[name] for name in ('PATH', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL')
               if name in os.environ}
        env.update({f'XDG_{key.upper()}_HOME': str(root / key) for key in ('config', 'data', 'cache', 'state')})
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError
        code, stdout, _ = _run_opencode(['opencode', 'run', '--standalone', '--format', 'json',
                                         '--model', _OLLAMA_MODEL, '--', messages[0]['content']],
                                        cwd=root / 'work', env=env, timeout=remaining)
        parsed = _parse_opencode_events(stdout)
        result['text'] = result['raw_text'] = parsed['text']
        result['reasoning'] = parsed['reasoning']
        result['usage'] = parsed['usage']
        result['finish_reason'] = parsed['finish_reason']
        result['raw_safe_metadata'].update({'opencode_config_audit': audit, 'ollama_digest': _OLLAMA_DIGEST,
                                            'ollama_inventory_matches': True,
                                            'event_types': parsed['event_types'],
                                            'completion_evidence': ('step_finish_event' if parsed['finish_reason']
                                                                    else 'cli_exit_only_finish_reason_unavailable'),
                                            'retry_policy': 'native_client_internal_unknown',
                                            'timeout_enforcement': 'process_group_timeout',
                                            'identity_validation': 'selected_model_and_inventory_only'})
        result['status'] = 'ok' if code == 0 and result['text'] else 'incomplete_stream'
        if code != 0:
            result['error'] = {'code': 'opencode_exit', 'exit_code': code}

    def close(self):
        if self._native is not None:
            self._native.close()
            self._native = None
        self._model = None
        self._tokenizer = None
        if self._opencode_temp is not None:
            self._opencode_temp.cleanup()
            self._opencode_temp = None


def create_adapter(config):
    return Adapter(config)
