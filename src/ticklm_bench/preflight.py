"""Run one reproducible development-only local or signed-in client preflight."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .adapters import create_adapter
from .tasks import build_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', choices=['mlx', 'claude_code', 'codex'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-id')
    parser.add_argument('--revision')
    parser.add_argument('--quantization')
    parser.add_argument('--effort', default='high')
    parser.add_argument('--profile', default='startup')
    parser.add_argument('--allowed-resolved-model', action='append', default=[])
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1103)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'refusing existing output: {output}')
    config = {'type': args.type, 'model': args.model, 'effort': args.effort,
              'profile': args.profile, 'allowed_resolved_models': args.allowed_resolved_model}
    if args.model_id:
        config['model_id'] = args.model_id
    if args.revision:
        config['revision'] = args.revision
    if args.quantization:
        config['quantization'] = args.quantization
    if args.type == 'mlx':
        config['thinking'] = False
    case = build_suite('dev', pairs_per_family=1, seed=args.seed)[0]
    adapter = create_adapter(config)
    try:
        result = adapter.generate([{'role': 'user', 'content': case['prompt']}],
                                  max_tokens=args.max_tokens, timeout_s=args.timeout,
                                  request_id=f'preflight-{args.type}-{args.seed}')
        payload = {'scope': 'One development case; diagnostic preflight, not a benchmark score.',
                   'case': case, 'adapter_metadata': adapter.metadata, 'result': result,
                   'exact_letter_matches': result['status'] == 'ok' and result['text'].strip() == case['answer']}
        fd, temp = tempfile.mkstemp(dir=output.parent, prefix='.tmp-preflight-')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write('\n')
            os.link(temp, output)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    finally:
        adapter.close()


if __name__ == '__main__':
    main()
