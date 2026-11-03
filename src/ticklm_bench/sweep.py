"""Run a prespecified local model list sequentially, retaining failed preflights."""

import argparse
import json
import uuid
from pathlib import Path

from .adapters import create_adapter
from .runner import INSTRUCTION, read_rows, run, write_json
from .tasks import build_suite


def sweep(plan_file):
    plan = json.loads(Path(plan_file).read_text())
    for entry in plan["runs"]:
        config_path = Path(entry["config"])
        config = json.loads(config_path.read_text())
        if config.get("paid"):
            raise ValueError("this sequential sweep only supports local models and existing sign-ins")
        output = Path(entry["output"])
        if output.exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["status"] != "complete":
                raise ValueError("existing incomplete output requires explicit runner resume")
            print(f"Already complete: {config['system_id']}", flush=True)
            continue
        preflight = Path(entry["preflight"])
        if not preflight.exists():
            case = build_suite("dev", 1, 1103)[0]
            adapter = create_adapter(config)
            try:
                result = adapter.generate([{"role": "user", "content": INSTRUCTION + case["prompt"]}],
                                          max_tokens=config.get("max_tokens", 256),
                                          timeout_s=config.get("timeout_s", 120),
                                          request_id="preflight-" + str(uuid.uuid4()))
            finally:
                adapter.close()
            preflight.parent.mkdir(parents=True, exist_ok=True)
            write_json(preflight, {"case": case, "config": config, "result": result})
        else:
            result = json.loads(preflight.read_text())["result"]
        if result["status"] != "ok":
            print(f"Excluded after failed transport/runtime preflight: {config['system_id']}", flush=True)
            continue
        print(f"Starting: {config['system_id']} ({len(read_rows(entry['dataset']))} cases)", flush=True)
        run(entry["dataset"], config_path, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    sweep(args.plan)


if __name__ == "__main__":
    main()
