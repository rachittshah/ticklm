"""Compare scored temporal cases with counterfactual-pair bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _index(records):
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each case must be a JSON object")
        required = ("id", "pair_id", "family", "prompt", "answer", "choices", "prediction")
        if any(key not in record for key in required):
            raise ValueError("case is missing a required field")
        for key in ("id", "pair_id", "family", "prompt"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"case {key} must be a nonempty string")
        if (not isinstance(record["choices"], dict) or set(record["choices"]) != set("ABCD")
                or not isinstance(record["answer"], str) or not isinstance(record["prediction"], str)
                or record["answer"] not in "ABCD" or record["prediction"] not in "ABCD"
                or len(record["answer"]) != 1 or len(record["prediction"]) != 1):
            raise ValueError("case must have four choices and valid answer and prediction")
        case_id = record["id"]
        if case_id in indexed:
            raise ValueError(f"duplicate case ID: {case_id}")
        indexed[case_id] = record
    if not indexed:
        raise ValueError("input must contain at least one case")
    return indexed


def _metrics(pairs, rng, reps):
    # Each row is a pair; item accuracy is averaged within the pair first.
    measures = {key: [] for key in ("item_accuracy", "pair_both_correct",
                                     "prediction_changed_within_pair")}
    for base, candidate in pairs:
        row = []
        for records in (base, candidate):
            correct = [r["prediction"] == r["answer"] for r in records]
            row.append((sum(correct) / 2, float(all(correct)),
                        float(records[0]["prediction"] != records[1]["prediction"])))
        for index, key in enumerate(measures):
            measures[key].append((row[0][index], row[1][index]))
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in measures.items()}
    sampled = {key: [] for key in measures}
    # Bounded chunks avoid allocating all pair indices for very large requested runs.
    for start in range(0, reps, 1000):
        idx = rng.integers(0, len(pairs), size=(min(1000, reps - start), len(pairs)))
        for key, arr in arrays.items():
            differences = arr[:, 1] - arr[:, 0]
            sampled[key].append(differences[idx].mean(axis=1))
    result = {"pair_count": len(pairs), "item_count": 2 * len(pairs)}
    for key, arr in arrays.items():
        low, high = np.percentile(np.concatenate(sampled[key]), [2.5, 97.5])
        base, candidate = arr.mean(axis=0)
        result[key] = {"base": float(base), "candidate": float(candidate),
                       "difference": float(candidate - base),
                       "difference_ci_95": [float(low), float(high)]}
    return result


def compare_records(base_records, candidate_records, seed=1103, reps=10000):
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    base = _index(base_records)
    candidate = _index(candidate_records)
    if set(base) != set(candidate):
        raise ValueError("case IDs do not match between inputs")
    groups = {}
    for case_id in sorted(base):
        a, b = base[case_id], candidate[case_id]
        for field in ("prompt", "answer", "pair_id", "choices", "family"):
            if a[field] != b[field]:
                raise ValueError(f"case {case_id} has mismatched {field}")
        groups.setdefault(a["pair_id"], []).append((a, b))
    pairs = []
    families = {}
    for pair_id in sorted(groups):
        members = groups[pair_id]
        if len(members) != 2:
            raise ValueError(f"pair {pair_id} must contain exactly two cases")
        if members[0][0]["family"] != members[1][0]["family"]:
            raise ValueError(f"pair {pair_id} has mismatched family")
        pair = ([member[0] for member in members], [member[1] for member in members])
        pairs.append(pair)
        families.setdefault(members[0][0]["family"], []).append(pair)
    rng = np.random.default_rng(seed)
    overall = _metrics(pairs, rng, reps)
    by_family = {family: _metrics(families[family], rng, reps) for family in sorted(families)}
    return {"pair_count": len(pairs), "item_count": 2 * len(pairs), "overall": overall,
            "per_family": by_family, "seed": seed, "bootstrap_reps": reps,
            "bootstrap_unit": "counterfactual pair",
            "interpretation": "Intervals describe synthetic sample variability; they do not establish population generalization."}


def _read_jsonl(path):
    data = Path(path).read_bytes()
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    return records, hashlib.sha256(data).hexdigest()


def _write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-report-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        # link is atomic and fails if another process has already claimed the path.
        os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compare"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.bootstrap_reps <= 0 or args.seed < 0:
        parser.error("bootstrap reps must be positive and seed must be nonnegative")
    base, base_hash = _read_jsonl(args.base)
    candidate, candidate_hash = _read_jsonl(args.candidate)
    result = compare_records(base, candidate, seed=args.seed, reps=args.bootstrap_reps)
    result["source_hashes"] = {"base_sha256": base_hash, "candidate_sha256": candidate_hash,
                               "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["inputs"] = {"base": args.base, "candidate": args.candidate}
    _write_new_json(args.output, result)


if __name__ == "__main__":
    main()
