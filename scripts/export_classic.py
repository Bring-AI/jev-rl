"""Create a browser bundle and auditable, compact paper data from measured runs."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from jev_reward.classic.envs import TASKS
from jev_reward.classic.rewards import rule_level
from jev_reward.learning import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jevrl-v1"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/jevrl-v1"))
    parser.add_argument("--site-dir", type=Path, default=Path("src/jev_reward/static/classic"))
    args = parser.parse_args()
    data_dir = args.site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (args.site_dir / "scores").mkdir(exist_ok=True)
    (args.experiment_dir / "results").mkdir(exist_ok=True)
    summaries, entries, provenance, cache_stats = {}, [], {}, {}
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    expected = {(c["task"], c["provider"], c["seed"]) for c in protocol["configs"]}
    observed = set()
    for path in sorted(args.run_dir.glob("*/*/*/run.json")):
        run = json.loads(path.read_text())
        key = run["task"]["slug"], run["provider"], run["config"]["seed"]
        if run["status"] != "completed" or key not in expected:
            raise ValueError(f"Invalid experiment run: {path}")
        observed.add(key)
        task, provider, seed = key
        filename = f"{task}-{provider}-{seed}.json"
        provenance[filename] = {
            "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        browser = copy.deepcopy(run)
        browser["browser_copy"] = (
            "Compact metrics and replay; local run retains every evaluation outcome and full checkpoints."
        )
        for point in browser["checkpoints"]:
            for field in ("returns", "lengths", "successes"):
                point["evaluation"].pop(field, None)
        (data_dir / filename).write_text(
            json.dumps(browser, separators=(",", ":"), allow_nan=False)
        )
        entries.append(
            {"task": task, "provider": provider, "seed": seed, "file": f"data/{filename}"}
        )
        # Compact research record retains all seed-level measurements and training curves.
        research = copy.deepcopy(run)
        for point in research["checkpoints"]:
            point["evaluation"].pop("replay", None)
        research["test"].pop("replay", None)
        save_json(args.experiment_dir / "results" / filename, research)
        summaries.setdefault(task, {}).setdefault(provider, []).append(
            {
                "seed": seed,
                "return": run["test"]["native_return_mean"],
                "success": run["test"]["success_rate"],
                "steps": run["steps"],
                "episodes": len(run["history"]),
                "duration_seconds": run["duration_seconds"],
            }
        )
    if observed != expected:
        raise ValueError(f"Missing experiment runs: {expected - observed}")
    for task in TASKS:
        random_path = args.run_dir / f"random-{task}.json"
        random_data = json.loads(random_path.read_text())
        random_data.pop("replay", None)
        save_json(args.experiment_dir / f"random-{task}.json", random_data)
        for provider, runs in list(summaries[task].items()):
            summaries[task][provider] = {
                "runs": runs,
                "return_mean": float(np.mean([r["return"] for r in runs])),
                "return_sd": float(np.std([r["return"] for r in runs], ddof=1)),
                "success_mean": float(np.mean([r["success"] for r in runs])),
                "success_sd": float(np.std([r["success"] for r in runs], ddof=1)),
            }
        summaries[task]["random"] = {
            "return_mean": random_data["native_return_mean"],
            "success_mean": random_data["success_rate"],
        }
        path = args.experiment_dir / "judgments" / f"{task}.json"
        cache = json.loads(path.read_text())
        differences, agreement = [], []
        for entry in cache["entries"].values():
            level = rule_level(entry["context"])
            expected_reward = cache["reward_values"][level]
            verdict = entry["verdict"]
            differences.append(abs(expected_reward - verdict["reward"]))
            agreement.append(
                max(verdict["probabilities"], key=verdict["probabilities"].get) == str(level)
            )
        cache_stats[task] = {
            "answers": len(differences),
            "reward_mae": float(np.mean(differences)),
            "reward_max_error": float(np.max(differences)),
            "argmax_rule_agreement": float(np.mean(agreement)),
            "usage": cache["source_usage"],
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "served_models": sorted({v["verdict"]["model"] for v in cache["entries"].values()}),
        }
        shutil.copyfile(path, args.site_dir / "scores" / f"{task}.json")
    summary = {
        "tasks": summaries,
        "cache": cache_stats,
        "total_training_steps": sum(c["total_steps"] for c in protocol["configs"]),
        "training_runs": len(entries),
        "training_seeds": sorted({c["seed"] for c in protocol["configs"]}),
        "note": "SD is between 3 trained policies; 100 test episodes per policy, not 300 independent training seeds.",
        "cost_note": "Recorded cache-construction usage; excludes exploratory diagnostic requests and earlier Key Quest runs.",
        "random_policy_rng": "NumPy SeedSequence([environment_seed, 0x4A455652]); independent action stream.",
    }
    save_json(args.experiment_dir / "summary.json", summary)
    save_json(args.experiment_dir / "provenance.json", provenance)
    shutil.copyfile(args.run_dir / "protocol.json", args.experiment_dir / "protocol.json")
    save_json(
        args.site_dir / "manifest.json",
        {"tasks": [t.spec() for t in TASKS.values()], "runs": entries, "summary": summary},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
