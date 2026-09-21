"""Collect explicit JEV judgments or run a frozen, matched multi-seed protocol."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from jev_reward.classic.envs import TASKS
from jev_reward.classic.rewards import ScoreCache, all_contexts, digest
from jev_reward.classic.training import ClassicConfig, brief, evaluate_policy, train_classic
from jev_reward.learning import save_json


def worker(config, output):
    result = train_classic(ClassicConfig(**config), output)
    return brief(result)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "run"])
    parser.add_argument("--output", type=Path, default=Path("runs/classic-benchmark"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/jevrl-v1/judgments"))
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["native", "rules", "jev"],
        choices=["native", "rules", "jev"],
    )
    parser.add_argument("--games", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 42])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override per-task protocol budgets for separate pilot runs",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=100)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "collect":
        for task in args.games:
            path = args.cache_dir / f"{task}.json"
            cache = ScoreCache(task, "jev", path, max_calls=160)
            try:
                contexts = all_contexts(task)
                for index, context in enumerate(contexts, start=1):
                    cache.score(context)
                    if index % 10 == 0 or index == len(contexts):
                        print(
                            json.dumps({"game": task, "answers": index, "total": len(contexts)}),
                            flush=True,
                        )
                cache.save(path)
            finally:
                cache.save(path)
                cache.close()
            print(json.dumps(cache.stats()), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    configs = []
    for task in args.games:
        for provider in args.providers:
            for seed in args.seeds:
                cache_path = (
                    str((args.cache_dir / f"{task}.json").resolve()) if provider == "jev" else None
                )
                if cache_path:
                    # Reject absent/incomplete caches before launching any training jobs.
                    cache = ScoreCache(task, "jev", cache_path, frozen=True)
                    for context in all_contexts(task):
                        cache.score(context)
                config = ClassicConfig(
                    task=task,
                    provider=provider,
                    seed=seed,
                    total_steps=args.steps or TASKS[task].steps,
                    checkpoint_steps=min(10000, args.steps or TASKS[task].steps),
                    eval_episodes=args.eval_episodes,
                    test_episodes=args.test_episodes,
                    cache_path=cache_path,
                    frozen_cache=provider == "jev",
                )
                configs.append(asdict(config))
    protocol_path = args.output / "protocol.json"
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configs": configs,
        "development_seed_start": 10000,
        "test_seed_start": 100000,
        "selection": "Final checkpoint; no best-checkpoint selection",
        "cache_hashes": {
            task: digest(json.loads((args.cache_dir / f"{task}.json").read_text()))
            for task in args.games
            if "jev" in args.providers
        },
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous["configs"] != configs or previous["cache_hashes"] != protocol["cache_hashes"]:
            raise ValueError("Protocol differs from existing output; use a new directory")
    else:
        save_json(protocol_path, protocol)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = []
        for config in configs:
            output = args.output / config["task"] / config["provider"] / str(config["seed"])
            if (output / "run.json").exists():
                result = json.loads((output / "run.json").read_text())
                if result["status"] != "completed" or result["config"] != config:
                    raise ValueError(f"Incomplete or mismatched existing run: {output}")
                print(json.dumps({"existing": brief(result)}), flush=True)
                continue
            futures.append(pool.submit(worker, config, output))
        for future in as_completed(futures):
            summary = future.result()
            print(json.dumps(summary), flush=True)
            if summary["status"] != "completed":
                raise RuntimeError("A benchmark run did not complete")
    for task in args.games:
        random_result = evaluate_policy(None, task, args.test_episodes, 100000)
        save_json(args.output / f"random-{task}.json", random_result)
    print(f"All results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
