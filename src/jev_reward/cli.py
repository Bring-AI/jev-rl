import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .judges import JudgeError, make_judge
from .learning import TrainConfig, train


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="JEV Reward Arcade")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the live training dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--output", default="runs")
    training = commands.add_parser("train", help="Train from the terminal and save artifacts")
    training.add_argument("--provider", choices=["demo", "jev"], default="demo")
    training.add_argument("--episodes", type=int, default=350)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--max-calls", type=int, default=300)
    training.add_argument("--checkpoint-every", type=int, default=25)
    training.add_argument("--output", type=Path, default=Path("runs/cli"))
    classic = commands.add_parser("classic", help="Train a standard Gymnasium game with DQN")
    classic.add_argument(
        "--game", choices=["cartpole", "mountaincar", "acrobot", "frozenlake"], default="cartpole"
    )
    classic.add_argument("--provider", choices=["native", "rules", "jev"], default="rules")
    classic.add_argument("--steps", type=int, default=None)
    classic.add_argument("--seed", type=int, default=7)
    classic.add_argument("--checkpoint-steps", type=int, default=10000)
    classic.add_argument("--eval-episodes", type=int, default=20)
    classic.add_argument("--test-episodes", type=int, default=100)
    classic.add_argument("--cache", default=None)
    classic.add_argument("--frozen-cache", action="store_true")
    classic.add_argument("--max-calls", type=int, default=2000)
    classic.add_argument("--output", type=Path, default=Path("runs/classic"))
    args = parser.parse_args()
    if args.command == "classic":
        try:
            from .classic.envs import TASKS
            from .classic.training import ClassicConfig, brief, train_classic
        except ImportError:
            parser.exit(1, "Classic games require: uv sync --extra classic\n")
        try:
            config = ClassicConfig(
                task=args.game,
                provider=args.provider,
                seed=args.seed,
                total_steps=args.steps or TASKS[args.game].steps,
                checkpoint_steps=args.checkpoint_steps,
                eval_episodes=args.eval_episodes,
                test_episodes=args.test_episodes,
                cache_path=args.cache,
                frozen_cache=args.frozen_cache,
                max_calls=args.max_calls,
            )
            result = train_classic(config, args.output)
        except (JudgeError, ValueError) as exc:
            parser.exit(1, str(exc) + "\n")
        print(json.dumps(brief(result), indent=2))
        print(f"Artifacts: {args.output.resolve()}")
        if result["status"] != "completed":
            raise SystemExit(130 if result["status"] == "stopped" else 1)
        return
    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(args.output), host=args.host, port=args.port)
        return
    try:
        config = TrainConfig(
            episodes=args.episodes, seed=args.seed, checkpoint_every=args.checkpoint_every
        )
        if args.max_calls < 1:
            raise ValueError("max-calls must be positive")
        judge = make_judge(args.provider, args.max_calls)
        result = train(config, judge, args.output)
    except (JudgeError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "provider", "before", "after", "judge", "error")
            },
            indent=2,
        )
    )
    print(f"Artifacts: {args.output.resolve()}")
    if result["status"] == "failed":
        raise SystemExit(1)
    if result["status"] == "stopped":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
