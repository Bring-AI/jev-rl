# Contributing

Install with `uv sync --python 3.12`, then run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run jev-arcade serve
# In another terminal:
uv run playwright install chromium
uv run python scripts/browser_smoke.py
```

Keep game facts separate from numerical rewards. Add a regression test for changes
to the Bellman update, terminal behavior, reward parsing, caching or API budget.
Never publish synthetic/demo data as JEV results. Keep `.env`, credentials and
private game context out of commits. Real API tests are opt-in and consume quota.

Useful extensions: other games, another judge, neural policies, repeated seeds,
reward-hacking diagnostics. Discuss substantial changes in an issue first.
