"""Tests must never inherit developer credentials or make paid calls."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_judge_credentials(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("OPENROUTER_", "TYPESAFE_")) or name == "JEV_GATEWAY":
            monkeypatch.delenv(name, raising=False)
