"""Preserve per-run model selection after removing the Harbor dependency."""

import asyncio
import os
from pathlib import Path

import pytest

from agents590.contracts import RunContext
from agents590.krisj_starter import StarterAgent


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")


@pytest.mark.parametrize(
    ("run_model", "azure_default", "legacy_default", "expected"),
    [
        ("openai/run-deployment", "azure-default", "legacy-default", "run-deployment"),
        ("bare-run-deployment", None, None, "bare-run-deployment"),
        ("openai/run-deployment", "", "", "run-deployment"),
        (None, "azure-default", "legacy-default", "azure-default"),
        (None, None, "legacy-default", "legacy-default"),
    ],
)
def test_per_run_model_precedes_environment_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_model: str | None,
    azure_default: str | None,
    legacy_default: str | None,
    expected: str,
) -> None:
    for name, value in [
        ("AZURE_OPENAI_DEPLOYMENT", azure_default),
        ("OPENAI_MODEL", legacy_default),
    ]:
        if value is not None:
            monkeypatch.setenv(name, value)
    context = RunContext(cwd=tmp_path, model=run_model)
    assert StarterAgent().provider_settings(context).model == expected


@pytest.mark.parametrize("model", ["", " ", "openai/", "anthropic/model", "azure/model"])
def test_invalid_explicit_model_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "working-default")
    with pytest.raises(ValueError):
        StarterAgent().provider_settings(RunContext(cwd=tmp_path, model=model))


def test_missing_model_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        StarterAgent().provider_settings(RunContext(cwd=tmp_path))


def test_starter_loop_is_still_a_scaffold(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        asyncio.run(StarterAgent().run("test", RunContext(cwd=tmp_path)))
