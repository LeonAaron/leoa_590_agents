"""Exercise discovery of additional agent packages and invalid registrations."""

import asyncio
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from agents590 import registry
from agents590.contracts import RunContext


def test_an_additional_registered_name_uses_the_same_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = [
        EntryPoint(
            name="custom-agent",
            value="agents590.krisj_echo:EchoAgent",
            group=registry.ENTRY_POINT_GROUP,
        )
    ]
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: points)
    assert registry.agent_names() == ["custom-agent"]
    first = registry.create_agent("custom-agent")
    second = registry.create_agent("custom-agent")
    assert first is not second
    assert asyncio.run(first.run("plugin instruction", RunContext(cwd=tmp_path))) == (
        "plugin instruction"
    )


def test_duplicate_agent_names_fail_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    point = EntryPoint(
        name="duplicate", value="agents590.krisj_echo:EchoAgent", group=registry.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [point, point])
    with pytest.raises(ValueError, match="Multiple installed packages"):
        registry.create_agent("duplicate")


def test_a_factory_must_return_an_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    point = EntryPoint(name="invalid", value="builtins:object", group=registry.ENTRY_POINT_GROUP)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [point])
    with pytest.raises(TypeError, match="async run"):
        registry.create_agent("invalid")
