"""Discover agent factories through standard Python package entry points."""

from collections.abc import Callable
from importlib.metadata import entry_points
from inspect import iscoroutinefunction
from typing import cast

from agents590.contracts import Agent

ENTRY_POINT_GROUP = "agents590.agents"


def agent_names() -> list[str]:
    """List installed agents without importing their implementations."""
    return sorted({point.name for point in entry_points(group=ENTRY_POINT_GROUP)})


def create_agent(name: str) -> Agent:
    """Create a fresh agent from its registered, no-argument factory."""
    matches = [point for point in entry_points(group=ENTRY_POINT_GROUP) if point.name == name]
    if not matches:
        raise ValueError(f"Unknown agent {name!r}. Run agents590 --list to see available agents.")
    if len(matches) != 1:
        raise ValueError(f"Multiple installed packages registered agent {name!r}.")
    factory = cast(Callable[[], object], matches[0].load())
    agent = factory()
    if not isinstance(agent, Agent) or not iscoroutinefunction(agent.run):
        raise TypeError(f"Agent {name!r} must implement async run(instruction, context).")
    return agent
