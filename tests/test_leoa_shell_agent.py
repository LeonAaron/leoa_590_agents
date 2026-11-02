"""Tests for the shell agent's command execution tool."""

from pathlib import Path

from agents590.contracts import RunContext
from agents590.leoa_ShellAgent import run_shell


def test_run_shell_uses_context_directory_and_returns_output_and_exit_code(tmp_path: Path) -> None:
    context = RunContext(cwd=tmp_path)
    command = "pwd; printf 'stderr output\\n' >&2; exit 7"

    output = run_shell(command, context)

    assert output == f"{tmp_path}\nstderr output\n\n(exit code 7)"
