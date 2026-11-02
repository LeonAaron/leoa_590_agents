"""Tests for the shell agent command execution tool."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from agents590.contracts import RunContext
from agents590.leoa_ShellAgent import list_skills, run_shell


def test_run_shell_uses_context_directory_and_returns_output_and_exit_code(tmp_path: Path) -> None:
    context = RunContext(cwd=tmp_path)
    python_code = (
        "import os, sys; print(os.getcwd()); print('stderr output', file=sys.stderr); sys.exit(7)"
    )
    arguments = [sys.executable, "-c", python_code]
    command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)

    output = run_shell(command, context)

    assert output == f"{tmp_path}\nstderr output\n\n(exit code 7)"


def test_list_skills_includes_frontmatter_but_not_body(tmp_path: Path) -> None:
    skill = tmp_path / ".agents" / "skills" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\ndescription: useful skill\n---\nsecret skill body")

    listing = list_skills(RunContext(cwd=tmp_path))

    assert str(skill) in listing
    assert "description: useful skill" in listing
    assert "secret skill body" not in listing
