"""AI Agent that completes coding tasks by running shell commands."""

import json
import os
import subprocess
import sys
from typing import Any

from openai.types.responses import ResponseFunctionToolCall

from agents590.contracts import RunContext
from agents590.provider import UNCOpenAISettings

# No __pycache__ folders in the task directory to fix an evaluation failure
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  


# System Prompt
SYSTEM = "You are a coding agent in a Linux shell. Make the requested changes and verify them."


# The only tool that the model can use is running a shell command
TOOL: Any = {
    "type": "function",
    "name": "shell",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}

# Allows AI model to maintain its reasoning between turns
INCLUDE: list[Any] = ["reasoning.encrypted_content"]  


def run_shell(command: str, context: RunContext) -> str:
    """Run a shell command and return the output and exit code."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=context.cwd, capture_output=True, text=True, timeout=60
        )
    # Timeout for command execution
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."

    # Return the output and exit code
    return f"{(result.stdout + result.stderr)[-10000:]}\n(exit code {result.returncode})"


class leoa_ShellAgent:
    async def run(self, instruction: str, context: RunContext) -> str:
        settings = UNCOpenAISettings.from_env(model=context.model)

        # Maintain chat history, starting with system prompt and instruction
        history: list[Any] = [{"role": "user", "content": f"{SYSTEM}\n\n{instruction}"}]

        async with settings.create_async_client() as client:
            # Give the AI model up to 6 turns to complete the task successfully
            for _ in range(6):

                # response from AI model
                response = await client.responses.create(
                    model=settings.model, input=history, tools=[TOOL], include=INCLUDE, store=False
                )
                # Add response to history logs
                history.extend(response.output)

                # Track shell commands requested by the AI model
                calls = [c for c in response.output if isinstance(c, ResponseFunctionToolCall)]
                commands: list[list[str]] = []

                # Run all commands
                for call in calls:
                    command = json.loads(call.arguments)["command"]
                    output = run_shell(command, context)

                    # Store command history for tracing
                    commands.append([command, output])

                    # Update history logs as addional context for AI model
                    history.append(
                        {"type": "function_call_output", "call_id": call.call_id, "output": output}
                    )

                # Enable the Harbor UI to show token useage and command history
                if usage := response.usage:
                    cached = usage.input_tokens_details.cached_tokens

                    # Break up token useage
                    tokens = [usage.input_tokens, cached, usage.output_tokens]

                    # Create comprehensive trace and display
                    trace = {"tokens": tokens, "text": response.output_text, "commands": commands}
                    print("TRACE " + json.dumps(trace), file=sys.stderr)

                # No commands means the model finished
                if not calls:
                    return response.output_text

        return "FAILURE: Ran out of turns."


r"""
Important commands:

cd "C:\Users\Leon Aaron\Documents\Semester 5\AI\agent_evaluation_harness\agent_evals"

make agent AGENT=leoa_ShellAgent AGENTS_PROJECT=../leoa_590_agents ENV_FILE=../leoa_590_agents/.env

In separate terminal:

uv run --locked harbor view jobs

"""
