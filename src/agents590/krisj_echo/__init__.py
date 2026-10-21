"""A deterministic agent for exercising the CLI and evaluation integrations."""

from agents590.contracts import RunContext


class EchoAgent:
    """Return the instruction without executing tools or making model requests."""

    async def run(self, instruction: str, context: RunContext) -> str:
        return instruction
