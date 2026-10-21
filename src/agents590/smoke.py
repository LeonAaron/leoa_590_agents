"""An explicit, small Responses API call to check UNC Azure configuration."""

from agents590.provider import UNCOpenAISettings


def main() -> None:
    """Run with: uv run --env-file .env openai-smoke."""
    try:
        settings = UNCOpenAISettings.from_env()
    except ValueError as error:
        raise SystemExit(str(error)) from None

    with settings.create_client() as client:
        response = client.responses.create(
            model=settings.model,
            input="Reply with exactly: UNC Azure connection OK",
            max_output_tokens=256,
            store=False,
            background=False,
        )
    print(response.output_text)


if __name__ == "__main__":
    main()
