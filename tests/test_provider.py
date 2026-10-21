"""Verify UNC routing and authentication without contacting any model service."""

import asyncio
import json
import os
from functools import partial

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agents590 import provider, smoke
from agents590.provider import UNC_ENDPOINT, UNCOpenAISettings, deployment_name


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(name)


@pytest.fixture
def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("OPENAI_MODEL", "course-deployment")


def test_legacy_configuration_defaults_to_unc(configured_environment: None) -> None:
    settings = UNCOpenAISettings.from_env()
    assert settings.endpoint == UNC_ENDPOINT
    assert settings.model == "course-deployment"
    assert settings.api_key == "synthetic-unc-key"
    assert settings.api_key not in repr(settings)


def test_azure_aliases_take_precedence(
    configured_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", " azure-subscription-key ")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", " azure-deployment ")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", f" {UNC_ENDPOINT}/ ")
    settings = UNCOpenAISettings.from_env()
    assert settings.api_key == "azure-subscription-key"
    assert settings.model == "azure-deployment"
    assert settings.endpoint == f"{UNC_ENDPOINT}/"


def test_azure_only_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "synthetic-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "enabled-deployment")
    assert UNCOpenAISettings.from_env().model == "enabled-deployment"


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "OPENAI_MODEL"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_credentials_or_model_fail_before_client_creation(
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    def unexpected_client(**kwargs: object) -> None:
        pytest.fail("Invalid configuration must not construct an API client.")

    monkeypatch.setattr(provider, "OpenAI", unexpected_client)
    with pytest.raises(SystemExit, match=name) as error:
        smoke.main()
    assert "synthetic-unc-key" not in str(error.value)


@pytest.mark.parametrize(
    "alias", ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT"]
)
def test_explicit_blank_azure_alias_does_not_fall_back(
    configured_environment: None, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    monkeypatch.setenv(alias, " ")
    with pytest.raises(ValueError):
        UNCOpenAISettings.from_env()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("course-deployment", "course-deployment"),
        (" openai/course-deployment ", "course-deployment"),
    ],
)
def test_deployment_mapping(model: str, expected: str) -> None:
    assert deployment_name(model) == expected


@pytest.mark.parametrize("model", ["", " ", "openai/", "other/model", "openai/a/b", "a b"])
def test_invalid_deployment_mapping(model: str) -> None:
    with pytest.raises(ValueError, match="UNC deployment name"):
        deployment_name(model)


def test_harbor_model_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unc-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "smoke-deployment")
    assert UNCOpenAISettings.from_env(model="openai/harbor-deployment").model == "harbor-deployment"


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://azureaiapi.cloud.unc.edu",
        "https://api.openai.com",
        "https://API.OPENAI.COM./",
        "https://openai.com",
        "https://azureaiapi.cloud.unc.edu/openai/v1",
        "https://azureaiapi.cloud.unc.edu?api-version=preview",
        "https://azureaiapi.cloud.unc.edu#fragment",
        "https://user:synthetic-secret@azureaiapi.cloud.unc.edu",
        "https://azureaiapi.cloud.unc.edu:8080",
        "https://azureaiapi.cloud.unc.edu:invalid",
        "https://[invalid",
        "https:///missing-host",
    ],
)
def test_invalid_or_public_endpoint_fails_without_exposing_values(
    configured_environment: None, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("OPENAI_ENDPOINT", endpoint)
    with pytest.raises(ValueError) as error:
        UNCOpenAISettings.from_env()
    assert "synthetic-secret" not in str(error.value)


def _response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_mock",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_mock",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "UNC Azure connection OK",
                            "annotations": [],
                        }
                    ],
                }
            ],
        },
    )


@pytest.mark.parametrize("endpoint_suffix", ["", "/"])
def test_smoke_routes_real_sdk_request_to_unc(
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    endpoint_suffix: str,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    monkeypatch.setenv("OPENAI_ENDPOINT", UNC_ENDPOINT + endpoint_suffix)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_VERSION", "ignored-for-v1")
    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(provider, "OpenAI", partial(OpenAI, http_client=http_client))

    smoke.main()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{UNC_ENDPOINT}/openai/v1/responses"
    assert request.headers["authorization"] == "Bearer synthetic-unc-key"
    assert "api-key" not in request.headers
    assert json.loads(request.content) == {
        "model": "course-deployment",
        "input": "Reply with exactly: UNC Azure connection OK",
        "max_output_tokens": 256,
        "store": False,
        "background": False,
    }
    assert capsys.readouterr().out == "UNC Azure connection OK\n"
    assert http_client.is_closed


def test_async_client_uses_same_unc_route_and_azure_auth(
    configured_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "synthetic-azure-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
    settings = UNCOpenAISettings.from_env(model="openai/harbor-deployment")

    async def run() -> None:
        async with settings.create_async_client() as client:
            assert client.max_retries == 0
            assert client.timeout == 30.0
            await client.responses.create(
                model=settings.model, input="test", store=False, background=False
            )

    asyncio.run(run())
    assert len(requests) == 1
    assert str(requests[0].url) == f"{UNC_ENDPOINT}/openai/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer synthetic-azure-key"
    assert json.loads(requests[0].content)["model"] == "harbor-deployment"
    assert http_client.is_closed
