"""Offline tests for the model-provider factory (DeepSeek integration).

These tests never make network requests. The real ``strands.models.openai``
and ``strands.models.bedrock`` modules are imported (proving the declared
``openai`` dependency resolves); only the provider constructors and the process
environment are monkeypatched.
"""

from __future__ import annotations

import pytest

from src.agent.model_factory import build_model


class _RecordingOpenAIModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordingBedrockModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MODEL_PROVIDER", "MODEL_ID", "AWS_REGION", "API_KEY"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# none
# --------------------------------------------------------------------------- #
def test_none_provider_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    assert build_model() is None
    monkeypatch.setenv("MODEL_PROVIDER", "none")
    assert build_model() is None
    monkeypatch.setenv("MODEL_PROVIDER", "")
    assert build_model() is None


# --------------------------------------------------------------------------- #
# bedrock (unchanged behavior)
# --------------------------------------------------------------------------- #
def test_bedrock_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock")
    monkeypatch.setenv("MODEL_ID", "some-model")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr("strands.models.bedrock.BedrockModel", _RecordingBedrockModel)

    model = build_model()
    assert isinstance(model, _RecordingBedrockModel)
    assert model.kwargs == {"model_id": "some-model", "region_name": "us-east-1"}


def test_bedrock_missing_vars_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock")
    with pytest.raises(ValueError, match="MODEL_ID and AWS_REGION"):
        build_model()
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(ValueError, match="MODEL_ID and AWS_REGION"):
        build_model()


# --------------------------------------------------------------------------- #
# deepseek
# --------------------------------------------------------------------------- #
def test_deepseek_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setattr("strands.models.openai.OpenAIModel", _RecordingOpenAIModel)

    model = build_model()
    assert isinstance(model, _RecordingOpenAIModel)
    assert model.kwargs["client_args"] == {
        "base_url": "https://api.deepseek.com",
        "api_key": "test-api-key",
    }
    assert model.kwargs["model_id"] == "deepseek-flash"
    # The DeepSeek thinking/extra_body parameter is not supported by the API and
    # must never be sent (it caused HTTP 400 Bad Request).
    assert "params" not in model.kwargs


def test_deepseek_missing_api_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    with pytest.raises(ValueError, match="API_KEY"):
        build_model()
    monkeypatch.setenv("API_KEY", "   ")
    with pytest.raises(ValueError, match="API_KEY"):
        build_model()


def test_deepseek_missing_model_id_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("API_KEY", "test-api-key")
    with pytest.raises(ValueError, match="MODEL_ID"):
        build_model()
    monkeypatch.setenv("MODEL_ID", "   ")
    with pytest.raises(ValueError, match="MODEL_ID"):
        build_model()


# --------------------------------------------------------------------------- #
# unknown provider
# --------------------------------------------------------------------------- #
def test_unknown_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="none, bedrock, deepseek"):
        build_model()


# --------------------------------------------------------------------------- #
# no secret leakage
# --------------------------------------------------------------------------- #
def test_no_secret_leak_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("API_KEY", "sk-sentinel-1234567890")
    # MODEL_ID is missing -> the error must not contain the secret value.
    with pytest.raises(ValueError) as excinfo:
        build_model()
    message = str(excinfo.value)
    assert "sk-sentinel-1234567890" not in message
    assert "MODEL_ID" in message
