"""Offline tests for the model-provider factory and exact-combination policy.

These tests never make network requests. The real ``strands.models.openai``
and ``strands.models.bedrock`` modules are imported (proving the declared
``openai`` dependency resolves); only the provider constructors and the process
environment are monkeypatched.

Only obvious fake/sentinel credentials are used — no real credential is read.
"""

from __future__ import annotations

import os

import pytest

from src.agent import model_factory
from src.agent.model_factory import (
    COMBINATION_INDEX,
    PROVIDER_REGISTRY,
    CompatibilityStatus,
    ModelCombination,
    ProviderDescriptor,
    RuntimeProviderConfig,
    build_model,
    build_model_from_runtime_config,
    provider_status_report,
    resolve_combination,
    verified_combinations,
)

_DEV_MODEL_ID = "test-dev-model"
_CONSUMER_MODEL_ID = "test-consumer-model"
_EXPERIMENTAL_MODEL_ID = "test-experimental-model"
_HIDDEN_MODEL_ID = "test-hidden-verified-model"


class _RecordingOpenAIModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordingBedrockModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordingModel:
    """Records exactly what a provider builder was handed."""

    def __init__(self, combination, runtime_fields, credential):
        self.combination = combination
        self.runtime_fields = dict(runtime_fields)
        self.credential = credential
        self.model_id = combination.model_id


def _make_recording_builder(calls: list):
    def _build(combination, runtime_fields, credential):
        calls.append(
            {
                "combination": combination,
                "runtime_fields": dict(runtime_fields),
                "credential": credential,
            }
        )
        return _RecordingModel(combination, runtime_fields, credential)

    return _build


def _fixture_descriptor(
    provider_id: str,
    *,
    model_id: str,
    status: CompatibilityStatus,
    ui_exposed: bool = False,
    credential_scope: tuple[str, ...] = (),
    runtime_env: tuple[str, ...] = (),
    calls: list | None = None,
    build=None,
) -> ProviderDescriptor:
    if build is None:
        build = _make_recording_builder([] if calls is None else calls)
    return ProviderDescriptor(
        provider_id=provider_id,
        display_name=f"{provider_id} (test fixture)",
        credential_scope=credential_scope,
        runtime_env=runtime_env,
        base_url="https://fixture.invalid",
        build=build,
        combinations=(
            ModelCombination(
                provider_id=provider_id,
                model_id=model_id,
                compatibility_status=status,
                ui_exposed=ui_exposed,
            ),
        ),
    )


def _register(monkeypatch: pytest.MonkeyPatch, descriptor: ProviderDescriptor) -> None:
    """Register a test-only provider in the real registry for one test."""
    monkeypatch.setitem(PROVIDER_REGISTRY, descriptor.provider_id, descriptor)
    for combination in descriptor.combinations:
        monkeypatch.setitem(
            COMBINATION_INDEX,
            (combination.provider_id, combination.model_id),
            combination,
        )


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MODEL_PROVIDER",
        "MODEL_ID",
        "AWS_REGION",
        "API_KEY",
        "OPENAI_BASE_URL",
        "BASE_URL",
        "FIXTURE_A_KEY",
        "FIXTURE_B_KEY",
    ):
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


def test_offline_mode_is_registered_but_never_consumer_exposed() -> None:
    combination = resolve_combination("none", "none")
    assert combination is not None
    assert combination.compatibility_status is CompatibilityStatus.VERIFIED
    assert combination.ui_exposed is False
    assert all(entry["provider_id"] != "none" for entry in verified_combinations())


# --------------------------------------------------------------------------- #
# bedrock
# --------------------------------------------------------------------------- #
def test_bedrock_selection_uses_registered_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    _register(
        monkeypatch,
        _fixture_descriptor(
            "bedrock",
            model_id="test-bedrock-model",
            status=CompatibilityStatus.EXPERIMENTAL,
            runtime_env=("AWS_REGION",),
            build=model_factory._build_bedrock,
        ),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock")
    monkeypatch.setenv("MODEL_ID", "test-bedrock-model")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr("strands.models.bedrock.BedrockModel", _RecordingBedrockModel)

    model = build_model()
    assert isinstance(model, _RecordingBedrockModel)
    assert model.kwargs == {"model_id": "test-bedrock-model", "region_name": "us-east-1"}


def test_bedrock_missing_vars_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock")
    with pytest.raises(ValueError, match="MODEL_ID and AWS_REGION"):
        build_model()
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(ValueError, match="MODEL_ID and AWS_REGION"):
        build_model()


def test_bedrock_unregistered_model_id_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shipping registry has no approved exact Bedrock model id yet.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "bedrock")
    monkeypatch.setenv("MODEL_ID", "some-model")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(ValueError, match="Unregistered model combination"):
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
# unknown provider / unknown combination
# --------------------------------------------------------------------------- #
def test_unknown_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="none, bedrock, deepseek"):
        build_model()


def test_unknown_provider_error_is_derived_from_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    _register(
        monkeypatch,
        _fixture_descriptor(
            "acmeprovider",
            model_id=_DEV_MODEL_ID,
            status=CompatibilityStatus.EXPERIMENTAL,
        ),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    with pytest.raises(ValueError) as excinfo:
        build_model()
    assert "acmeprovider" in str(excinfo.value)


def test_registered_provider_does_not_authorise_an_arbitrary_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-reasoner")
    monkeypatch.setenv("API_KEY", "test-api-key")
    with pytest.raises(ValueError, match="Unregistered model combination"):
        build_model()


def test_unsupported_combination_is_refused_on_the_dev_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-chat")
    monkeypatch.setenv("API_KEY", "test-api-key")
    with pytest.raises(ValueError, match="UNSUPPORTED model combination"):
        build_model()


def test_experimental_combination_is_allowed_on_the_dev_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setattr("strands.models.openai.OpenAIModel", _RecordingOpenAIModel)

    combination = resolve_combination("deepseek", "deepseek-flash")
    assert combination is not None
    assert combination.compatibility_status is CompatibilityStatus.EXPERIMENTAL
    assert build_model() is not None


# --------------------------------------------------------------------------- #
# registry integrity / status semantics
# --------------------------------------------------------------------------- #
def test_registry_integrity() -> None:
    seen: set[tuple[str, str]] = set()
    for key, descriptor in PROVIDER_REGISTRY.items():
        assert key == descriptor.provider_id
        assert descriptor.display_name.strip()
        assert isinstance(descriptor.credential_scope, tuple)
        assert isinstance(descriptor.runtime_env, tuple)
        assert descriptor.base_url is None or descriptor.base_url.startswith("https://")
        assert callable(descriptor.build)
        assert isinstance(descriptor.combinations, tuple)
        for combination in descriptor.combinations:
            assert combination.provider_id == descriptor.provider_id
            assert combination.model_id.strip()
            assert isinstance(combination.compatibility_status, CompatibilityStatus)
            pair = (combination.provider_id, combination.model_id)
            assert pair not in seen, f"duplicate combination {pair}"
            seen.add(pair)

    assert set(COMBINATION_INDEX) == seen
    for descriptor in PROVIDER_REGISTRY.values():
        for combination in descriptor.combinations:
            key = (combination.provider_id, combination.model_id)
            assert COMBINATION_INDEX[key] is combination


def test_resolve_combination_requires_an_exact_pair() -> None:
    assert resolve_combination("deepseek", "deepseek-flash") is COMBINATION_INDEX[
        ("deepseek", "deepseek-flash")
    ]
    assert resolve_combination("deepseek", "deepseek-unknown") is None
    assert resolve_combination("bedrock", "deepseek-flash") is None
    assert resolve_combination("", "") is None
    assert resolve_combination("DEEPSEEK", "deepseek-flash") is not None


def test_verified_combinations_requires_verified_and_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturehidden",
            model_id=_HIDDEN_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=False,
        ),
    )
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureopen",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
        ),
    )
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureexperimental",
            model_id=_EXPERIMENTAL_MODEL_ID,
            status=CompatibilityStatus.EXPERIMENTAL,
            ui_exposed=True,
        ),
    )

    model_ids = {entry["model_id"] for entry in verified_combinations()}
    assert _CONSUMER_MODEL_ID in model_ids
    assert _HIDDEN_MODEL_ID not in model_ids  # VERIFIED but withheld
    assert _EXPERIMENTAL_MODEL_ID not in model_ids  # the flag cannot upgrade status


def test_provider_status_report_exposes_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "sk-sentinel-1234567890")
    report = provider_status_report()

    by_id = {entry["provider_id"]: entry for entry in report}
    assert set(by_id) == set(PROVIDER_REGISTRY)
    assert by_id["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert by_id["deepseek"]["credential_scope"] == ["API_KEY"]

    statuses = {
        combination["model_id"]: combination["compatibility_status"]
        for combination in by_id["deepseek"]["combinations"]
    }
    assert statuses["deepseek-flash"] == "EXPERIMENTAL"
    assert statuses["deepseek-chat"] == "UNSUPPORTED"

    for entry in report:
        for combination in entry["combinations"]:
            assert set(combination) == {
                "model_id",
                "compatibility_status",
                "ui_exposed",
                "certification_ref",
                "notes",
            }

    dumped = repr(report)
    assert "sk-sentinel-1234567890" not in dumped


# --------------------------------------------------------------------------- #
# builder contract
# --------------------------------------------------------------------------- #
def test_dev_path_passes_the_resolved_combination_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    calls: list = []
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturedev",
            model_id=_DEV_MODEL_ID,
            status=CompatibilityStatus.EXPERIMENTAL,
            calls=calls,
        ),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "fixturedev")
    monkeypatch.setenv("MODEL_ID", _DEV_MODEL_ID)

    model = build_model()

    assert calls[0]["combination"] is COMBINATION_INDEX[("fixturedev", _DEV_MODEL_ID)]
    assert model.model_id == _DEV_MODEL_ID


def test_builder_never_re_reads_model_id_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    captured: dict = {}

    def _build(combination, runtime_fields, credential):
        # Environment drift must not change the model the builder produces: the
        # resolved combination owns the model id.
        monkeypatch.setenv("MODEL_ID", "hijacked-model")
        captured["model_id"] = combination.model_id
        return _RecordingModel(combination, runtime_fields, credential)

    _register(
        monkeypatch,
        ProviderDescriptor(
            provider_id="fixturedrift",
            display_name="drift fixture",
            credential_scope=(),
            runtime_env=(),
            base_url="https://fixture.invalid",
            build=_build,
            combinations=(
                ModelCombination(
                    provider_id="fixturedrift",
                    model_id=_DEV_MODEL_ID,
                    compatibility_status=CompatibilityStatus.EXPERIMENTAL,
                ),
            ),
        ),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "fixturedrift")
    monkeypatch.setenv("MODEL_ID", _DEV_MODEL_ID)

    model = build_model()

    assert captured["model_id"] == _DEV_MODEL_ID
    assert model.model_id == _DEV_MODEL_ID


def test_callers_cannot_override_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid")
    monkeypatch.setenv("BASE_URL", "https://attacker.invalid")
    monkeypatch.setattr("strands.models.openai.OpenAIModel", _RecordingOpenAIModel)

    model = build_model()

    assert model.kwargs["client_args"]["base_url"] == "https://api.deepseek.com"


def test_runtime_config_has_no_base_url_field() -> None:
    assert set(RuntimeProviderConfig.__dataclass_fields__) == {
        "provider_id",
        "model_id",
        "credential",
        "runtime_fields",
    }


def test_runtime_path_forwards_only_declared_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturefields",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
            credential_scope=("API_KEY",),
            runtime_env=("AWS_REGION",),
            calls=calls,
        ),
    )
    model = build_model_from_runtime_config(
        RuntimeProviderConfig(
            provider_id="fixturefields",
            model_id=_CONSUMER_MODEL_ID,
            credential="sentinel-credential",
            runtime_fields={
                "AWS_REGION": "us-east-1",
                "base_url": "https://attacker.invalid",
                "API_KEY": "smuggled-value",
            },
        )
    )

    assert calls[0]["runtime_fields"] == {"AWS_REGION": "us-east-1"}
    assert calls[0]["credential"] == "sentinel-credential"
    assert model.model_id == _CONSUMER_MODEL_ID


def test_provider_credentials_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    calls_a: list = []
    calls_b: list = []
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturea",
            model_id=_DEV_MODEL_ID,
            status=CompatibilityStatus.EXPERIMENTAL,
            credential_scope=("FIXTURE_A_KEY",),
            calls=calls_a,
        ),
    )
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureb",
            model_id=_DEV_MODEL_ID,
            status=CompatibilityStatus.EXPERIMENTAL,
            credential_scope=("FIXTURE_B_KEY",),
            calls=calls_b,
        ),
    )
    monkeypatch.setenv("FIXTURE_A_KEY", "sentinel-key-a")
    monkeypatch.setenv("FIXTURE_B_KEY", "sentinel-key-b")
    monkeypatch.setenv("MODEL_PROVIDER", "fixturea")
    monkeypatch.setenv("MODEL_ID", _DEV_MODEL_ID)

    build_model()

    assert calls_a[0]["credential"] == "sentinel-key-a"
    assert calls_b == []
    assert "sentinel-key-b" not in repr(calls_a)


# --------------------------------------------------------------------------- #
# consumer / BYOK runtime path
# --------------------------------------------------------------------------- #
def test_runtime_path_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(provider_id="acme", model_id="x", credential="sentinel")
        )


def test_runtime_path_rejects_unregistered_combination() -> None:
    with pytest.raises(ValueError, match="Unregistered model combination"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(
                provider_id="deepseek", model_id="deepseek-unknown", credential="sentinel"
            )
        )


def test_runtime_path_rejects_experimental_combination() -> None:
    with pytest.raises(ValueError, match="Not available for consumer runtime use"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(
                provider_id="deepseek", model_id="deepseek-flash", credential="sentinel"
            )
        )


def test_runtime_path_rejects_verified_but_not_exposed_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturehiddenruntime",
            model_id=_HIDDEN_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=False,
        ),
    )
    with pytest.raises(ValueError, match="Not available for consumer runtime use"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(
                provider_id="fixturehiddenruntime",
                model_id=_HIDDEN_MODEL_ID,
                credential="sentinel",
            )
        )


def test_runtime_path_allows_verified_exposed_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureconsumer",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
            credential_scope=("API_KEY",),
            calls=calls,
        ),
    )

    model = build_model_from_runtime_config(
        RuntimeProviderConfig(
            provider_id="fixtureconsumer",
            model_id=_CONSUMER_MODEL_ID,
            credential="sentinel-credential",
        )
    )

    assert model.model_id == _CONSUMER_MODEL_ID
    assert calls[0]["credential"] == "sentinel-credential"


def test_runtime_path_requires_declared_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureregion",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
            runtime_env=("AWS_REGION",),
        ),
    )
    with pytest.raises(ValueError, match="AWS_REGION runtime configuration"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(provider_id="fixtureregion", model_id=_CONSUMER_MODEL_ID)
        )

    model = build_model_from_runtime_config(
        RuntimeProviderConfig(
            provider_id="fixtureregion",
            model_id=_CONSUMER_MODEL_ID,
            runtime_fields={"AWS_REGION": "us-east-1"},
        )
    )
    assert model.model_id == _CONSUMER_MODEL_ID


def test_runtime_path_requires_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixturecred",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
            credential_scope=("API_KEY",),
        ),
    )
    with pytest.raises(ValueError, match="API_KEY"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(provider_id="fixturecred", model_id=_CONSUMER_MODEL_ID)
        )
    with pytest.raises(ValueError, match="API_KEY"):
        build_model_from_runtime_config(
            RuntimeProviderConfig(
                provider_id="fixturecred", model_id=_CONSUMER_MODEL_ID, credential="   "
            )
        )


def test_runtime_path_never_mutates_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    _register(
        monkeypatch,
        _fixture_descriptor(
            "fixtureenv",
            model_id=_CONSUMER_MODEL_ID,
            status=CompatibilityStatus.VERIFIED,
            ui_exposed=True,
            credential_scope=("API_KEY",),
        ),
    )
    before = dict(os.environ)

    model = build_model_from_runtime_config(
        RuntimeProviderConfig(
            provider_id="fixtureenv",
            model_id=_CONSUMER_MODEL_ID,
            credential="sk-runtime-sentinel",
        )
    )

    assert dict(os.environ) == before
    assert "API_KEY" not in os.environ
    assert "sk-runtime-sentinel" not in os.environ.values()
    assert model.credential == "sk-runtime-sentinel"


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


def test_runtime_config_repr_excludes_the_credential() -> None:
    # A request-scoped secret must never surface through an implicit string
    # conversion (logging, debugging, error reporting).
    secret = "sk-runtime-sentinel-do-not-leak-in-repr"
    config = RuntimeProviderConfig(
        provider_id="deepseek", model_id="deepseek-flash", credential=secret
    )
    assert secret not in repr(config)
    assert secret not in str(config)
    assert "deepseek-flash" in repr(config)  # the repr stays useful


def test_runtime_path_never_leaks_the_credential_in_errors() -> None:
    secret = "sk-runtime-sentinel-do-not-leak"
    for config in (
        RuntimeProviderConfig(provider_id="deepseek", model_id="deepseek-flash", credential=secret),
        RuntimeProviderConfig(provider_id="deepseek", model_id="deepseek-unknown", credential=secret),
        RuntimeProviderConfig(provider_id="acme", model_id="deepseek-flash", credential=secret),
    ):
        with pytest.raises(ValueError) as excinfo:
            build_model_from_runtime_config(config)
        assert secret not in str(excinfo.value)
