"""Configurable model provider factory.

Reads ``MODEL_PROVIDER`` / ``MODEL_ID`` / ``AWS_REGION`` from the environment.
No credentials are read from ``.env`` automatically and none are logged.
"""

from __future__ import annotations

import os

from strands.models.model import Model


def build_model() -> Model | None:
    """Return a Strands model instance, or ``None`` for the offline path.

    Supported providers: ``none`` (default, deterministic offline path),
    ``bedrock``, and ``deepseek`` (OpenAI-compatible). Unknown providers fail
    fast with a clear message rather than silently falling back to a stub.
    """
    provider = os.environ.get("MODEL_PROVIDER", "none").strip().lower()
    if provider in ("", "none"):
        return None

    if provider == "bedrock":
        model_id = os.environ.get("MODEL_ID")
        region = os.environ.get("AWS_REGION")
        if not model_id or not region:
            raise ValueError(
                "MODEL_PROVIDER=bedrock requires MODEL_ID and AWS_REGION environment variables"
            )
        from strands.models.bedrock import BedrockModel

        return BedrockModel(model_id=model_id, region_name=region)

    if provider == "deepseek":
        model_id = os.environ.get("MODEL_ID")
        api_key = os.environ.get("API_KEY")
        if not model_id or not model_id.strip():
            raise ValueError("MODEL_PROVIDER=deepseek requires MODEL_ID")
        if not api_key or not api_key.strip():
            raise ValueError("MODEL_PROVIDER=deepseek requires API_KEY")
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            client_args={
                "base_url": "https://api.deepseek.com",
                "api_key": api_key,
            },
            model_id=model_id,
        )

    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider!r} (supported: none, bedrock, deepseek)")
