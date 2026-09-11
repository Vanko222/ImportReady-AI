"""Configurable model provider factory.

Reads ``MODEL_PROVIDER`` / ``MODEL_ID`` / ``AWS_REGION`` from the environment.
No credentials are read from ``.env`` automatically and none are logged.
"""

from __future__ import annotations

import os

from strands.models.model import Model


def build_model() -> Model | None:
    """Return a Strands model instance, or ``None`` for the offline path.

    Supported providers: ``none`` (default, deterministic offline path) and
    ``bedrock``. Unknown providers fail fast with a clear message rather than
    silently falling back to a stub.
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

    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider!r} (supported: none, bedrock)")
