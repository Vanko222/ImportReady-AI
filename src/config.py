"""Centralized configuration for the ImportReady AI MVP (P0-1)."""

from __future__ import annotations

import os
from pathlib import Path

# Environment override for the trusted data directory. Kept optional so the
# default local layout is used in normal development.
DATA_DIR_ENV = "IMPORTREADY_DATA_DIR"

# The six approved Data Schema v1 files. Load order is not semantically
# significant, but keeping the list in one place avoids scattered literals.
DATA_FILE_NAMES: tuple[str, ...] = (
    "product_taxonomy.json",
    "compliance_rules.json",
    "policy_sources.json",
    "compliance_costs.json",
    "known_gaps.json",
    "pending_policy_updates.json",
)

EXPECTED_SCHEMA_VERSION = "1.0"


def project_root() -> Path:
    """Absolute path of the repository root (src/config.py -> two levels up)."""
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the trusted data directory.

    Honors IMPORTREADY_DATA_DIR when set, otherwise defaults to
    <project_root>/data. The caller is responsible for existence checks so
    that a missing directory surfaces as a clear repository error.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).resolve()
    return project_root() / "data"
