"""Cached, credential-free UI resources.

Only the read-only approved repository and the deterministic analysis service are
cached. Model objects are **never** cached, because a consumer BYOK model object
holds the session credential by reference.
"""

from __future__ import annotations

import streamlit as st

from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.analysis import AnalysisService


@st.cache_resource(show_spinner=False)
def get_repository() -> JsonComplianceRepository:
    """Approved knowledge base (read-only; contains no credentials)."""
    return JsonComplianceRepository()


@st.cache_resource(show_spinner=False)
def get_analysis_service() -> AnalysisService:
    """Accepted analysis service over the approved repository (no credentials)."""
    return AnalysisService(get_repository())
