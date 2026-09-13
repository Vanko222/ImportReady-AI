"""ImportReady provider-certification package (P3.2A — offline mechanism).

Public API (imported explicitly; there is no ``__all__`` duplication):

- :func:`src.certification.certification.run_certification` — bounded, deterministic gate runner
- :func:`src.certification.certification.preflight_live_target` — live CLI/env binding check
- :func:`src.certification.certification.offline_preview` — non-live CLI validation path
- ``python -m src.certification`` — thin CLI (live probing is deferred to P3.2B)

The real-provider adapter (``ProductionTarget``, live Gate-1 request, real proxy wiring) is deferred
to P3.2B; offline targets are injected through :class:`CertificationTarget`.
"""

from __future__ import annotations

from src.certification.certification import (
    CASE_IDS, CertificationRecord, CertificationTarget, FailureCategory, GateResult, GateStatus,
    OutputCaptureBoundary, OutputCaptureUnavailable, RunBudget, RunLimits, aggregate_gates,
    aggregate_status, check_final_response, offline_preview, overall_status, preflight_live_target,
    run_certification, sanitize, scan_for_secrets, write_evidence,
)

__version__ = "0.1.0"
