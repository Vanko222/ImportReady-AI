"""Thin CLI for the P3.2 certification mechanism.

Path map:

* ``--live-probe`` performs ONE manual minimal provider request (no tools, no gates) after validating
  CLI/env/registry binding (Step 2).
* ``--live --ack "<exact literal>"`` (P3.2B Step 3 Phase 2) reports the sanitized execution plan and
  **stops before any execution**: no model is constructed and no request is made.
* ``--live --ack "<exact literal>" --execute`` (Phase 3) additionally calls the existing certification
  runner once through ``run_guarded_live_certification`` and reports the record summary (overall status,
  gates, evidence filename + SHA-256) and the runner's exit code.
* ``--live`` without that acknowledgement (or with a wrong one) fails closed: the rejected value is
  never echoed, an ``ERROR [ACKNOWLEDGEMENT_REQUIRED|ACKNOWLEDGEMENT_REJECTED]`` line is printed and the
  exit code is 2.
* every other path is offline and never constructs a model or touches the network.

The guarded path reads nothing from the environment at all (not even the two non-secret binding names):
the only inputs are the command-line target, the requested cases, the acknowledgement literal, the
``--execute`` opt-in and the evidence path shown in the plan. Credential handling stays where it already
is — the operator's own shell, read only by the existing ``model_factory`` dev path at execution time
(the adapter/runner supply it; the CLI never sees it).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.certification.certification import CASE_IDS, default_evidence_dir, offline_preview

# Shown to the operator as the acknowledgement to type. Purely informational: the accepted value is
# compared by ``live_acknowledgement_required`` against the literal for the exact combination.
ACKNOWLEDGEMENT_PLACEHOLDER = "CERTIFY <provider>/<model>"
PASSING_OVERALL = ("PASS", "INCOMPLETE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="importready-certification",
        description="Certify one exact (provider_id, model_id) combination. Offline; --live is guarded.")
    parser.add_argument("--provider", required=True, help="exact provider_id, e.g. deepseek")
    parser.add_argument("--model", required=True, help="exact model_id, e.g. deepseek-flash")
    parser.add_argument("--live", action="store_true",
                        help="guarded real-provider path: renders the plan, then stops")
    parser.add_argument("--ack", default=None,
                        help=f'exact acknowledgement for --live, e.g. "CERTIFY deepseek/deepseek-flash" '
                             f"({ACKNOWLEDGEMENT_PLACEHOLDER}; no bypass exists)")
    parser.add_argument("--execute", action="store_true",
                        help="with --live and the exact --ack: call the existing certification runner")
    parser.add_argument("--cases", default=",".join(CASE_IDS), help="comma-separated case ids (A,B,C)")
    parser.add_argument("--out", default=None, help="evidence artifact path (must be outside the repo)")
    parser.add_argument("--repo-root", default=None, help="repository root for evidence-path confinement")
    parser.add_argument("--live-probe", action="store_true", help="MANUAL: one minimal real provider request")
    return parser


def _record_summary(record: object) -> tuple[str, str] | None:
    """Compact, sanitized rendering of a certification record (never a path, never a credential).

    Returns ``None`` when the object does not look like a record, so an injected result degrades to the
    headline status rather than raising.
    """
    overall, render = getattr(record, "overall", None), getattr(record, "render_markdown", None)
    if not callable(overall) or not callable(render):
        return None
    try:
        text = render()
    except Exception:  # noqa: BLE001 - a rendering failure must not hide the run result
        return None
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    evidence_rows = [row for row in rows if "evidence" in row]
    # A readable headline plus the evidence rows (filename + SHA-256) for the operator's record.
    return str(overall()), "\n".join(rows[:6] + evidence_rows)


def _live_command(args: argparse.Namespace, cases: tuple[str, ...]) -> int:
    """Guarded live path: resolve the target, require the exact acknowledgement, render the plan.

    Fails closed before any model or request. The acknowledgement is checked **before** the plan is
    rendered, so a rejected value (which could be a pasted credential) is never echoed back. Without
    ``--execute`` the command stops after the plan (Phase 2 behaviour). With ``--execute`` it calls the
    existing runner once through the adapter bridge and reports the record summary.
    """
    from src.certification.live_target import (
        Acknowledgement,
        live_acknowledgement_required,
        render_live_certification_plan,
        require_combination,
    )

    try:
        # Pure registry lookup: no model, no credential, no network.
        combination = require_combination(args.provider, args.model)
    except ValueError as exc:
        print(f"ERROR [CONFIGURATION_MISMATCH]: {exc}")
        return 2
    acknowledgement: Acknowledgement = live_acknowledgement_required(
        combination.provider_id, combination.model_id, args.ack)
    if not acknowledgement.accepted:
        # The provided value is never printed: it is only compared, then discarded.
        print(f"ERROR [{acknowledgement.code}]: {acknowledgement.detail}")
        print(f"usage: --ack \"{acknowledgement.literal}\"")
        print("no execution performed and no plan rendered")
        return 2
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    evidence_path = Path(args.out) if args.out else default_evidence_dir(repo_root) / (
        f"{combination.provider_id}__{combination.model_id}.txt")
    # Registry facts are displayed information only: they never gate whether the command proceeds.
    plan = render_live_certification_plan(
        combination.provider_id, combination.model_id, cases=cases, evidence_path=evidence_path,
        status=combination.compatibility_status.value, ui_exposed=combination.ui_exposed,
        certification_ref=combination.certification_ref, acknowledgement=acknowledgement.literal)
    print(plan.text)
    if not args.execute:
        print("acknowledgement accepted; STEP 3 PHASE 2 STOPS HERE — no provider request was made")
        print("add --execute to call the existing certification runner")
        return 0
    return _execute_live(args, cases, combination, repo_root, evidence_path)


def _execute_live(args: argparse.Namespace, cases: tuple[str, ...], combination,
                  repo_root: Path, evidence_path: Path) -> int:
    """Call the existing runner once through the adapter bridge; report the record and the exit code."""
    from src.certification.certification import classify_provider_error, sanitize
    from src.certification.live_target import run_guarded_live_certification

    print("--execute: calling the existing certification runner through the live adapter "
          "(the operator's shell supplies the credential; this CLI never reads it)")
    # The frozen runner's live preflight compares its target with these two non-secret binding names.
    # They are read here, in the CLI, exactly as the Step 2 probe path reads them; nothing else is read.
    environ = {name: os.environ[name] for name in ("MODEL_PROVIDER", "MODEL_ID") if name in os.environ}
    try:
        record = run_guarded_live_certification(
            combination.provider_id, combination.model_id, cases=cases, environ=environ,
            repo_root=repo_root, evidence_out=evidence_path, write_evidence_file=True)
    except Exception as exc:  # noqa: BLE001 - a construction failure is evidence, never a raw traceback
        # e.g. the operator's shell has no credential: report the frozen taxonomy category, sanitized.
        print(f"ERROR [{classify_provider_error(exc).value}]: {type(exc).__name__} "
              f"{sanitize(str(exc))[:300]}")
        print("no certification record was produced; set the provider's environment in your own shell "
              "and retry")
        return 1
    summary = _record_summary(record)
    overall = summary[0] if summary else str(getattr(record, "status", "UNKNOWN"))
    print(f"overall status: {overall}")
    if summary is not None:
        print(summary[1])
    print("any provider request was made by the acknowledged run through the adapter and the frozen "
          "runner, never by this CLI; the registry status was not modified")
    return 0 if overall in PASSING_OVERALL else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [c for c in (item.strip() for item in str(args.cases).split(",")) if c and c not in CASE_IDS]
    if unknown:
        print(f"ERROR [CONFIGURATION_MISMATCH]: unknown case ids {unknown}")
        return 2
    cases = tuple(c for c in (item.strip() for item in str(args.cases).split(",")) if c)
    if args.live_probe:
        from src.certification.live_target import live_probe

        # Only the two non-secret names are read; the credential stays in the process environment.
        environ = {name: os.environ[name] for name in ("MODEL_PROVIDER", "MODEL_ID") if name in os.environ}
        result = live_probe(args.provider, args.model, environ)
        print(f"probe {args.provider}/{args.model}: {'OK' if result.ok else 'FAILED'} ({result.code})")
        print(f"latency_ms={result.latency_ms} stop_reason={result.stop_reason} content_kinds={list(result.content_kinds)} reasoningContent={result.reasoning_observed}")
        for line in ([f"response={result.response_text!r}"] if result.response_text else []) + list(result.captured):
            print(line)
        if result.error_message:
            print(f"error={result.error_type} {result.error_message}")
        return 0 if result.ok else 1
    if args.live:
        return _live_command(args, cases)
    code, message = offline_preview(args.provider, args.model)
    print(message)
    return code


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
