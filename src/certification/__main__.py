"""Thin CLI for the P3.2 certification mechanism.

``--live-probe`` performs ONE manual minimal provider request (no tools, no gates) after validating
CLI/env/registry binding; full ``--live`` certification remains refused. Every other path is offline
and never constructs a model or touches the network.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.certification.certification import CASE_IDS, offline_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="importready-certification",
        description="Certify one exact (provider_id, model_id) combination. Offline; --live needs P3.2B.")
    parser.add_argument("--provider", required=True, help="exact provider_id, e.g. deepseek")
    parser.add_argument("--model", required=True, help="exact model_id, e.g. deepseek-flash")
    parser.add_argument("--live", action="store_true", help="real provider calls (P3.2B only; refused here)")
    parser.add_argument("--cases", default=",".join(CASE_IDS), help="comma-separated case ids (A,B,C)")
    parser.add_argument("--out", default=None, help="evidence artifact path (must be outside the repo)")
    parser.add_argument("--repo-root", default=None, help="repository root for evidence-path confinement")
    parser.add_argument("--live-probe", action="store_true", help="MANUAL: one minimal real provider request")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [c for c in (item.strip() for item in str(args.cases).split(",")) if c and c not in CASE_IDS]
    if unknown:
        print(f"ERROR [CONFIGURATION_MISMATCH]: unknown case ids {unknown}")
        return 2
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
        print("ERROR [CONFIGURATION_MISMATCH]: live certification is not authorized in P3.2A; "
              "the real-provider adapter is deferred to P3.2B")
        return 2
    code, message = offline_preview(args.provider, args.model)
    print(message)
    return code


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
