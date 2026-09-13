"""Thin CLI for the P3.2 certification mechanism.

Live probing is explicitly opt-in and belongs to P3.2B; the real-provider adapter is deferred, so
``--live`` fails closed. Without it this command only validates the exact registered combination —
it never constructs a model and never touches the network.
"""

from __future__ import annotations

import argparse
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [c for c in (item.strip() for item in str(args.cases).split(",")) if c and c not in CASE_IDS]
    if unknown:
        print(f"ERROR [CONFIGURATION_MISMATCH]: unknown case ids {unknown}")
        return 2
    if args.live:
        print("ERROR [CONFIGURATION_MISMATCH]: live certification is not authorized in P3.2A; "
              "the real-provider adapter is deferred to P3.2B")
        return 2
    code, message = offline_preview(args.provider, args.model)
    print(message)
    return code


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
