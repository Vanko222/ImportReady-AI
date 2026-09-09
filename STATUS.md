# STATUS.md

## Current Phase

**Data foundation complete and validated. Next phase: technical architecture and MVP implementation.**

## Completed

- Development environment ready: VS Code, Git/GitHub, Docker, Node.js, DeepSeek Harness.
- Formal private GitHub repository created and connected.
- Core project documents established: `AGENTS.md`, `PRODUCT.md`, `STATUS.md`, `DECISIONS.md`.
- MVP scope locked to two broad categories:
  - children's toys
  - small consumer electronics
- Product classification supports:
  - `childrens_toys`
  - `small_consumer_electronics`
  - `dual`
  - `unsupported`
  - `uncertain`
- Compliance research completed using four approved read-only research files.
- Supplement research completed for status semantics, source-update metadata, P0 clarification mappings, and compliance-cost references.
- Data Schema v1 created:
  - 86 canonical product attributes
  - 41 compliance rules
  - 70 policy sources
  - 32 compliance-cost records
  - 16 persistent known gaps
  - 0 pending policy updates
- Data Schema v1 independently validated:
  - JSON parsing passed
  - IDs are unique
  - rule/source/attribute references resolve
  - all 33 P0 clarification mappings are retained
  - no temporary `NEW ...` attributes remain
  - `PROPOSED` / `WATCHLIST` rules cannot create active obligations
  - `QUOTE_REQUIRED` is never treated as zero
- `compliance_costs.json` cleanup completed:
  - all 32 records now include `additional_source_ids`
  - `C-E-001`, `C-E-002`, `C-E-003` changed from `DIRECT` to `PLANNING_ONLY`
  - all other JSON files remained unchanged
- Validated Data Schema v1 baseline has been committed and pushed.

## Active Product Logic

- Active obligation evaluation requires:
  - `evidence_status = VERIFIED`
  - `rule_status = EFFECTIVE`
  - runtime trigger satisfied
- Missing trigger facts must return `NEEDS_INFO` or `REVIEW_REQUIRED`, not a false green result.
- `dual` products evaluate both relevant category paths and collect the union of required attributes, de-duplicated by canonical ID.
- P1 rules remain review/escalation records where clarification mappings are incomplete.
- Policy changes must follow:
  `detect candidate -> human review -> approve/reject -> trusted data update`
- Compliance costs distinguish:
  - `DIRECT`
  - `PLANNING_ONLY`
  - `QUOTE_REQUIRED`
  - `DISPLAY_ONLY`

## Known Gaps / Risks

- No end-to-end application has been implemented yet.
- Technical architecture is not yet finalized in `ARCHITECTURE.md`.
- Low / Medium / High risk-level logic is not yet formally defined.
- Tariffs, marketplace commissions, product cost, and logistics cost are not part of the current compliance dataset and require separate Cost Calculator logic.
- Some research gaps remain intentionally unresolved, including:
  - standalone lead/phthalates testing prices
  - public TCB fees
  - robust UN 38.3 market pricing
  - Amazon provider pricing and restricted help-page details
  - state-law overlays outside the current bounded scope
  - carrier-specific lithium transport rules

## Next

1. Define the minimal MVP technical architecture and create `ARCHITECTURE.md`.
2. Define the Low / Medium / High risk decision logic.
3. Let Harness create the minimal project skeleton.
4. Build the first end-to-end P0 flow:
   `product input -> classification -> rule routing -> clarification -> compliance result -> cost impact -> report`
5. Add policy-update candidate + human approval flow.
6. Add tests, then Dockerize the working local MVP.

## Immediate Goal

Get one complete, testable end-to-end product path working before adding breadth or UI polish.
