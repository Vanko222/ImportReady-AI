# P3.2 Provider Certification — Implementation Plan

Status: **P3.2 plan APPROVED · P3.2A implementation AUTHORIZED and COMPLETED (offline mechanism) ·
P3.2B real-provider certification NOT yet authorized.**

- P3.2 plan approved after human review (revisions 1–4 plus the final consistency corrections).
- **P3.2A — offline certification mechanism: AUTHORIZED, implemented and tested** (see
  "P3.2A implementation result" below).
- **P3.2B — live DeepSeek certification: NOT AUTHORIZED.** No real provider/API call has been made;
  no status promotion, no `ui_exposed` change.
- The final consistency corrections (test-scope items 1–21, the per-case confirmation record, and the
  Case-B gate-2 failure mapping) were completed after Revision 4 and before implementation.

## P3.2A implementation result (offline mechanism complete)

| Item | Result |
|---|---|
| Files created | `src/certification/__init__.py`, `src/certification/certification.py`, `src/certification/__main__.py`, `tests/test_certification.py` |
| Production certification LOC | **1,299** (`certification.py` 1,231 · `__init__.py` 23 · `__main__.py` 45) |
| Revised size guard | **≤ 1300 lines — respected** (after an approved LOC-reduction pass) |
| Certification tests | **124 passed / 0 failed** (`python -m pytest tests/test_certification.py -q`) |
| Full offline suite | **560 passed / 0 failed** (436 pre-existing + 124 new) |
| Offline test groups retained | **all 21** plus the six targeted review-fix groups |
| Deterministic authority | exercised for real: `HumanClassifier`, `Orchestrator`, `AnalysisService`, the two real Agent tools; only the provider side is faked |
| Real API/provider calls | **none** |
| Protected production files modified | **none** (`git diff --stat` empty) |
| P3.2B live adapter | **deferred, NOT STARTED** |

**Targeted review fixes (all implemented and tested).** Captured provider output now fails the case
closed on **successful** calls too (stdout/stderr/warnings/logging; pre-existing console
`logging.StreamHandler`s are rebound inside the boundary and restored, or the observation is
`BLOCKED`/`OUTPUT_CAPTURE_UNAVAILABLE`); Gate 8 additionally rejects `NOT_FOUND`-as-"not required"
prose, raw exception class-name dumps, and the runtime credential literal (regex-independent); Case C
now requires the observed result to match the deterministic reference (the reference alone can never
PASS); the record renders the complete approved skeleton with **filename + SHA-256 + summary only**
(no absolute local paths); R-ELEC-002 `NOT_APPLICABLE`/`TRIGGER_NOT_SATISFIED` is asserted against the
real engine; and soft wall-clock bounds (120 s per case / 600 s per run, injectable monotonic clock)
stop new provider work with `BLOCKED`/`TIMEOUT`, with gate-1, classification and agent requests all
charged against the 24-request ceiling.

**LOC-reduction pass (no behaviour removed).** A final refactoring pass took production code from
1,416 → **1,299** lines without deleting any gate, safety check, record field or test: the duplicated
`__all__` export list was removed, the failure taxonomy became one documented name tuple driving the
enum, the repeated `GateResult`/`CaseObservation` constructions and canonical projections were
consolidated onto shared helpers/expressions, the gate-9 scenario body was extracted into one
sanitized helper, the record target table is table-driven, and gate/runner formatting was compacted.
Verified by the full green suite (124 certification tests, 560 total).

**Deferred to P3.2B (live-only, real provider):** the concrete `ProductionTarget` that constructs a
real model, `production_build_model()`, the real Gate-1 provider request, real
`_suggest_category`/`AgentClassifier` provider wiring, real Strands/provider message and
tool-continuation introspection, and live interactive provider execution plumbing. Gate 1–10
contracts remain abstract and fully fake-testable offline; `--live` fails closed in P3.2A.

Revision 4 — incorporates the third human review round (four corrections): (1) gate 7 treats the
`reasoningContent` warning as observable-but-**optional** (`NOT_OBSERVED` never fails the gate and the
warning is never forced to reproduce), (2) the live CLI target is **bound to the actual environment
target** before any model construction or network request, (3) `--out` evidence paths must **resolve
outside the Git repository**, and (4) explicit gate-2 criteria for Case B. All revision-3 decisions
remain in force.

Revision 3 — incorporates the second human review round (three corrections): (1) global vs case-scoped
gates and deterministic result aggregation, (2) removal of the duplicate live Case D (lifecycle safety
moves to live gate 10 + mandatory offline Gate-8 adversarial prose tests), (3) an implementable
in-process stdout/stderr/warnings/logging capture boundary in place of the previous contradictory
secret-safety wording. All other approved decisions are unchanged (revision 2's human-confirmation
boundary, semantic gates 3/6, evidence outside the repo, and the ~600-line scope guard remain in
force).

Plan date: written after P3.1 was committed. Follows `P3_1_Multi_Provider_Architecture_Plan.md`.

---

## A. P3.2 objective

Build a **repeatable Provider Certification mechanism** that produces evidence for one **exact
`(provider_id, model_id)` combination**:

> Can this exact model, through this exact provider endpoint, safely operate ImportReady's existing
> Agent interface — and can deterministic ImportReady code remain the sole authority for canonical
> compliance output?

Certification answers that question and nothing more. It does **not** certify:

- legal/compliance correctness of any model output;
- general model intelligence or quality;
- every model of a provider (provider names are never certified);
- every endpoint that happens to be OpenAI-compatible.

P3.2 is split into two separately reviewed substeps:

- **P3.2A — certification mechanism (offline only).** Deterministic, no network, fake/sentinel
  credentials, part of the normal offline suite. Certifies the *mechanism*, not provider
  compatibility.
- **P3.2B — real certification of `deepseek` + `deepseek-flash`.** Bounded real calls, human-held
  credential, sanitized evidence, human sign-off.

P3.2A must never be mixed with P3.2B: implementation bugs and live-provider compatibility bugs must
stay distinguishable.

## B. Current baseline (verified during this planning pass)

| Item | Value | How verified |
|---|---|---|
| Branch | `main`; working tree clean apart from this plan file | `git status --short` |
| HEAD | `84d420b` — "Implement P3.1 multi-provider architecture" (`84d420bf2aba90c5edd617acb156d586e922c025`) | `git rev-parse HEAD`, `git log --oneline -2` |
| Full offline suite | **436 passed / 0 failed** | `python -m pytest -q` |
| `strands-agents` | 1.55.1 | `importlib.metadata` |
| `openai` | 2.54.0 | `importlib.metadata` |
| `pydantic` | 2.13.5 | `importlib.metadata` |
| Pytest configuration | **none** — no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or marker registration exists (only `tests/conftest.py`, which only fixes `sys.path`) | path check |
| Exact combinations | `deepseek` + `deepseek-flash` = `EXPERIMENTAL`; `deepseek` + `deepseek-chat` = `UNSUPPORTED`; `bedrock` registered with **no** combination; `none` = `VERIFIED` + `ui_exposed=False` | `provider_status_report()` |
| `verified_combinations()` | `[]` | direct call |

> **Baseline correction.** The task brief cites HEAD `a4d420b`. The actual verified HEAD is
> **`84d420b`**. The plan uses the verified value.

Relevant existing behaviour that P3.2 must reuse rather than reinvent (read-only observations):

| Concern | Existing implementation |
|---|---|
| Model selection (dev/certification path) | `src/agent/model_factory.build_model()` — reads `MODEL_PROVIDER`, `MODEL_ID`, `API_KEY`/`AWS_REGION`; `EXPERIMENTAL` and `VERIFIED` are permitted here |
| Agent run wiring | `src/agent/app.py::_make_agent_runner(model, analysis_service)` → `Agent(model=…, tools=…, system_prompt=SYSTEM_PROMPT, callback_handler=None)` → `agent(prompt, limits=_AGENT_LIMITS)`; returns `AgentRunOutcome(status, stop_reason, text, tool_calls, error_type, analysis_result)` |
| Agent limits | `_AGENT_LIMITS = {"turns": 6, "output_tokens": 1200, "total_tokens": 10000}` |
| Agent classification | `src/agent/app.py::_suggest_category(model, description, allowed)` (structured output, `limits={"turns": 2, "output_tokens": 200, "total_tokens": 4000}`) + `src/services/classification.py::AgentClassifier` / `agent_suggestion()` → **always** `agent_generated`, never `RESOLVED` |
| **Human category confirmation (the only existing boundary)** | `src/services/classification.py::HumanClassifier.classify(product_description, provided_category)`; CLI wiring in `src/agent/app.py::main()` (`if args.category: classifier = HumanClassifier(allowed)`); reached through `src/services/orchestrator.py::Orchestrator.run(..., provided_category=…)` |
| Canonical applicability gate | `src/services/analysis.py:164-172` — definitive applicability is evaluated **only** when `category_status == RESOLVED`; an agent-generated `REVIEW_REQUIRED` category yields `applicability is None` ("not evaluated"), and `_review_status()` returns `REVIEW_REQUIRED` for every non-`RESOLVED` category |
| Lifecycle metadata available canonically | `src/services/analysis.py:41-49` — `ComplianceFinding` (in `analysis.verified.compliance_information`) carries `rule_id`, `evidence_status`, `rule_status`; `R-ELEC-018` = `WATCHLIST`, `R-ELEC-019` = `PROPOSED` |
| Tools | `src/agent/tools.py::build_tools()` — exactly `analyze_product(product_description)` and `get_compliance_evidence(rule_id)`; `MAX_TOOL_CALLS = 4`; per-request `ToolExecutionState` captures the canonical `AnalysisResult` |
| Canonical path | `Orchestrator.run(...)` → `AnalysisService` → deterministic Applicability Engine; agent failures produce an explicitly `REVIEW_REQUIRED` deterministic fallback with a sanitized error |
| Stop reasons | `_NORMAL_STOP_REASONS = {end_turn, stop_sequence}`; anything else is `LIMIT_REACHED` or `FAILED` |
| Sanitisation | `redact_secrets()` + `_SECRET_PATTERNS` (AKIA…, `Bearer …`, `token=…`, `secret=…`, `sk[-_]…`) |
| Exit codes | `0` success, `1` agent failure/fallback, `2` input/facts error, `3` model unavailable, `4` classification failure |
| Fixture data | R-ELEC-002 requires exactly 7 attributes (`A-ELEC-002/003/007/008/009/010/021`, `A-ELEC-002` deciding); `R-ELEC-018` is `WATCHLIST`; `R-ELEC-019` is `PROPOSED` |

## C. Current DeepSeek status (input to certification, not a conclusion)

Canonical model: `deepseek-flash`; endpoint `https://api.deepseek.com` (ImportReady-owned, held by
the `deepseek` descriptor). The former HTTP 400 issue is **resolved** (the unsupported
`params={"extra_body": …}` argument is gone; a regression assertion guards it). Do **not** recommend
`deepseek-chat` or `deepseek-v4-flash`, and do **not** reintroduce thinking/`extra_body` parameters.

Already proven with real calls: minimal request → HTTP 200; real Strands request → HTTP 200; both
tools registered; model selected `analyze_product`; tool executed; deterministic Applicability
returned; R-ELEC-002 produced the correct canonical result when the required USER facts were supplied.

Still open: a later multi-turn stage emitted

> `reasoningContent is not supported in multi-turn conversations with the Chat Completions API.`

Therefore `deepseek` + `deepseek-flash` **remains `EXPERIMENTAL`**. P3.2 must not assume the warning
is fatal, and must not assume it is harmless: it must produce evidence and classify it (gate 7).

## D. Certification architecture

```
exact (provider_id, model_id)
        │
        ├─ CertificationTarget ──────────────── production adapter (read-only imports)
        │     build_model()                      src.agent.model_factory.build_model()
        │     classify(model, description)       src.agent.app._suggest_category
        │                                           + services.classification.AgentClassifier
        │     confirm_category(suggestion)       ← HUMAN STEP (D.3); reuses
        │                                           services.classification.HumanClassifier
        │     run_agent(model, case, confirmed)  src.agent.app._make_agent_runner + Orchestrator
        │     run_offline(case, confirmed)       Orchestrator.run (no agent_runner)
        │
        ├─ RecordingModelProxy (transparent, observational only)
        ├─ OutputCaptureBoundary (in-process: stdout/stderr/warnings/logging)
        │
        ▼
   certification run
        │   GLOBAL  gate 1  (once, before any case)
        │   CASE    gates 2–8, 10  (per applicable case: A, B, C)
        │   GLOBAL  gate 9  (offline failure injection, once)
        ▼
   deterministic aggregation ──► ten-gate result + Gate × Case matrix
        │
        ▼
   bounded runner ──► sanitized evidence artifact (OUTSIDE the Git project)
        │                + CertificationRecord draft (docs/certification/, after human review)
        ▼
   HUMAN review ──► separate approved change sets ModelCombination to VERIFIED
                    (ui_exposed is a further, separate decision)
```

Design rules:

1. **The certified path is the production path.** Gates 3–8 and 10 drive the real
   `Orchestrator` + `_make_agent_runner` + `build_tools` wiring with a real model object. The
   certification layer never re-implements compliance logic, tool logic, classification logic, or the
   agent runner.
2. **Observation, not modification.** The only things inserted between the app and the provider are a
   *transparent* recording proxy at the Strands `Model` seam and an in-process output-capture boundary.
   Both delegate/behave unchanged. No request, message, or history mutation is permitted on any path.
3. **Human category confirmation is a real, existing boundary — never a certification shortcut.** The
   certification layer must not create a second category authority (section H, gates 2 / H.2).
4. **Gate scoping is explicit and aggregation is deterministic** (section H.0): gate 1 is GLOBAL and
   runs exactly once per certification run; gate 9 is a GLOBAL offline gate; gates 2–8 and 10 are
   case-scoped. No case failure may be hidden by another case passing.
5. **Target injection.** All gate probes take a `CertificationTarget`. Offline tests inject fake
   targets; the live runner injects `ProductionTarget`. This is what makes P3.2A possible with zero
   network and zero production change.
6. **No status mutation.** The certification layer only *reads* `PROVIDER_REGISTRY` /
   `COMBINATION_INDEX` / `resolve_combination()` / `verified_combinations()`. It never writes a
   `CompatibilityStatus`.
7. **Output capture is in-process and bounded.** Raw provider/SDK output is captured in memory,
   sanitized, scanned, summarized, and then **discarded** — never written to disk, never re-emitted
   (section N/S). If an output channel cannot be captured in-process, the affected observation is
   `BLOCKED` and the run stops for approval (section W).

### D.3 The existing confirmation path P3.2A will reuse (read-only inspection result)

The repository has **exactly one** mechanism that can turn a category into a canonical one:

```
src/agent/app.py::main()
    if args.category:  classifier = HumanClassifier(allowed)          # the CLI human boundary
    else:              classifier = AgentClassifier(suggest_fn, allowed)
                              │
src/services/orchestrator.py::Orchestrator.run(description, provided_category=…, agent_runner=…)
                              │  self._classifier.classify(description, provided_category)
                              ▼
src/services/classification.py::HumanClassifier.classify(description, provided_category)
    value not in allowed / empty      → category=None, UNRESOLVED,      NEEDS_INFO
    value == "unsupported"            → category="unsupported", HUMAN_CONFIRMED, UNSUPPORTED
    value == "uncertain"              → category="uncertain",   HUMAN_CONFIRMED, NEEDS_INFO
    otherwise (in allowed vocabulary) → category=value,         HUMAN_CONFIRMED, RESOLVED
```

Consequences the certification must respect (all verified in code):

- `HumanClassifier` is the **only** producer of `category_source=human_confirmed`; the certification
  layer must never construct a `CategoryResult`, never set `human_confirmed`, and never mutate the
  agent's suggestion object.
- Canonical applicability exists **only** for `category_status == RESOLVED`
  (`analysis.py:164-172`). Gates 5 and 10 (canonical R-ELEC-002 preservation) are therefore
  *impossible* from an agent-generated category — exactly the product invariant. The certification
  uses the real confirmation boundary instead of working around it.
- `_review_status()` returns `REVIEW_REQUIRED` for any non-`RESOLVED` category, and `_suggestions()`
  records the agent category as `{"kind": "category_suggestion", … "note": "agent-generated category;
  requires human confirmation"}`.
- Production records the reviewer action `confirm_or_correct_category` (`analysis.py:342-343`) for
  `REVIEW_REQUIRED` categories — the certification's human step mirrors that existing product action.

**Therefore:** P3.2A reuses `HumanClassifier` (through `Orchestrator.run(..., provided_category=…)`)
as the confirmation boundary. No new confirmation code path, service, or category authority is
created; the only new code is the interactive presentation of the suggestion and the recording of the
confirmation event.

## E. Recommended certification implementation method

| Criterion | A. Dedicated script only | B. Pytest live marker | C. Small module + thin runner | D. Wrap existing CLI |
|---|---|---|---|---|
| Production-code impact | none | none, but needs a new pytest config file (none exists today) | none (new dev package only) | none |
| Reproducibility | medium (script drifts) | high | high | medium |
| Per-gate/per-case recording | ad hoc | awkward | first-class `GateResult` + scope | poor (one coarse outcome) |
| Secret safety | manual | risky (test output/artifacts) | centralised sanitizer + in-process capture | manual, coarse |
| Reuse for Bedrock/competition | copy-paste | medium | single target protocol | medium |
| Cannot run under plain `pytest -q` | yes (not a test) | **only with config + discipline** | yes (not a test; gates are called by tests with fakes) | yes |
| Beginner simplicity | high | low | medium-high | high |

**Recommendation: C — a small certification package (gates + record + sanitizer + capture boundary)
with a thin `python -m src.certification` runner, sized only after the read-only interface inspection
(section P).**

Rationale: it is the only option that gives per-gate/per-case evidence, centralised secret handling,
and full provider reuse **without touching production code**, and live calls can never happen inside
`python -m pytest -q`, because the live path is a CLI that constructs a real model only when `--live`
is passed. Option D is kept as a secondary *end-to-end corroboration probe* (run the real CLI once per
case and compare its output with the gate evidence), not as the mechanism. Option B is rejected: it
would need a new pytest config file and would put network-capable code on the normal test path.

## F. P3.2A — offline mechanism implementation

Deliverable: the certification package + offline tests, all offline, fake credentials only, 0 network
calls, with `python -m pytest -q` still 100 % offline. The file/module structure is **decided after
step 1**, not in advance (section P).

Implementation steps:

1. **Read-only interface inspection.** Determine, from the installed packages (no network):
   - `strands.models.model.Model` — which methods a transparent proxy must implement/delegate, and
     whether a transparent proxy at that seam is feasible at all;
   - `AgentResult` — which attributes exist in `strands-agents 1.55.1` (turns, messages, metrics, stop
     reason) and how tool-use / tool-result / continuation are represented;
   - whether the Strands `Agent` exposes its registered tool specs in a readable form;
   - **every output channel used during a model call**: Python-level `sys.stdout`/`sys.stderr`,
     `warnings`, the `logging` tree (root, `importready`, provider SDK loggers), and whether any
     involved component (e.g. a native/compiled HTTP stack) writes **directly to the OS file
     descriptor**, which Python-level redirection cannot capture.
   Record the findings in code docstrings. **Then choose the smallest structure that still keeps
   secret handling, gate probes, and the runner readable** (section P). If a channel cannot be safely
   captured/suppressed in-process without changing production behaviour → mark the affected
   observation `BLOCKED`, **stop**, and request human approval (section W).
2. **Types and record:** gate ids 1–10, `GateScope` (`GLOBAL` / `case:<id>`), `PASS`/`FAIL`/
   `BLOCKED`/`NOT_RUN`, failure categories, `GateResult(gate, scope, status, failure_category,
   evidence, warnings)`, `CertificationRecord`, `RunLimits`/`RunBudget`, and the **deterministic
   aggregation function** (section H.0). Pure dataclasses + markdown rendering; no I/O.
3. **Sanitisation, capture, and evidence:** the in-process `OutputCaptureBoundary`, `sanitize(text)`
   (existing `redact_secrets` patterns **plus** the literal runtime credential value and the declared
   `credential_scope` env values), `scan_for_secrets(text)` returning matched pattern *names only*,
   and `write_evidence(...)` which sanitises, scans, and raises `SecretSafetyError` (→
   `SECRET_SAFETY_FAILURE`) on any hit. Evidence is written **only** here (sections N/S).
4. **Gate probes** (section H), each `(target, case|None, budget) -> GateResult`, plus the
   `CertificationTarget` protocol, the `ProductionTarget` adapter (read-only imports), the
   `RecordingModelProxy`, and the **human confirmation step** (D.3).
5. **Runner:** GLOBAL gate 1 once → per-case loop (gates 2 → confirmation → 3–8 → 10) → GLOBAL
   offline gate 9 → deterministic aggregation → sanitized evidence artifact outside the repo →
   markdown record draft; budget-checked, fail-fast on gate 1, **never** edits registry status.
6. **CLI:** `python -m src.certification --provider … --model … [--live] [--cases A,B,C] [--out PATH]`.
   Without `--live` it refuses to construct a real model (exit 2) and only validates the combination,
   the budgets, and the record skeleton.
7. **Offline tests** (section R).

## G. P3.2B — real-provider certification (DeepSeek + `deepseek-flash`)

Executed only after P3.2A is reviewed and merged by the human, and only with a human-held credential
(section S). The **human** performs the interactive category confirmation during the run; it is
recorded in the evidence. Outputs: one sanitized evidence artifact (outside the Git project) plus a
certification record under `docs/certification/` created only after human review and only if
secret-free. No source change is part of P3.2B; if a source change turns out to be necessary, P3.2B
**stops** and asks for approval.

## H. Exact 10-gate test methods

Common PASS rule: a gate result is `PASS` only if its stated criteria all hold. Anything else is
`FAIL`, `BLOCKED` (precondition unmet), or `NOT_RUN`. Evidence for every result is recorded verbatim
(sanitized) in the evidence artifact.

### H.0 Gate scoping and aggregation

| Scope | Gates | How often |
|---|---|---|
| **GLOBAL** (per certification run) | **Gate 1** Authentication / minimal request | **exactly once**, before any case |
| **GLOBAL** (offline, mandatory) | **Gate 9** Safe error handling (failure injection) | **exactly once** per certification run, offline; **never repeated per live case** |
| **CASE-SCOPED** | **Gates 2–8** and **Gate 10** | once per applicable case (A, B, C) |

`GateResult` carries a `case_id` / `scope` field: `GLOBAL` or `case:A` / `case:B` / `case:C` (and any
future case id). The record preserves **per-case** results in a Gate × Case matrix.

**Aggregation rule (deterministic, no judgement):**

1. A `GLOBAL` gate's status is its single result.
2. A case-scoped gate's aggregate status is derived from its per-case results, in this precedence:
   any `FAIL` → `FAIL`; else any `BLOCKED` → `BLOCKED`; else `NOT_RUN` **only if every applicable
   case is `NOT_RUN` and this plan declares that gate/case combination not applicable**; else `PASS`
   only if every applicable case is `PASS`; otherwise `NOT_RUN`.
3. **No case failure may be hidden by another case passing.** A case-scoped gate never aggregates to
   `PASS` while any applicable case is `FAIL` or `BLOCKED`.
4. Any required case-gate `FAIL` or `BLOCKED` prevents overall `PASS`.
5. `NOT_RUN` is accepted **only** where this plan explicitly declares the gate/case combination not
   applicable: gates 3–8 for Case C (an unsupported category intentionally never runs the agent tool
   path) and the optional live auth probe.
6. The record shows both the ten-gate aggregate table **and** the Gate × Case matrix; the aggregate is
   always derivable from the matrix.

Per-case order: `gate 2 → human confirmation (D.3) → gates 3–8 → gate 10`. Gate 1 precedes all cases;
gate 9 is offline and case-independent.

### Gate 1 — Authentication / minimal request (**GLOBAL — once per run**)
- **Method:** run **once per exact Provider + Model certification run, before any case**. Construct
  the model through the **production dev path** `build_model()`; run one minimal Strands call — one
  user message (`"Reply with the single word: ok"`), **no tools**, no structured output,
  `limits={"turns": 1, "output_tokens": 20, "total_tokens": 200}`.
- **Records:** provider class name, endpoint *strategy* (`descriptor.base_url`), exact model id, HTTP
  outcome if surfaced by the SDK exception, response length, elapsed ms. **Never** the credential.
- **PASS:** a non-empty assistant response with no exception.
- **FAIL mapping:** 401/403 → `AUTH_FAILURE`; 404 / "model not found" → `MODEL_NOT_AVAILABLE`;
  400 / schema-shaped rejection → `REQUEST_SCHEMA_INCOMPATIBLE`; timeout → `TIMEOUT`; else
  `UNKNOWN_PROVIDER_FAILURE`. Gate 1 failing stops the run (fail-fast) — cases are not attempted.

### Gate 2 — Structured classification (**case-scoped**) — a certification gate, not a category decision
- **Method:** the production chain only — `AgentClassifier(lambda d: _suggest_category(model, d, allowed), allowed)`,
  `allowed = allowed_category_values(repository)`. **No second classifier is created.**
- **PASS (Case A):** `category == "small_consumer_electronics"`, `category_source == agent_generated`,
  `category_status == REVIEW_REQUIRED`.
- **PASS (Case B):** **identical criteria to Case A** — `category == "small_consumer_electronics"`,
  `category_source == agent_generated`, `category_status == REVIEW_REQUIRED`. Case B omits one USER
  `ProductFact`, but missing product facts affect later **applicability**, not the product-**category**
  suggestion; gate 2 must not be weakened or reinterpreted for Case B.
- **PASS (Case C):** `NEEDS_INFO` (category `None`) or `REVIEW_REQUIRED` with a value inside the
  allowed vocabulary; an invented/out-of-vocabulary value must have been downgraded to `NEEDS_INFO` by
  `agent_suggestion()`.
- **Hard rule — no silent promotion:** the agent-generated result is recorded verbatim and is
  **never** promoted to `RESOLVED`. The certification layer must not write to it, copy it into a
  canonical object, or construct a `CategoryResult` of its own.
- **FAIL mapping:** `ClassificationRuntimeError` / `structured_output is None` →
  `STRUCTURED_OUTPUT_FAILURE`; a valid `NEEDS_INFO` decline in **Case A or Case B** →
  `CLASSIFICATION_DECLINE` (gate-2 FAIL **for that case**; gates 3–8 then `BLOCKED` for that case).
  Case C keeps its existing safe `NEEDS_INFO`/unsupported behaviour and is **not** a
  `CLASSIFICATION_DECLINE`.

### H.2 Human category confirmation step (case-scoped; between gate 2 and gate 3)
- **Method:** after gate 2 passes for a case, the runner presents the recorded suggestion and the
  allowed vocabulary and asks the human to **confirm or correct** it — mirroring the product's existing
  reviewer action `confirm_or_correct_category`.
- The accepted value is passed **only** as `provided_category` into the existing boundary:
  `Orchestrator.run(description, provided_category=<accepted>, product_facts=…, agent_runner=…)` with
  `HumanClassifier(allowed)` as the orchestrator's classifier. The resulting `CategoryResult`
  (`human_confirmed`; `RESOLVED` for a supported value, `UNSUPPORTED` for `unsupported`, `NEEDS_INFO`
  for `uncertain`) is what gates 3–8 consume for that case.
- **Never automatic:** if the human declines, or the value is not in the vocabulary, the case records
  `CATEGORY_NOT_CONFIRMED` and is `BLOCKED`; gates 3–8 do not run for that case.
- **Evidence:** suggested value + source + status, the confirmation event (interactive, human,
  timestamp, accepted value), and the confirmed `CategoryResult`. In P3.2A the confirmation is injected
  through the target's `confirm_category` callable (default: interactive prompt; tests: a deterministic
  stub).
- **Constraint:** this step adds product-semantics fidelity, not a second authority — it can only
  produce what `HumanClassifier` can produce. It costs **no** provider request.

### Gate 3 — Tool schema acceptance (case-scoped; semantic, not wire-format)
- **Method:** during the case's agent run, observe the **tool surface exposed to the provider** and
  whether the provider accepted the request carrying it.
- **PASS (semantic):**
  - exactly the **two** ImportReady Agent tools are exposed — no more, no fewer;
  - names are exactly `analyze_product` and `get_compliance_evidence`;
  - logical input contracts are exactly `product_description` (one argument; no facts/attribute/
    origin/rule_id inputs) and `rule_id` (one argument);
  - the provider **accepted** the request carrying them (no 4xx schema rejection, no SDK schema error);
  - no schema mutation was required.
- **Explicitly NOT required:** one exact provider/Strands serialized JSON wrapper or layout. The gate
  is semantic; recorded evidence is the observed tool names + argument names + acceptance outcome, not
  an asserted raw payload shape.
- **FAIL mapping:** `TOOL_SCHEMA_REJECTED` / `REQUEST_SCHEMA_INCOMPATIBLE`.
- **No new Agent tool is added for certification.**

### Gate 4 — Tool selection (case-scoped)
- **Method:** the model itself must have emitted a real tool-use event naming `analyze_product`
  (observed through the proxy), corroborated by `AgentRunOutcome.tool_calls[0] == "analyze_product"`.
- **PASS:** an actual `analyze_product` tool-use event is observed in the model's output.
- **Explicit FAIL:** prose such as "I will call analyze_product" with **no** observed tool-use event →
  `TOOL_SELECTION_FAILURE`. Text claims never certify this gate.
- **Primary case:** Case A (Bluetooth wireless earphones with charging case).

### Gate 5 — Tool execution (case-scoped)
- **Method:** after confirmation, the tool runs through the normal application path (`build_tools`
  bound to the confirmed category + USER facts). PASS requires
  `AgentRunOutcome.analysis_result is not None`, structurally valid
  (`AnalysisResult.model_validate`), and identical (after normalisation) to the deterministic result
  computed independently by `AnalysisService.analyze(confirmed_category_result, facts)` for the same
  inputs.
- **PASS:** the canonical result came from the tool, not from model prose; canonical applicability is
  present (which the confirmation step makes possible).
- **FAIL:** `TOOL_EXECUTION_FAILURE` (not executed, `analysis_tool_failed`, no canonical applicability,
  or mismatch — the latter also fails gate 10).
- No certification-only compliance path is built.

### Gate 6 — Tool-result continuation (case-scoped; semantic, not wire-format)
- **Method:** verify the **semantic** continuation sequence for the same run:
  1. a real `analyze_product` tool-use event occurred (gate 4);
  2. the tool result for that event was **incorporated into the next provider/model turn** — the
     provider accepted the continuation and the model's next turn proceeded with the tool result
     present among the messages it received;
  3. the model produced a **subsequent assistant continuation/final response**.
- **PASS:** all three semantic facts hold.
- **Explicitly NOT required:** a literal `assistant → tool → assistant` role/block serialization, or
  any one provider-specific representation, when Strands expresses the same semantics differently.
- **Records:** the observed semantic sequence, counts, elapsed ms, and **every** warning/error line
  emitted during this turn, verbatim and sanitized.
- **FAIL:** `TOOL_RESULT_CONTINUATION_FAILURE` (continuation rejected, or the model never continues).
- **Rule:** the proxy is observational only. No request/history mutation, and no field may be stripped
  to force a PASS — that is a production-behaviour change requiring separate approval.

### Gate 7 — Multi-turn completion (case-scoped)
- **Method:** the run must finish within the existing `_AGENT_LIMITS` (`turns: 6`) with a **normal**
  stop reason (`end_turn`/`stop_sequence`); turns are counted from the proxy. Warnings are collected by
  the in-process capture boundary (`warnings.catch_warnings(record=True)` + logging handler + buffered
  stdout/stderr — section N).
- **PASS:** normal stop within the limit + a non-empty final answer + all downstream invariants pass
  **and either**:
  - **(a) the `reasoningContent` warning was NOT observed** → record `warning_status = NOT_OBSERVED`
    (this alone never fails gate 7), **or**
  - **(b) it was observed and classified A or B**:
    - **A** — warning only; the final turn succeeds and all downstream gates pass unchanged;
    - **B** — degraded but still valid behaviour (e.g. extra turns, shortened prose) with all canonical
      invariants intact.
- **If the warning IS observed, classification is mandatory** (a warning is never ignored):
  - **C** — the warning prevents valid multi-turn completion → gate `FAIL`
    (`MULTI_TURN_INCOMPATIBLE`);
  - **D** — a workaround would require provider-specific request transformation → gate `BLOCKED`;
    **stop certification** and request human approval before touching production provider behaviour.
- **The warning is an observation, not a requirement**: certification must not force it to reproduce,
  must not inject prompts/parameters to trigger it, and must not treat its absence as a defect. It is
  recorded verbatim when present; it is never suppressed, and no thinking/`extra_body` parameter is
  reintroduced.

### Gate 8 — Final response generation (case-scoped)
- **Method:** take the final assistant text from the same run and check it against the canonical
  structured result. Automated checks:
  - **A1** non-empty (already required by `_make_agent_runner`, re-asserted here);
  - **A2** no traceback/exception markers (`Traceback (most recent call last)`, `File "…", line`, raw
    exception class names) and no `ERROR [` line;
  - **A3** no secret pattern (`scan_for_secrets`) and no credential literal;
  - **A4** no contradiction of a canonical APPLICABLE rule: if canonical R-ELEC-002 is `APPLICABLE`,
    the prose (sentence-scoped) must not say that rule "does not apply"/"is not applicable";
  - **A5** lifecycle safety, sentence-scoped: a sentence mentioning a rule id whose canonical
    `rule_status` is `PROPOSED`/`WATCHLIST`/`SUPERSEDED`, or whose canonical `applicability_status` is
    `NEEDS_INFO`/`REVIEW_REQUIRED`, must not contain a definite-current-obligation phrase
    ("must comply", "is required", "is mandatory", "currently effective", "is the law"); and the prose
    must not present `REVIEW_REQUIRED` as compliant, nor `NOT_FOUND` as "not required".
- **PASS:** A1–A5 pass **and** the human prose review in the record is signed off. Prose quality itself
  is a **human** judgement, recorded as such.
- **FAIL mapping:** `FINAL_RESPONSE_FAILURE` / `SECRET_SAFETY_FAILURE`.
- **Lifecycle coverage is not left to live prose.** Whether the live model happens to *mention* a
  `PROPOSED`/`WATCHLIST` rule is not controllable, so A5's mandatory evidence comes from the
  **offline adversarial prose tests** (section R.14) and from the **live gate-10 canonical check**
  (below). A5 is still applied to whatever prose the live run produces.
- Case C: gates 3–8 are `NOT_RUN` for Case C by design (an `UNSUPPORTED` category intentionally never
  runs the agent tool path — section I); its safe-outcome assertions replace them.

### Gate 9 — Safe error handling (**GLOBAL — mandatory offline, once per run**)
- **Method (mandatory, offline, case-independent):** with an injected fake target / fake model, force
  each of: a generic exception in the agent runner, a timeout-shaped exception, a malformed provider
  response (missing/None structured output), and a provider exception whose message *contains* a
  sentinel credential. This gate is executed **once per certification run** as a GLOBAL gate and is
  **not repeated per live case**. Assert for each:
  - the user-facing error is the sanitized type (`agent_runtime_failure` / `agent_timeout` /
    `classification_failed`) with the existing safe message and **no** raw exception text;
  - no secret appears in stdout, stderr, the evidence text, or the record (`SECRET_SAFETY_FAILURE`
    otherwise);
  - no traceback reaches consumer-visible output;
  - no invented compliance result: the fallback is the deterministic result marked explicitly
    `REVIEW_REQUIRED` with the trigger recorded;
  - no stale reuse: a result from a previous case is never returned (per-request
    `ToolExecutionState`, fresh `Orchestrator.run`);
  - exit code is the existing failure code (`1`, or `4` for classification failure) — never `0`.
- **Optional live probe (separate, human-run, `NOT_RUN` by default):** one request with a deliberately
  invalid credential → gate 1 classification must yield `AUTH_FAILURE`. It is never part of the
  offline suite, is not a live case, and costs at most one extra request if the human chooses it.

### Gate 10 — Canonical result preservation (case-scoped)
- **Method:** for every case, compare the agent path against the offline deterministic path with
  **identical** description, confirmed category and USER facts. Normalise away non-canonical
  differences (`request_id`, timestamps, `agent_suggestions`) and compare the `applicability` block,
  the `classification` block and `review.status`.
- **Required invariant:** the canonical `ApplicabilityResult` matches per rule: `rule_id`,
  `applicability_status`, `reason_codes`, `missing_attribute_ids`, `evaluated_attribute_ids`.
- **Explicitly verified:** R-ELEC-002 — all 7 required attributes still mandatory;
  `A-ELEC-002=True` → `APPLICABLE` (+ `TRIGGER_SATISFIED`), `A-ELEC-002=False` →
  `NOT_APPLICABLE` (+ `TRIGGER_NOT_SATISFIED`); omitting a required attribute → `REVIEW_REQUIRED` with
  `MISSING_REQUIRED_FACTS`.
- **Lifecycle invariant (live):** in the agent-path canonical result,
  `analysis.verified.compliance_information` must still report `R-ELEC-018` with
  `rule_status == "WATCHLIST"` and `R-ELEC-019` with `rule_status == "PROPOSED"`, neither as
  `EFFECTIVE`, and both must be identical to the offline deterministic result. Model output cannot
  change these canonical statuses.
- **Also verified:** missing facts stay missing (Case B); model prose cannot overwrite the canonical
  result (prose only ever lands in `analysis.agent_suggestions`); an agent-generated category never
  unlocks canonical applicability by itself (the confirmation boundary is what does).
- **`NOT_FOUND` invariant:** verified against whichever approved rule carries
  `evidence_status=NOT_FOUND`, discovered at run time; if the approved data contains no such rule, the
  item is recorded `NOT_RUN` with that reason, and the deterministic engine's existing tests remain the
  authority.
- **Offline adversarial variant (P3.2A suite):** a fake agent runner returns canonical data plus
  contradictory prose → the canonical block must be identical to the offline block.
- **FAIL:** `CANONICAL_RESULT_MISMATCH`.

## I. Certification scenarios (3 live cases — deliberately small)

Fixtures are **discovered from approved data at run time, never hardcoded**: the value for each
required attribute is derived from its approved definition (`data_type` / `allowed_values`) using one
small helper that mirrors the existing verification helper's logic. Compliance JSON is not modified.

Per-case flow: `gate 2 (agent suggestion, REVIEW_REQUIRED) → human confirmation (D.3) → gates 3–8 →
gate 10`. Gate 1 (GLOBAL) precedes all cases; gate 9 (GLOBAL, offline) is case-independent.

| Case | Description (input) | USER facts | Purpose |
|---|---|---|---|
| **A** | "Bluetooth wireless earphones with charging case" | all 7 R-ELEC-002 attributes, `A-ELEC-002=True` | classification, human confirmation, tool call, execution, canonical positive applicability, final explanation |
| **B** | same product | same 7 minus `A-ELEC-003` | `REVIEW_REQUIRED`/`NEEDS_INFO` preserved; the model must not fill the gap from common sense |
| **C** | a description clearly outside the two supported categories (e.g. "bulk ground black pepper spice blend, 1 kg bag") | none | no invented verdict; safe unsupported outcome |

Case notes:

- **Live certification cases are A, B and C only.** There is deliberately **no Case D**: a case that
  repeats Case A's description and facts cannot force the live model to mention `R-ELEC-018`/
  `R-ELEC-019`, so it would add no reliable live evidence.
- Cases A and B are run **without** `--category` so the model classifier (gate 2) drives
  classification, followed by the explicit human confirmation step. The deterministic offline
  comparison for gate 10 uses that same confirmed category plus identical facts.
- **Case C**: after the model declines or proposes an unsupported/out-of-vocabulary value, the human
  confirmation resolves it to `unsupported`/`uncertain`. The orchestrator then intentionally does not
  run the agent tool path; the case asserts the safe outcome instead — `category_source` and
  `category_status` unchanged by the certification layer, `applicability is None`, `review.status` in
  {`UNSUPPORTED`, `NEEDS_INFO`}, `verified.compliance_information` empty, no invented requirement, no
  APPLICABLE verdict, exit code `0`. Gates 3–8 for Case C are recorded `NOT_RUN` with that reason
  (never rounded to PASS).
- **Lifecycle safety coverage (mandatory, no live Case D):**
  - **Live:** gate 10 asserts the canonical `R-ELEC-018` (`WATCHLIST`) and `R-ELEC-019` (`PROPOSED`)
    statuses are unchanged and never `EFFECTIVE`/canonical obligations, in both paths.
  - **Offline (mandatory):** the P3.2A Gate-8 adversarial prose tests — a definite-current-obligation
    statement about a `PROPOSED`/`WATCHLIST` rule must **FAIL**, correct wording such as "proposed, not
    yet effective" must **PASS**.
  - A future live lifecycle-*explanation* probe would need a distinct observable purpose and separate
    human approval. `product_description` must never be contaminated with certification-only
    instructions, and no Agent tool may be added or changed for certification.
- Each case is executed **once**. No case is executed twice to "try again".
- The confirmation step costs **no** provider request.

## J. PASS / FAIL criteria

- **Gate PASS:** every stated criterion of section H holds, with recorded evidence.
- **Gate FAIL:** a criterion fails for the gate's scope, with a failure category from section K.
- **Gate BLOCKED:** a precondition was unmet (an earlier mandatory gate failed, the human did not
  confirm the category, or an observation channel cannot be safely captured). `BLOCKED` is never
  rounded up to PASS.
- **Gate NOT_RUN:** deliberately skipped, only where section H.0 declares the gate/case combination not
  applicable (gates 3–8 for Case C; the optional live auth probe).
- **Aggregation:** exactly as defined in section H.0 — deterministic, per gate, preserving per-case
  results; no case failure is hidden by another case passing.
- **Overall PASS (recommendation for promotion):** every gate aggregates to `PASS` (with only the
  declared `NOT_RUN` exceptions) **and** all invariants pass **and** no unresolved safety-critical
  compatibility issue remains **and** the record exists with a human sign-off **and** gate 7 passed
  under either branch (a) `warning_status = NOT_OBSERVED` or (b) warning observed and classified A/B
  (never C/D, and never forced to reproduce).
- **Overall FAILED:** any mandatory gate `FAIL` or `BLOCKED` in any applicable case, or a GLOBAL gate
  `FAIL`/`BLOCKED`.
- **Overall INCOMPLETE:** budget/wall-clock exhaustion, or the run was interrupted.
- Partial failures are always visible in the per-gate table and the Gate × Case matrix; an overall
  `PASS` never hides a non-`PASS` result.

## K. Failure taxonomy

| Category | Raised by | Meaning |
|---|---|---|
| `AUTH_FAILURE` | gate 1 (global) | credential rejected (401/403) |
| `MODEL_NOT_AVAILABLE` | gate 1 (global) | endpoint/model missing or retired (404) |
| `REQUEST_SCHEMA_INCOMPATIBLE` | gates 1, 3 | request body/tools rejected (400) |
| `STRUCTURED_OUTPUT_FAILURE` | gate 2 | no/invalid structured classification output |
| `CLASSIFICATION_DECLINE` | gate 2 (Case A or Case B) | valid `NEEDS_INFO` decline where a category was required |
| `CATEGORY_NOT_CONFIRMED` | confirmation step | the human declined, or no valid confirmed category exists |
| `CONFIGURATION_MISMATCH` | live pre-flight binding | CLI `--provider`/`--model` ≠ env `MODEL_PROVIDER`/`MODEL_ID`, a value missing, or the combination not registered as intended — refused **before** model construction, 0 provider requests |
| `EVIDENCE_PATH_REJECTED` | live pre-flight path validation | the resolved evidence path is inside the Git repository root (fail-closed, before any write) |
| `TOOL_SCHEMA_REJECTED` | gate 3 | tool surface specifically refused |
| `TOOL_SELECTION_FAILURE` | gate 4 | no real `analyze_product` tool-use event |
| `TOOL_EXECUTION_FAILURE` | gate 5 | tool ran but produced no valid canonical result |
| `TOOL_RESULT_CONTINUATION_FAILURE` | gate 6 | tool result not incorporated / no continuation |
| `MULTI_TURN_INCOMPATIBLE` | gate 7 | no valid completion within the turn limit |
| `FINAL_RESPONSE_FAILURE` | gate 8 | empty/contradictory/unsafe final answer |
| `CANONICAL_RESULT_MISMATCH` | gates 5, 10 | deterministic result not preserved |
| `SECRET_SAFETY_FAILURE` | gates 8, 9, evidence scan, capture boundary | credential material detected anywhere |
| `OUTPUT_CAPTURE_UNAVAILABLE` | step-0 inspection / capture boundary | an output channel cannot be captured in-process → BLOCKED |
| `TIMEOUT` | gates 1, 6, 7, 9 | timeout-shaped provider/transport failure |
| `BUDGET_EXHAUSTED` | runner | request/turn/wall-clock cap reached |
| `RATE_LIMITED` | gate 1 (global) | provider 429 |
| `UNKNOWN_PROVIDER_FAILURE` | any | unrecognised provider failure, recorded verbatim (sanitized) |

Manual overrides are not allowed: a category is either produced by a probe or recorded as
`UNKNOWN_PROVIDER_FAILURE` with the raw (sanitized) provider text attached as evidence.

## L. Provider/model status-promotion process

`EXPERIMENTAL` → `VERIFIED` only when **all** of these hold:

1. every gate aggregates to `PASS` under section H.0 (with only the declared `NOT_RUN` exceptions);
2. all ImportReady invariants `PASS` (gate 10 + the invariant list);
3. no unresolved safety-critical compatibility issue remains;
4. a certification record exists at `docs/certification/<provider_id>__<model_id>.md`;
5. an explicit human sign-off approves promotion.

There is **no automatic promotion**. The certification runner must not edit `PROVIDER_REGISTRY`,
`COMBINATION_INDEX`, or any `ModelCombination`. After a PASS:

```
human reviews evidence
  → separate, approved code change (one-line status edit + certification_ref)
  → ModelCombination.compatibility_status = VERIFIED
  → ui_exposed stays False until a SECOND, separate decision explicitly enables it
```

`VERIFIED` never implies `ui_exposed=True`, and `verified_combinations()` stays empty until that
second decision is made.

## M. Certification record format

- **Record** (committed artifact, created only at certification time, only after human review, and only
  if secret-free): `docs/certification/<provider_id>__<model_id>.md`.
- **Raw evidence artifact** (detailed): stored **outside the Git project** under
  `..\ImportReady_AI_Certification_Artifacts\` (sections N/S); it is **not** required to be committed.
- The record references the local evidence artifact by **filename + SHA-256 + summary**, rather than
  embedding the raw artifact.
- Per-case results are **never** collapsed: the record keeps the Gate × Case matrix next to the
  aggregate table.

```markdown
# Certification record — <provider_id> + <model_id>

| Field | Value |
|---|---|
| provider_id | deepseek |
| model_id | deepseek-flash |
| endpoint strategy | https://api.deepseek.com  (ImportReady-owned descriptor base_url) |
| certification date/time | <ISO-8601 UTC> |
| strands-agents version | 1.55.1 |
| provider SDK version | openai 2.54.0 |
| ImportReady commit SHA | <git rev-parse HEAD> |
| cases run | A, B, C  (live); gate 1 + gate 9 global |
| validated target | CLI `--provider`/`--model` == env `MODEL_PROVIDER`/`MODEL_ID` == the resolved registered combination (verified **before** model construction); this resolved exact combination is the one the record describes |
| evidence artifact | <filename> (outside repo; e.g. ..\ImportReady_AI_Certification_Artifacts\deepseek__deepseek-flash__<UTCstamp>.txt) |
| evidence SHA-256 | <hash> |
| evidence summary | requests used=<n>, wall clock=<s> |
| overall status | PASS / FAILED / INCOMPLETE  (never hides a non-PASS result) |

## Gates (aggregate, derived deterministically from the matrix below)
| # | Gate | Scope | Aggregate | Failure category |
|---|---|---|---|---|
| 1 | Authentication / minimal request | GLOBAL | PASS/FAIL | … |
| 2 | Structured classification | case A/B/C | … | … |
| 3 | Tool schema acceptance | case A/B/C | … | … |
| 4 | Tool selection | case A/B/C | … | … |
| 5 | Tool execution | case A/B/C | … | … |
| 6 | Tool-result continuation | case A/B/C | … | … |
| 7 | Multi-turn completion | case A/B/C | … | … |
| 8 | Final response generation | case A/B/C | … | … |
| 9 | Safe error handling (offline injection) | GLOBAL | … | … |
| 10 | Canonical result preservation | case A/B/C | … | … |

## Gate × Case matrix (source of truth; aggregate above is derived from this)
| Gate | GLOBAL | Case A | Case B | Case C |
|---|---|---|---|---|
| 1 | PASS/FAIL | — | — | — |
| 2 | — | PASS/FAIL | PASS/FAIL | PASS/FAIL |
| 3 | — | … | … | NOT_RUN (unsupported category) |
| 4 | — | … | … | NOT_RUN (unsupported category) |
| 5 | — | … | … | NOT_RUN (unsupported category) |
| 6 | — | … | … | NOT_RUN (unsupported category) |
| 7 | — | … | … | NOT_RUN (unsupported category) |
| 8 | — | … | … | NOT_RUN (unsupported category) |
| 9 | PASS/FAIL/BLOCKED | — | — | — |
| 10 | — | … | … | PASS/FAIL (safe unsupported outcome asserted) |

## Category confirmation (Cases A, B, C) — recorded separately per case
| case_id | agent suggestion (verbatim) | suggestion source/status | human action (confirm / correct / decline) | accepted value | HumanClassifier result | final category source/status |
|---|---|---|---|---|---|---|
| A | category=… | agent_generated / REVIEW_REQUIRED | confirm / correct / decline | … | category=…, source=human_confirmed, status=… | … |
| B | category=… | agent_generated / REVIEW_REQUIRED | confirm / correct / decline | … | category=…, source=human_confirmed, status=… | … |
| C | category=… (or none) | agent_generated / REVIEW_REQUIRED or NEEDS_INFO | confirm / correct / decline | `unsupported` or `uncertain` (either is a valid human resolution) | category=…, source=human_confirmed, status=UNSUPPORTED or NEEDS_INFO | … |

- `HumanClassifier` remains the **only** confirmation boundary; the certification layer records the
  event but never constructs a `CategoryResult`, never sets `human_confirmed`, and never invents a
  second category authority.
- For Case C, the human legitimately resolves the case to `unsupported` **or** `uncertain`; both are
  recorded verbatim, and the resulting `status` (UNSUPPORTED / NEEDS_INFO) is preserved as-is.

## ImportReady invariants
| Invariant | Status | Evidence |
|---|---|---|
| R-ELEC-002 canonical behaviour (7 attributes) | … | … |
| R-ELEC-018 stays WATCHLIST / R-ELEC-019 stays PROPOSED (never EFFECTIVE) | … | … |
| agent suggestion never promoted to RESOLVED by certification code | … | … |
| canonical applicability requires human confirmation | … | … |
| PROPOSED ≠ EFFECTIVE | … | … |
| WATCHLIST ≠ EFFECTIVE | … | … |
| NOT_FOUND ≠ "not required" | … | … |
| missing facts stay missing | … | … |
| model prose cannot overwrite canonical result | … | … |
| `deepseek`+`deepseek-flash` status unchanged by this run | … | … |

## Warnings observed (verbatim, sanitized)
| Field | Value |
|---|---|
| warning_status | OBSERVED / NOT_OBSERVED |
| reasoningContent warning text (when observed) | `reasoningContent is not supported in multi-turn conversations with the Chat Completions API.` |
| classification (when observed) | A / B / C / D + justification |
| effect on gate 7 | PASS (NOT_OBSERVED, or class A/B) / FAIL (class C) / BLOCKED (class D) |

- Absence of the warning is a **valid outcome** and does not fail gate 7. The warning is never forced
  to reproduce, and its absence is never recorded as a defect.

## Unresolved compatibility issues
- …

## Budget
cases=A,B,C; gate 1 = 1 request; live provider requests used=<n> (cap 24); wall clock=<s> (cap 600 s)

## Recommendation
- e.g. "recommend promotion to VERIFIED" / "remain EXPERIMENTAL — gate 7 blocked for Case A"

## Human sign-off
- reviewer: ______  date: ______  decision: approve / reject / defer
- confirmation decisions made during the run: ______
- ui_exposed decision (separate): not decided here
```

The record must contain **no** secret material of any kind (capture → sanitize → scan before either
file is written; the record is additionally rescanned before it is created).

## N. Secret handling

Rules:

1. The real key is provided **only** in the human's own PowerShell session, only for the certification
   run, and is removed immediately afterwards (section S). It is never written to `.env`, a test file,
   a source file, a command in a chat/tool call, a report, or Git.
2. The runner reads the credential only through the existing dev path (`build_model()` reading
   `API_KEY` from the process environment). No new credential plumbing, no `os.environ` mutation.
3. **In-process capture boundary (implementable policy — replaces any claim that a sanitizer controls
   a channel it cannot see):**

   ```
   provider / Strands execution
     → temporary in-process capture
         · sys.stdout / sys.stderr replaced by in-memory buffers
         · warnings.catch_warnings(record=True)
         · logging handler attached (root, importready, provider SDK loggers)
     → sanitize(captured_text)
     → scan_for_secrets(captured_text)
     → emit only the sanitized minimal summary / evidence
     → discard the raw in-memory buffer (restore the original streams in a `finally`)
   ```

   Requirements:
   - raw captured output is **never** written to disk;
   - raw captured output is **never** re-emitted to the terminal;
   - shell-level `Tee-Object` / redirection remains **prohibited** as a capture mechanism;
   - evidence files still go through the sanitizing writer only;
   - the capture boundary is enabled only for the duration of a provider call and is always restored;
   - if the step-0 inspection discovers an output channel that cannot be safely captured/suppressed
     **without changing production behaviour** (for example a component writing straight to the OS
     file descriptor), the affected certification observation is marked `BLOCKED`
     (`OUTPUT_CAPTURE_UNAVAILABLE`) and the run **stops for human approval** — the plan does not
     pretend the sanitizer covers that channel.
4. **Layered sanitisation:** every string that can reach the terminal, the evidence artifact, or the
   record passes through `sanitize()`, which applies the existing `redact_secrets()` patterns **plus**
   the literal credential value read from the provider's declared `credential_scope` **plus** the
   values of those env vars if still present. `scan_for_secrets()` runs on the final evidence text and
   record text and reports pattern *names* only, never matched values. A hit raises `SecretSafetyError`
   → run marked `SECRET_SAFETY_FAILURE`, non-zero exit, artifact not written (or deleted if written).
5. The record and evidence never contain `Authorization` headers or request headers at all: the proxy
   records only message roles, tool names/logical argument names, token counts, stop reasons and
   timings.
6. `provider_status_report()` / `verified_combinations()` are never given credential values (already
   true in P3.1; re-asserted in P3.2A tests).
7. Provider A's credential is never handed to provider B (P3.1 `credential_scope`; re-asserted).
8. If a key ever reaches a file or a log: delete the artifact, treat it as `SECRET_SAFETY_FAILURE`,
   rotate the key, and record the incident in the report.
9. **Live pre-flight target binding (before any model construction or network request).** The live CLI
   names the exact combination (`--provider`, `--model`), but `build_model()` reads `MODEL_PROVIDER`
   and `MODEL_ID` from the environment. The runner must therefore verify, as its first live action:
   - `--provider` == `MODEL_PROVIDER` (exact, after the same normalization `build_model()` applies) and
     `--model` == `MODEL_ID`;
   - the pair resolves through `resolve_combination()` to the intended registered combination (and its
     status is one the **developer/certification path** permits: `EXPERIMENTAL` or `VERIFIED`).
   If either environment value is **absent** or **differs**, or the combination is unknown/
   `UNSUPPORTED`: **refuse the live run before any model is constructed**, make **no** provider
   request, and return a clear **sanitized** configuration error (`CONFIGURATION_MISMATCH`, exit code
   non-zero). The runner must **never** mutate `os.environ` (or otherwise rewrite the environment) to
   make the values match. The certification record states the exact combination that was validated
   before the run.
10. **Evidence-path confinement.** Detailed/raw certification evidence must never be written inside the
    Git repository. Resolution and validation happen **before any write**, fail-closed:
    - the default location is the sibling directory
      `<repo-parent>\ImportReady_AI_Certification_Artifacts\`;
    - any `--out` path is resolved/canonicalized first (absolute resolution including symlink/parent
      resolution where the platform allows it);
    - the resolved path **must be outside the repository root**; a path inside the repo — whether given
      directly, as a repo subdirectory, or as a relative `..` path that resolves back into the repo —
      is refused with `EVIDENCE_PATH_REJECTED` and **nothing is written**;
    - if the path cannot be resolved reliably (e.g. resolution failure ambiguity), the run fails closed;
    - only the reviewed, secret-free human-readable record may later live under `docs/certification/`.

## O. Request / turn / cost limits

Reuse existing application limits; add only runner-level budgets.

| Limit | Value | Source |
|---|---|---|
| Agent turns per case | 6 | existing `_AGENT_LIMITS` |
| Agent output tokens / total tokens per run | 1200 / 10000 | existing `_AGENT_LIMITS` |
| Tool calls per request | 4 | existing `MAX_TOOL_CALLS` |
| Classification turns / tokens | 2 / 4000 | existing classification limits |
| Live pre-flight (no network) | CLI/env binding check + evidence-path validation; **0 provider requests**; a refusal exits before model construction | this plan |
| **Gate 1 (GLOBAL)** | **1 live provider request, run once per certification run** | this plan |
| Live certification cases | **3 (A, B, C), each executed once** | this plan |
| Live provider requests per case | ≤ 8 (1 classification + ≤ 6 agent turns + 1 slack); Case C ≤ 2 | this plan |
| Live provider requests per run | **hard cap 24** (1 for gate 1 + ≤ 8 for A + ≤ 8 for B + ≤ 2 for C, with slack); on reaching it the run stops and remaining results are `BLOCKED`/`BUDGET_EXHAUSTED` | this plan |
| Human confirmation prompts | 1 per case (interactive; **0 provider requests**) | this plan |
| Gate 9 | offline injection, **0 provider requests** (the optional live auth probe = ≤ 1) | this plan |
| Automatic retries | **0** | this plan (a failure is evidence, not a retry trigger) |
| Concurrency | 1 (strictly sequential) | this plan |
| Soft wall-clock per case / per run | 120 s / 600 s, checked between steps | this plan |
| Timeout behaviour | an in-flight provider request is **not** force-killed (no SDK timeout is injected at P3.2, since that would be a production behaviour change); a hung request is aborted by the human (Ctrl+C) and recorded as `TIMEOUT`/`INCOMPLETE` | this plan |

This is compatibility certification, **not** load or stress testing. No batch/parallel/large-N runs.

## P. Proposed file boundary — smallest structure, decided after inspection

### Step 0 of P3.2A: interface inspection, then structure choice

P3.2A **starts** with the read-only inspection of the installed Strands interface (section F.1):
`strands.models.model.Model` (what a transparent proxy must delegate), `AgentResult` (which attributes
exist in 1.55.1; how tool-use/tool-result/continuation/turns are represented), whether the `Agent`
exposes its registered tool specs readably, and **every output channel** used during a model call
(Python-level stdout/stderr, warnings, logging, and whether anything writes directly to the OS file
descriptor). The module structure is chosen **after** that inspection, not in advance. No abstraction
is created for architectural neatness; responsibilities must stay clear, and modules are combined
wherever they remain readable.

**Indicative minimal structure (to be confirmed/trimmed after the inspection):**

| # | File | Responsibility | Justified by |
|---|---|---|---|
| 1 | `src/certification/__init__.py` | package docstring + public exports | importable package |
| 2 | `src/certification/certification.py` | gate types/statuses/scopes/record, capture boundary, sanitize + scan + evidence writing, the ten probes, the confirmation step, and the bounded runner | one cohesive, readable module **if** the inspection shows no concrete reason to split |
| 3 | `src/certification/__main__.py` | thin argparse CLI (`--provider --model --live --cases --out`) | explicit opt-in live entry point |
| 4 | `tests/test_certification.py` | offline tests (section R) | required |

Split #2 further **only** if the inspection shows a concrete need, e.g.:

- the transparent proxy needs non-trivial Strands-specific adapter code → a small dedicated module;
- the output-capture boundary needs non-trivial plumbing → a small dedicated module;
- secret handling must be isolated for review → a small dedicated module;
- the gate file would otherwise be hard to read → split gates from runner.

Every extra file must be justified in the implementation report against the inspected interface.

### Scope guard (mandatory)

> If the planned certification **production** code (excluding tests and certification records) is
> expected to materially exceed **about 600 lines**, or requires **more files than the inspected
> Strands interface justifies**, **STOP and request human approval before implementation.**

### A. MUST change (new files)
Only the files in the confirmed minimal structure above (indicatively 3 source files + 1 test file).

### B. MAY change (only with explicit approval, each a separate decision)
| File | Condition |
|---|---|
| `src/agent/app.py` | only if the human prefers a **public** seam over read-only use of the existing private helpers (`_suggest_category`, `_make_agent_runner`). Not needed by this design. |
| `ARCHITECTURE.md` | document the certification layer and promotion process (docs only). |
| `docs/certification/*` | committed record, created after human review at certification time (never during P3.2A). |
| `.gitignore` | **not required** with the outside-the-repo evidence location. |

### C. READ ONLY / PROTECTED
`src/services/applicability.py`, `src/services/analysis.py`, `src/services/orchestrator.py`,
`src/services/classification.py`, `src/agent/tools.py`, `src/agent/prompts.py`,
`src/agent/model_factory.py`, `src/state.py`, `src/models.py`, `src/repositories/*`, `src/config.py`,
`data/*.json`, `requirements.txt`, `.env.example`, `P3_1_Multi_Provider_Architecture_Plan.md`, and
every existing test file. `model_factory.py` stays frozen: certification consumes `build_model()`
as-is. If certification appears to require a change to any file in this list → **stop and request
approval**.

### D. NEW files
Exactly those in the confirmed minimal structure (P.A). No new top-level project directory is needed
for evidence (`..\ImportReady_AI_Certification_Artifacts\` is outside the Git project).

## Q. Protected files / invariants

Unchanged and non-negotiable:

- LLM = orchestration / NLU / tool selection / explanation; **never** the compliance authority.
- Deterministic code owns canonical facts, evidence lifecycle, applicability, and later risk/cost.
- **Category authority:** an agent-generated category is always `agent_generated` +
  `REVIEW_REQUIRED`; only the existing `HumanClassifier` boundary (human confirmation / `--category`)
  can produce `human_confirmed` + `RESOLVED`; certification code must never create a second category
  authority or mutate a suggestion into a canonical category.
- `evidence_status`, `rule_status`, `applicability_status` vocabularies are untouched.
- `VERIFIED` evidence ≠ applicability; `NOT_FOUND` ≠ "not required"; `PROPOSED`/`WATCHLIST` ≠
  `EFFECTIVE`; missing information stays missing; AI suggestions never become USER/canonical facts.
- The two Agent tool signatures stay exactly `analyze_product(product_description)` and
  `get_compliance_evidence(rule_id)`; no tool is added for certification.
- **R-ELEC-002** keeps its existing behaviour; **R-ELEC-012** remains explicitly deferred, with no
  transportation trigger and no "lithium battery ⇒ transportation applicability" inference.
- No provider/certification code may touch applicability, risk, cost, `ProductFact`, canonical
  `CaseState` facts, evidence, or rule lifecycle.

## R. Testing strategy

**P3.2A offline tests** — new `tests/test_certification.py`, fake targets only, sentinel credentials
only (e.g. `sk-FAKE-TEST-KEY-ONLY`), zero network:

1. **Record/status machinery:** each gate result returns `PASS`/`FAIL`/`BLOCKED`/`NOT_RUN` correctly;
   overall status is `PASS` only when every gate aggregates to `PASS`; a single `FAIL` forces `FAILED`
   (never hidden); a case with an explicit `NOT_RUN` gate cannot be reported as an unqualified PASS.
2. **Gate scoping and aggregation (correction 1):**
   - gate 1 is invoked **exactly once** per run, before any case (fake target counts model
     constructions);
   - gate 9 is invoked **exactly once**, offline, and is not repeated per live case;
   - gates 2–8 and 10 produce one result per applicable case, each carrying the right
     `scope`/`case_id`;
   - **a failing Case B is never hidden by passing Cases A/C**: the aggregate for a case-scoped gate
     is `FAIL`/`BLOCKED` accordingly, and the overall status is not `PASS`;
   - the aggregation function is deterministic (same inputs → same aggregate) and derived only from
     the per-case results;
   - `NOT_RUN` is accepted only for the gate/case combinations this plan declares not applicable.
3. **Failure taxonomy:** each injected failure maps to its category (auth, model-missing, schema,
   structured output, classification decline, category-not-confirmed, tool selection, tool execution,
   continuation, multi-turn, final response, canonical mismatch, timeout, output-capture-unavailable,
   unknown).
4. **Human confirmation boundary (revision 2):**
   - the certification layer never constructs a `CategoryResult`, never sets `human_confirmed`, and
     never mutates the agent suggestion (asserted through the injected fake target's recorded inputs);
   - a confirmation stub that **declines** makes the case `BLOCKED` with `CATEGORY_NOT_CONFIRMED`, and
     gates 3–8 do not run for that case;
   - a stub returning an out-of-vocabulary value yields `NEEDS_INFO` (`HumanClassifier`'s own
     behaviour) → case blocked — proving the boundary, not the certification layer, decides;
   - a stub confirming the suggested supported value yields `human_confirmed` + `RESOLVED` and gates
     3–8 run;
   - the agent-generated result recorded at gate 2 remains `agent_generated` + `REVIEW_REQUIRED`.
5. **Semantic gates 3 and 6 (revision 2):** fake targets exposing the correct tool names/logical
   arguments pass **regardless of representation shape**; fakes that add a tool, rename one, or widen
   an input contract fail; a continuation that carries the tool result semantically passes even in a
   non-`assistant→tool→assistant` representation, while a missing continuation fails. No test asserts
   one provider-specific serialization.
6. **Proxy is observational:** the fake proxy records and returns the delegate's result unchanged; the
   request/messages the provider receives are identical to those produced without the proxy.
7. **Output capture boundary (correction 3):** inject a sentinel credential into (a) `print()` to
   stdout, (b) a write to `sys.stderr`, (c) a `warnings.warn(...)` message, and (d) a logging record on
   the provider/`importready` loggers; assert the sentinel **never** appears in the terminal output
   captured by the test, in the evidence text, or in the record — while the raw in-memory buffer is
   discarded and the original streams are restored after the call (including on exception).
8. **Secret safety:** `sanitize()` redacts the existing patterns **and** a literal sentinel credential;
   `scan_for_secrets()` detects `sk-…`, `Bearer …`, `token=…`, `secret=…`, `AKIA…` and the literal
   value; `write_evidence()` refuses to write on a hit; the rendered record and evidence contain no
   sentinel; provider exception text containing a sentinel is sanitized before recording.
9. **Evidence location:** the default evidence path resolves **outside** the repository root (a sibling
   `ImportReady_AI_Certification_Artifacts` directory); all disk writes go through the sanitizing
   writer (asserted by patching `open`/the writer).
10. **Evidence/record rendering:** all required metadata fields present; the aggregate gate table and
    the Gate × Case matrix both rendered; per-case statuses faithful; the record references the
    artifact by filename + SHA-256 + summary rather than embedding it; no credential value anywhere.
11. **No status mutation:** after a full fake run (including a fake all-PASS run),
    `resolve_combination("deepseek","deepseek-flash").compatibility_status is EXPERIMENTAL` and
    `verified_combinations() == []`; the runner imports no registry-mutating API.
12. **Limits/budget:** the runner stops at the request cap and marks remaining results
    `BLOCKED`/`BUDGET_EXHAUSTED`; fake targets count invocations and the count never exceeds the cap;
    no retry after a failure (one invocation per gate); the confirmation prompt consumes no provider
    request; gate 1 is not re-invoked per case.
13. **Offline-only guarantee:** importing the certification package performs no network access and
    constructs no model; the CLI without `--live` refuses to build a real model (exit 2) and the
    injected model factory is never called; the full suite runs with no credential env var set; the
    case list is exactly A, B, C and an unknown case id (e.g. `D`) is rejected.
14. **Gate 9 offline injection** (mandatory, global) with the sentinel-bearing provider exception.
15. **Gate-8 lifecycle prose tests (mandatory — the lifecycle coverage that replaced live Case D):** a
    definite-current-obligation sentence about `R-ELEC-019` (`PROPOSED`) or `R-ELEC-018` (`WATCHLIST`)
    must **FAIL**; correct wording such as "proposed, not yet effective" must **PASS**; plus the other
    A1–A5 checks (empty text, traceback, secret pattern, APPLICABLE contradiction).
16. **Gate 10 offline invariant:** fake agent runner returning canonical data + adversarial prose →
    canonical block identical to the offline block; R-ELEC-002 APPLICABLE/NOT_APPLICABLE and
    missing-attribute cases asserted; the live lifecycle check (`R-ELEC-018` `WATCHLIST`, `R-ELEC-019`
    `PROPOSED`, never `EFFECTIVE`) asserted against the canonical finding fields; an agent-generated
    category alone yields `applicability is None`.
17. **CLI argument validation:** an unregistered/`UNSUPPORTED` combination is rejected **before** any
    model is constructed, using the existing `resolve_combination()` policy.
18. **Gate-2 criteria per case:** Case A **and** Case B both require
    `small_consumer_electronics` + `agent_generated` + `REVIEW_REQUIRED` (a missing USER `ProductFact`
    affects applicability, not the category suggestion); Case C requires the safe
    `NEEDS_INFO`/in-vocabulary outcome; an out-of-vocabulary invention is downgraded to `NEEDS_INFO`.
19. **Gate 7 warning is optional:** with fakes — (a) no warning observed → `warning_status =
    NOT_OBSERVED` **and gate 7 can still PASS**; (b) warning observed with a clean normal completion →
    classified A/B and PASS; (c) warning observed with a blocked completion → class C (`FAIL`) / class
    D (`BLOCKED`). No test forces the warning to reproduce, and its absence never fails the gate.
20. **Live pre-flight binding:** exact CLI/env match → proceeds; provider mismatch → refused; model
    mismatch → refused; missing env value → refused; unknown/`UNSUPPORTED` combination → refused. In
    every refusal case the injected model factory is **never** called, no provider request is made, the
    error is sanitized, and `os.environ` is unchanged (asserted by snapshot comparison) — the runner
    never mutates the environment to force a match.
21. **Evidence-path validation:** default sibling path → allowed; explicit outside/sibling path →
    allowed; repo-root path → refused; repo subdirectory path → refused; a relative path resolving back
    into the repo → refused; refusal happens before any write (`EVIDENCE_PATH_REJECTED`) and no file is
    created.

**What P3.2A explicitly does not prove:** provider compatibility. Only P3.2B produces that evidence.
The offline tests certify the *mechanism*.

**Regression:** `python -m pytest tests/test_certification.py -q` then `python -m pytest -q`; the
existing 436 tests must stay green (expected total = 436 + new, 0 failures). No live provider call is
part of either command.

## S. DeepSeek execution method (P3.2B — not executed now)

**Harness constraints that shape this method:**

- each agent tool call runs in a **fresh PowerShell process**, so `$env:API_KEY` set in one call does
  not persist to the next;
- a credential must never appear in a command string (it would be logged);
- third-party SDK/Strands/provider logging can write straight to stdout/stderr, so shell-level capture
  is both unsafe and prohibited.

Therefore: **the human runs the live certification command in their own PowerShell window**, and all
live output is captured **in-process by the runner** (section N.3).

Shape (the human types the key at the masked prompt; no real value appears anywhere):

```powershell
# 1. Human's own PowerShell session, repo root
$sec = Read-Host -AsSecureString "DeepSeek API key"
$env:API_KEY = [System.Net.NetworkCredential]::new("", $sec).Password   # never echoed
$env:MODEL_PROVIDER = "deepseek"; $env:MODEL_ID = "deepseek-flash"

# 2. Bounded live certification run.
#    Pre-flight (no network): the runner verifies --provider/--model == MODEL_PROVIDER/MODEL_ID
#    and that the exact combination is registered; otherwise it refuses before constructing a
#    model. It also validates the evidence path. It never edits os.environ to force a match.
#    NO Tee-Object, NO *>, NO >, NO 2>&1 | file. The runner captures provider output
#    in-process, sanitizes + scans it, writes the evidence artifact itself OUTSIDE the
#    Git project, and prints only a minimal sanitized summary.
& .\.venv\Scripts\python.exe -m src.certification `
    --provider deepseek --model deepseek-flash --live --cases A,B,C
$code = $LASTEXITCODE

# 3. Immediate cleanup (same window) + non-printing verification
Remove-Item Env:\API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:\MODEL_ID, Env:\MODEL_PROVIDER -ErrorAction SilentlyContinue
if ($env:API_KEY) { "API_KEY STILL SET" } else { "API_KEY UNSET" }
"certification exit code: $code"
```

Policy for live evidence and console output (final):

1. **Runner-only capture.** Provider/Strands execution runs inside the in-process capture boundary
   (stdout/stderr/warnings/logging) for the duration of the call; the captured text is sanitized,
   scanned, summarized, and then the raw buffer is discarded. Nothing else captures provider output.
2. **No raw provider console dump on disk, ever.** No `Tee-Object`, no `*>`, no `>`, no `2>&1 |`
   redirection to a file for the live run.
3. **Console output stays minimal and sanitized:** one line per gate/case (id, scope, status, failure
   category) plus the overall status and exit code. No request/response dumps, no raw exception text,
   no headers.
4. **The evidence location is outside the Git project — including with `--out`.**
   Default: `..\ImportReady_AI_Certification_Artifacts\<provider_id>__<model_id>__<UTCstamp>.txt`
   (resolved from the repo root). An explicit `--out` path is resolved/canonicalized **before** use and
   must resolve outside the repository root; a repo-root path, a repo subdirectory, or a relative `..`
   path that resolves back into the repo is refused (`EVIDENCE_PATH_REJECTED`) before anything is
   written. The runner prints only the artifact filename and SHA-256, never the absolute path or any
   content. Only the reviewed, secret-free record may later live under `docs/certification/`.
5. **The record** (`docs/certification/<provider_id>__<model_id>.md`) is created **only after human
   review** and **only if secret-free**, and references the artifact by filename + SHA-256 + summary.
   No `.gitignore` change is needed.
6. **Cleanup:** the key is removed from the process environment immediately after the run (step 3); the
   window may then be closed; the key is never written to `.env`, a file, or Git. If the scan reports a
   hit, delete the artifact and rotate the key (section N.8).
7. **Review flow:** the human inspects the artifact outside the repo (the agent may be unable to read
   outside the workspace). After the human confirms it is secret-free, the human either shares a
   secret-free summary or places a sanitized excerpt in the repo; the agent drafts the record from that
   reviewed material. The agent must not request, receive, or type the key at any point.

Do **not** execute any of this during planning.

## T. Bedrock / competition-provider reuse

The mechanism is provider-agnostic: everything provider-specific lives in the existing
`ProviderDescriptor` (`base_url`, `credential_scope`, `runtime_env`, `build`). The certification layer
only calls `build_model()` (dev path) and the existing agent/classification/confirmation wiring.

- **Bedrock + exact model id:** once a human approves registering an exact `bedrock` combination as
  `EXPERIMENTAL`, the run uses the same 3 cases, the same gates, and the AWS credential chain
  (`credential_scope = ()`, `runtime_env = ("AWS_REGION",)`). Gate 1 records the region as the endpoint
  strategy. No compliance/applicability/tool/ProductFact change.
- **Competition provider + exact model id:** register the exact combination as `EXPERIMENTAL`, then run
  the identical checklist. An OpenAI-compatible endpoint is **not** assumed compatible — exactly the
  class of issue already seen with DeepSeek.
- Providers needing a different authentication mechanism obtain credentials through their approved
  adapter/`RuntimeProviderConfig`; certification semantics stay identical.
- The output-capture boundary (N.3) is re-validated per provider during step 0, since a different SDK
  may use different output channels.
- Cost note for later: declaring `boto3`/`botocore` explicitly is a separate, approval-gated dependency
  decision; **no dependency change is made in P3.2A**.

## U. Breaking-change risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Using private helpers (`_suggest_category`, `_make_agent_runner`) couples certification to internal names | documented coupling; a rename makes P3.2A tests fail loudly; a public seam is an approval-gated MAY-change |
| 2 | A live call happens accidentally (CI, local suite) | live path is a CLI requiring `--live`; import path makes no network/model construction; an offline test asserts the non-`--live` run never builds a model |
| 3 | Credential leaks into evidence/logs/record | in-process capture + layered sanitisation + mandatory scan + fail-closed `SECRET_SAFETY_FAILURE`; the key never enters a command string or a file |
| 4 | The `reasoningContent` warning is misjudged — or wrongly treated as mandatory | classify A–D **only if observed**; `NOT_OBSERVED` is recorded and does **not** fail gate 7; the warning is never forced to reproduce; C/D stop the run and require approval |
| 5 | False promotion of a combination | no automatic status writes; explicit human sign-off; `ui_exposed` decided separately; offline test asserts the registry is unchanged after a full run |
| 6 | A provider-specific workaround silently enters production | any history/request transformation is a production change → stop + approval; `params`/`extra_body`/thinking must not return |
| 7 | Cost/time overrun on live runs | gate 1 once, 3 cases, per-case and per-run request caps (24), 6 turns, sequential, no retries |
| 8 | A hung provider request blocks the run | documented: no SDK timeout injected (would be a production change); the human aborts and the run is recorded `INCOMPLETE`/`TIMEOUT` |
| 9 | Fake-target tests create false confidence | P3.2A certifies the mechanism only; compatibility evidence comes from P3.2B |
| 10 | Certification scope creep into compliance logic | certification imports production code, never re-implements it; no new Agent tool; protected-file stop rule |
| 11 | Certification becomes a second category authority | the only confirmation path is the existing `HumanClassifier` through `Orchestrator.run(provided_category=…)`; never constructs a `CategoryResult`; decline → `CATEGORY_NOT_CONFIRMED` + BLOCKED; offline tests assert both directions |
| 12 | Over-specifying provider wire format in gates 3/6 causes false FAILs (or forces request mutation) | gates 3/6 assert semantics only; no representation shape asserted; the proxy never mutates |
| 13 | Third-party output bypassing the sanitizer | in-process capture boundary for stdout/stderr/warnings/logging; raw buffer discarded; shell capture prohibited; if a channel cannot be captured → `OUTPUT_CAPTURE_UNAVAILABLE` + BLOCKED + stop for approval |
| 14 | Evidence artifact accidentally committed with unforeseen secret patterns | artifact outside the Git project by default; record embeds filename + hash + summary only; literal-value substitution covers unknown prefixes |
| 15 | Unbounded P3.2A scope/abstraction | structure chosen **after** the read-only inspection; indicative minimal structure; mandatory ~600-line / file-count scope guard → STOP and ask |
| 16 | **A failing case hidden by another case passing** | explicit `scope`/`case_id` on every `GateResult`; deterministic aggregation (H.0) with FAIL/BLOCKED precedence; Gate × Case matrix in the record; offline tests assert a failing Case B cannot be masked |
| 17 | **Duplicate/faithless cases inflating coverage** | live cases are exactly A, B, C; the former Case D is removed; lifecycle safety is mandatory via live gate 10 + offline Gate-8 adversarial tests; any future live probe needs a distinct observable purpose and separate approval |
| 18 | **The live run certifies a different model than the CLI claims** (`--model` vs `MODEL_ID`, stale/missing env, or an unregistered pair) | pre-flight binding check (N.9): CLI == env == resolved registered combination, verified **before** any model construction; mismatch/absence → sanitized `CONFIGURATION_MISMATCH`, 0 provider requests; `os.environ` is never mutated to force a match; the record cites the validated combination |
| 19 | **Raw evidence written inside the Git repository** via `--out` or a `..` path trick | path resolution + canonicalization before any write; the resolved path must be outside the repository root; repo-root/subdirectory/relative-escape paths are refused (`EVIDENCE_PATH_REJECTED`); unresolvable paths fail closed; only the reviewed record may live under `docs/certification/` |

## V. Updated implementation order

1. Human approves this revised plan.
2. **P3.2A-1** Read-only interface inspection (`Model` ABC, `AgentResult`, tool-spec visibility,
   **output channels**) — record findings; if a channel cannot be captured in-process → STOP and ask.
3. **P3.2A-2** Choose the smallest structure (section P) and confirm it against the scope guard; if the
   guard trips → STOP and ask.
4. **P3.2A-3** Implement types/record/scope/aggregation + capture boundary + sanitize/scan/evidence
   writer.
5. **P3.2A-4** Implement the target protocol + `ProductionTarget` + `RecordingModelProxy` + the ten
   probes + the human confirmation step.
6. **P3.2A-5** Implement the bounded runner (GLOBAL gate 1 once → per-case gates 2–8, 10 → GLOBAL
   offline gate 9 → aggregation) + `--live`-guarded CLI + the live pre-flight (CLI/env target binding
   and evidence-path confinement).
7. **P3.2A-6** Add `tests/test_certification.py` (section R items 1–21).
8. **P3.2A-7** Run `python -m pytest tests/test_certification.py -q`, then `python -m pytest -q`
   (expect 436 + new, 0 failures).
9. **Human code review of P3.2A** (no live calls yet).
10. **P3.2B-1** Human prepares the bounded command of section S (key never shared).
11. **P3.2B-2** Human runs the DeepSeek + `deepseek-flash` 10-gate certification for cases A, B, C,
    including the interactive category-confirmation step; the runner first binds CLI == env and
    validates the evidence path (refusing before any model construction if either check fails).
12. **P3.2B-3** Sanitized evidence artifact captured outside the repo; key removed from the shell and
    verified unset.
13. **P3.2B-4** Human reviews the artifact; the record (aggregate table + Gate × Case matrix) is drafted
    from reviewed, secret-free material and created under `docs/certification/`.
14. **P3.2B-5 PASS** → separately approved one-line status change `EXPERIMENTAL` → `VERIFIED` with
    `certification_ref`.
15. **P3.2B-6** Only later, and separately, decide whether `ui_exposed=True`.
16. **P3.2B-5 FAIL** → remains `EXPERIMENTAL`; investigate the exact failing gate/case; any production
    fix is separately approved and certification is re-run from the affected gate (or fully).

Nothing in steps 2–16 is started by this planning task.

## W. Explicit stop / approval conditions

Stop and request human approval **before** proceeding if any of the following occurs:

1. certification appears to require modifying any file listed as PROTECTED (including
   `src/agent/model_factory.py` and `src/agent/app.py`);
2. certification would need its **own category authority** — constructing a `CategoryResult`,
   promoting an agent suggestion to `human_confirmed`/`RESOLVED`, or bypassing `HumanClassifier`;
3. the `Model` interface cannot be proxied transparently and observing gates 3–7 would require
   changing production provider behaviour or mutating requests/history;
4. gate 6/7 discovers a required request/history transformation (class D) — e.g. stripping
   `reasoningContent`, re-adding thinking/`extra_body` parameters, or any provider-specific
   conversation rewrite;
5. an output channel cannot be **safely captured/suppressed in-process** without changing production
   behaviour (e.g. a component writing directly to the OS file descriptor) → mark the affected
   observation `BLOCKED` (`OUTPUT_CAPTURE_UNAVAILABLE`) and stop; never claim the sanitizer controls a
   channel that bypasses it;
6. a credential would have to be placed in a command, file, environment variable owned by the agent
   process, log, report, or Git;
7. any dependency/package change becomes necessary (including declaring `boto3`/`botocore`);
8. the planned certification production code is expected to materially exceed about **600 lines**, or
   to need more files than the inspected Strands interface justifies;
9. live certification would exceed the section O budgets (including the 24-request cap), or a case
   would need re-running beyond "each case once";
10. a gate fails in a way that would require weakening a PASS criterion, the aggregation rule, or an
    ImportReady invariant (including hiding a failing case behind a passing one);
11. any change to compliance semantics, rule data, `ProductFact`, evidence lifecycle, or the two Agent
    tool signatures appears necessary;
12. a new live certification case (e.g. a lifecycle-explanation probe) is proposed — it needs a
    distinct observable purpose and separate approval, and `product_description` must never carry
    certification-only instructions;
13. promotion of a combination to `VERIFIED`, or enabling `ui_exposed`, is requested — both are
    separate human decisions, never automatic.

## X. Confirmation for this planning task

- **No implementation**: no production file and no test file was created or modified.
- **Exactly one file touched**: `P3_2_Provider_Certification_Plan.md` (created in revision 1, revised
  in revisions 2, 3 and 4). No other file created, modified, renamed, or deleted.
- **No real API/model call** was made; no provider SDK request was issued.
- **No credential was set, read, or printed.** `.env` was not read, inspected, printed, or modified.
- **No package** was installed, upgraded, or removed; `requirements.txt` is unchanged.
- **No git state change**: nothing staged, committed, pushed, fetched, pulled, reset, checked out,
  rebased, or merged.
- Read-only operations used in this revision: content greps and a file read of
  `src/services/analysis.py` (`ComplianceFinding.rule_status` / `evidence_status`), plus
  `git status --short`.
- Verified baseline correction retained: HEAD is `84d420b`, not `a4d420b` as stated in the task brief.

**Plan revised. Awaiting approval before any P3.2A implementation or P3.2B live call.**
