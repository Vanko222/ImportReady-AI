# ARCHITECTURE.md

## 1. Purpose

This document defines the MVP architecture for **ImportReady AI — AI Agent for Cross-border Import Risk Assessment**.

The architecture must support two goals:

1. Build a complete hackathon MVP quickly.
2. Keep clean extension points for post-hackathon development.

The MVP should remain simple, testable, traceable, and safe under uncertainty.

---

## 2. Core Principles

1. **Deterministic logic before LLM judgment**
   - Use Python for rule gates, applicability, risk, cost arithmetic, and ID validation.
   - Use the LLM mainly for natural-language understanding, classification assistance, orchestration, and report wording.

2. **Trusted data only**
   - Compliance conclusions must come from approved Data Schema v1.
   - The model must not invent rules, requirements, costs, or sources.

3. **Safe uncertainty**
   - Missing information must produce `NEEDS_INFO`, `REVIEW_REQUIRED`, `UNCERTAIN`, or `UNSUPPORTED`.
   - Unknown information must never become a false green result.

4. **Human-in-the-loop**
   - Users confirm missing product facts.
   - Policy changes require human approval before trusted data changes.

5. **Context on demand**
   - Load only the facts, rules, attributes, costs, and sources needed for the current step.

6. **Structured memory**
   - Separate session memory, case state, project documentation, and trusted compliance knowledge.

7. **Replaceable boundaries**
   - UI, API, model provider, storage, and policy-source adapters should be replaceable without rewriting core business logic.

8. **Traceability**
   - Compliance findings must resolve to canonical rule IDs and source IDs.

---

## 3. MVP Technology Stack

### Runtime

- Python 3.11+
- Streamlit — interactive demo UI
- FastAPI — lightweight external API
- Strands Agents SDK — Agent orchestration
- Pydantic — structured models and validation
- Local UTF-8 JSON — trusted MVP knowledge base
- Streamlit session state — active demo case memory
- pytest — testing
- Docker — packaging after local flow works

### Development

- DeepSeek Harness — primary coding agent
- Git/GitHub — source of truth

Harness is a development tool, not part of the runtime application.

### Not in MVP

Do not add unless required:

- React frontend
- authentication system
- PostgreSQL
- vector database
- microservices
- generic crawler
- scheduler
- autonomous trusted-policy updates
- complex multi-agent infrastructure

---

## 4. High-Level Architecture

```text
                    ┌───────────────┐
                    │ Streamlit UI  │
                    └───────┬───────┘
                            │
┌───────────────┐           │
│ FastAPI Layer │───────────┤
└───────────────┘           │
                            ▼
                  ┌──────────────────┐
                  │ App Orchestrator │
                  │ / Strands Agent  │
                  └────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
 Classification      Compliance         Cost / Risk /
    Service            Service          Report Services
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 Repository / Case Store
                           │
                           ▼
                Approved JSON Knowledge
```

Plain language:

The user can enter a product through Streamlit or the API. Both paths call the same core services. The system classifies the product, loads relevant approved rules, asks for missing facts when necessary, evaluates compliance, calculates risk and known cost impact, and returns a traceable report.

---

## 5. Layer Boundaries

Use this dependency direction:

```text
UI / API
→ Orchestrator
→ Services
→ Repository / Model Gateway / Case Store
→ JSON / External Model / Approved Sources
```

Rules:

- Streamlit logic stays in the UI layer.
- FastAPI logic stays in the API layer.
- Strands-specific code stays in the Agent layer.
- Compliance, risk, and cost logic must not depend on Streamlit or FastAPI.
- Core business logic must not depend on one model provider.
- Repository code hides whether storage is JSON now or a database later.
- Agent tools should be thin wrappers around application services.

---

## 6. Runtime Data

Trusted runtime files:

```text
data/
├── product_taxonomy.json
├── compliance_rules.json
├── policy_sources.json
├── compliance_costs.json
├── known_gaps.json
└── pending_policy_updates.json
```

### Responsibilities

**product_taxonomy.json**
- canonical attributes
- category metadata
- attribute-to-rule links

**compliance_rules.json**
- rule evidence and lifecycle
- trigger conditions
- required attributes/tests/documents
- clarification questions
- risk metadata
- source references

**policy_sources.json**
- authority
- source metadata
- official URLs
- update-check metadata
- traceability

**compliance_costs.json**
- compliance-cost references only
- not a complete landed-cost database

**known_gaps.json**
- persistent research/system limitations
- `NOT_FOUND` never means `NOT_REQUIRED`

**pending_policy_updates.json**
- candidate policy changes awaiting review

The four files in `research/` are immutable evidence snapshots and are not normal runtime inputs.

---

## 7. Domain Models

Use Pydantic models for structured runtime data.

Important models:

```text
ProductInput
ClassificationResult
ProductFact
CaseState
RuleEvaluation
RiskResult
CostResult
ReportResult
PolicyUpdateCandidate
```

All model-generated structured outputs must be validated before use.

---

## 8. Repository Layer

Define a `ComplianceRepository` interface.

Responsibilities:

- load and validate JSON
- index canonical IDs
- query rules by category
- resolve rule/source/attribute relationships
- expose compliance costs and known gaps
- fail clearly on broken references

Logical interface:

```python
class ComplianceRepository:
    def get_attribute(self, attribute_id): ...
    def get_rule(self, rule_id): ...
    def get_source(self, source_id): ...
    def get_rules_for_category(self, category): ...
    def get_costs_for_category(self, category): ...
    def get_known_gaps(self): ...
```

MVP implementation:

```text
JsonComplianceRepository
```

Future replacements may use SQLite, PostgreSQL, or a remote policy service.

---

## 9. Product Classification

Allowed categories:

```text
childrens_toys
small_consumer_electronics
dual
unsupported
uncertain
```

The classifier may use the LLM to understand natural-language product descriptions.

It must not:

- create regulations
- determine legal obligations from memory
- invent missing product facts
- invent source citations
- override supported-category boundaries

Suggested output:

```text
category
reason
facts_extracted
facts_missing
confidence
```

Routing:

```text
childrens_toys
→ toy rule path

small_consumer_electronics
→ electronics rule path

dual
→ both relevant rule paths

unsupported
→ stop safely

uncertain
→ ask clarification
```

For `dual`, collect required attributes from both relevant paths and de-duplicate by canonical attribute ID.

---

## 10. Compliance Evaluation

The rule engine must follow:

```text
Evidence Gate
→ Lifecycle Gate
→ Trigger Evaluation
→ Applicability Result
```

### Evidence Gate

Only:

```text
evidence_status = VERIFIED
```

may enter active obligation evaluation.

Other evidence states may generate review information but not confident obligations.

### Lifecycle Gate

Only:

```text
rule_status = EFFECTIVE
```

may create active obligations.

`PROPOSED`, `WATCHLIST`, `SUPERSEDED`, and `UNKNOWN` must not become active obligations.

### Trigger Evaluation

Evaluate `trigger_conditions` using structured product facts.

Use deterministic evaluation when possible.

### Applicability Result

Allowed runtime values:

```text
APPLICABLE
NOT_APPLICABLE
NEEDS_INFO
REVIEW_REQUIRED
```

Missing trigger facts must not automatically become `NOT_APPLICABLE`.

---

## 11. Clarification Engine

When information is missing:

```text
missing canonical attribute
→ approved clarification question
→ user answer
→ validate answer
→ update case facts
→ re-evaluate affected rules
```

Already answered facts should not be requested again unless changed.

The clarification loop is a normal workflow state, not an error.

---

## 12. Case State and Memory

Each analysis uses a structured `CaseState`.

Suggested fields:

```text
case_id
created_at
raw_product_input
classification
product_facts
missing_attribute_ids
rule_results
clarification_history
risk_result
cost_result
analysis_status
knowledge_snapshot
```

### Fact origins

Where useful, store:

```text
USER
CLASSIFIER
DERIVED
```

Important uncertain classifier-derived facts should be confirmable by the user.

### Case status

```text
IN_PROGRESS
NEEDS_INFO
REVIEW_REQUIRED
COMPLETE
UNSUPPORTED
```

Case status is separate from risk level.

---

## 13. Context Management

Use three runtime context layers.

### A. Stable Agent Context

Contains:

- system role
- supported categories
- safety constraints
- tool descriptions
- output contracts

### B. Structured Case Context

Contains:

- current classification
- known product facts
- unanswered facts
- relevant rule results
- current workflow state

### C. Task Knowledge Context

Contains only the relevant:

- rules
- attributes
- costs
- source metadata

Do not send the entire dataset on every model call.

Use:

```text
full trusted knowledge
→ repository filtering
→ relevant records
→ minimal model context
```

Do not depend on replaying an indefinitely growing raw chat history.

Convert important user answers into structured `CaseState`.

---

## 14. Memory Management

Separate four kinds of memory.

### Developer Project Memory

```text
PRODUCT.md
STATUS.md
DECISIONS.md
ARCHITECTURE.md
AGENTS.md
```

These guide development, not end-user analysis.

### Session Memory

MVP:

```text
Streamlit session state
```

Stores the current interactive case.

### Case Memory

Use a `CaseStore` interface.

MVP may store cases only in memory/session state.

Future implementations can use:

```text
SQLite
PostgreSQL
cloud storage
```

without changing compliance services.

### Trusted Knowledge Memory

Approved taxonomy, rules, sources, and costs.

This is separate from conversation memory.

The model must never automatically promote generated text, user chat, or external claims into trusted compliance knowledge.

---

## 15. Hallucination and Evidence Control

Hallucination control must be enforced in code.

### Structured Outputs

Validate all model outputs.

Reject:

- unknown category values
- malformed responses
- unknown canonical IDs
- invalid enum values

### ID Allowlist

All referenced:

```text
attribute_id
rule_id
source_id
cost_id
```

must resolve through the repository.

The model cannot create valid references by inventing plausible IDs.

### No Model-Generated Compliance Facts

The model may explain approved results.

It may not independently create:

- regulatory requirements
- certification requirements
- testing obligations
- regulatory costs
- policy sources

### Traceability

Each confirmed compliance finding should retain:

```text
rule_id
source_ids
evidence_status
rule_status
applicability_status
```

### Forbidden Behaviors

```text
missing fact → NOT_APPLICABLE
NOT_FOUND → NOT_REQUIRED
PROPOSED → active obligation
WATCHLIST → active obligation
QUOTE_REQUIRED → zero
unsupported product → guessed answer
```

---

## 16. Risk Engine

Risk calculation must be deterministic.

### HIGH

Return `HIGH` when a confirmed applicable active rule has an unmet requirement whose normalized `risk_if_missing` is `HIGH`.

### MEDIUM

Return at least `MEDIUM` when:

- a P0 decision is blocked by missing information, or
- a relevant rule requires `REVIEW_REQUIRED`, or
- a confirmed unmet requirement has medium severity,
- and no confirmed high-risk condition exists.

### LOW

Return `LOW` only when:

- relevant P0 rules are resolved
- no unresolved blocker exists
- no confirmed high/medium unmet requirement exists

### Unknown Severity

Do not guess.

Escalate to:

```text
REVIEW_REQUIRED
```

If analysis still contains `NEEDS_INFO` or `REVIEW_REQUIRED`, risk must be shown as provisional.

---

## 17. Cost Calculator

Respect current `calculator_use` semantics.

### DIRECT

May contribute to the known direct subtotal when applicable.

### PLANNING_ONLY

May be shown as a planning estimate/range.

Do not present as a universal fixed fee.

### QUOTE_REQUIRED

Must never become zero.

A required quote-only item means the total is incomplete.

### DISPLAY_ONLY

Show for information but exclude from arithmetic totals.

Suggested output:

```text
direct_subtotal
planning_low
planning_high
quote_required_items
display_only_items
cost_status
notes
```

Suggested status:

```text
COMPLETE_FOR_KNOWN_DIRECT_COSTS
INCOMPLETE_QUOTE_REQUIRED
REVIEW_REQUIRED
```

Future landed-cost expansion can add independent providers for:

- purchase price
- quantity
- logistics
- tariffs
- marketplace fees
- compliance costs

Do not force these future concepts into `compliance_costs.json`.

---

## 18. Report Generation

Final report sections:

1. Product Classification
2. Risk Level
3. Analysis Status
4. Detected Risks
5. Missing Requirements / Documents
6. Clarifications Needed
7. Estimated Compliance Cost Impact
8. Recommended Actions
9. Relevant Sources
10. Known Limitations

Flow:

```text
structured case result
→ deterministic report payload
→ optional LLM wording
→ Streamlit / API response
```

The LLM may improve wording but must not alter statuses, IDs, costs, or calculated results.

Source links must resolve from `policy_sources.json`.

---

## 19. Human-in-the-Loop

Two separate workflows are required.

### Product Clarification

```text
missing product fact
→ user question
→ user answer
→ structured fact update
→ re-evaluation
```

### Policy Governance

```text
approved source check
→ candidate change
→ pending update
→ human review
→ approve / reject
→ trusted knowledge changes only after approval
```

Do not merge these two approval concepts.

---

## 20. Policy Update Workflow

MVP scope:

- approved official sources only
- one demonstrable source path is sufficient initially
- no generic crawling
- no scheduler
- no autonomous trusted-data modification

Logical components:

```text
PolicySourceAdapter
PolicyUpdateDetector
PendingPolicyUpdateRepository
PolicyApprovalService
```

Only an `APPROVED` and schema-valid candidate may update trusted knowledge.

---

## 21. Model Gateway

Define a replaceable `ModelGateway`.

Possible model-assisted tasks:

- product classification
- clarification wording
- report wording
- policy-change interpretation

Business logic should use structured input/output contracts.

Credentials come from environment variables.

Example:

```text
MODEL_PROVIDER=
MODEL_ID=
MODEL_API_KEY=
```

Never commit real keys.

---

## 22. Agent Architecture

Strands Agent orchestrates controlled services/tools.

Possible tools:

```text
classify_product
get_required_clarifications
evaluate_compliance
calculate_risk
calculate_cost_impact
build_report
check_policy_update
review_policy_update
```

The Agent must not be the source of truth for:

- evidence status
- rule lifecycle
- applicability logic
- risk calculation
- cost arithmetic
- policy approval

Tools should return structured data whenever possible.

---

## 23. FastAPI Interface

The MVP includes a lightweight API layer for programmatic testing.

Streamlit and FastAPI must call the same core services.

```text
Streamlit ─┐
           ├→ Core Services
FastAPI ───┘
```

Do not duplicate business logic inside API endpoints.

### P0 Endpoints

#### `GET /health`

Purpose:

- verify application availability
- verify trusted data loads successfully

Example response:

```json
{
  "status": "ok",
  "schema_version": "1.0"
}
```

#### `POST /analyze`

Purpose:

Run one product-analysis step.

Example request:

```json
{
  "product_description": "Bluetooth headphones for children",
  "product_facts": {},
  "cost_inputs": {}
}
```

Example response shape:

```json
{
  "case_id": "case_xxx",
  "classification": {},
  "analysis_status": "NEEDS_INFO",
  "risk": {},
  "missing_attributes": [],
  "clarification_questions": [],
  "findings": [],
  "cost": {},
  "sources": []
}
```

If clarification is required, the client may call `/analyze` again with additional `product_facts`.

This keeps the MVP API simple and avoids building a complex conversational protocol.

### Optional P1 Endpoints

Only if needed:

```text
POST /policy/check
POST /policy/review
```

Do not implement them before the core analysis flow works.

---

## 24. API Key and Submission Safety

External model/API credentials must be separated from source code.

Use:

```text
.env
```

and ensure `.env` is listed in `.gitignore`.

Provide:

```text
.env.example
```

with variable names only.

Example:

```text
MODEL_PROVIDER=
MODEL_ID=
MODEL_API_KEY=
ENABLE_API=true
```

### Development/Test Version

The user can place a real API key in local `.env` for testing.

### Competition Submission

The submitted repository must contain:

- API interface code
- `.env.example`
- no real API key
- no private credentials

The API layer can also be disabled through configuration if desired.

No source-code deletion should be required before submission.

---

## 25. Errors and Safe Failure

### JSON load failure

Stop analysis and show a data/system error.

Do not continue with partial knowledge.

### Broken reference

Treat as validation failure.

Do not ask the model to fill it.

### Model unavailable

Show a model-service error.

Deterministic services should remain independently testable.

### Unsupported product

Return `UNSUPPORTED`.

### Uncertain classification

Return `UNCERTAIN` and request clarification.

### Missing quote

Show known subtotal and missing quote item.

Never use zero as the missing quote value.

---

## 26. Security

MVP rules:

- secrets only in environment variables
- `.env` excluded from Git
- `.env.example` contains no secrets
- trusted JSON is read-only during normal analysis
- policy updates use a dedicated controlled write path
- runtime users cannot execute arbitrary shell commands
- logs must not contain API keys
- policy checks use approved source URLs only

Production features such as authentication, permissions, audit logs, encryption, and tenant isolation are future work.

---

## 27. Logging

Use normal Python logging for the MVP.

Useful fields:

```text
case_id
workflow_step
category
rule_id
applicability_status
analysis_status
error_type
```

Do not log secrets.

---

## 28. Project Structure

```text
ImportReady-AI/
├── app.py
├── api.py
├── requirements.txt
├── .gitignore
├── .env.example
├── AGENTS.md
├── PRODUCT.md
├── STATUS.md
├── DECISIONS.md
├── ARCHITECTURE.md
│
├── research/
│   └── [four immutable research files]
│
├── data/
│   ├── product_taxonomy.json
│   ├── compliance_rules.json
│   ├── policy_sources.json
│   ├── compliance_costs.json
│   ├── known_gaps.json
│   └── pending_policy_updates.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── state.py
│   │
│   ├── repositories/
│   │   ├── compliance_repository.py
│   │   └── pending_update_repository.py
│   │
│   ├── services/
│   │   ├── classification.py
│   │   ├── clarification.py
│   │   ├── compliance.py
│   │   ├── risk.py
│   │   ├── cost.py
│   │   ├── reporting.py
│   │   └── policy_updates.py
│   │
│   └── agent/
│       ├── runtime.py
│       ├── gateway.py
│       ├── context_builder.py
│       └── tools.py
│
└── tests/
    ├── test_repository.py
    ├── test_classification.py
    ├── test_compliance.py
    ├── test_risk.py
    ├── test_cost.py
    └── test_api.py
```

This is a logical target structure.

Harness should create files incrementally instead of generating unnecessary empty modules.

---

## 29. Testing Strategy

### Data Tests

Verify:

- all JSON parses
- IDs are unique
- references resolve
- allowed statuses are valid
- `QUOTE_REQUIRED` is never treated as zero

### Deterministic Unit Tests

Verify:

- evidence gate
- lifecycle gate
- trigger evaluation
- missing-information behavior
- dual-category routing
- risk logic
- cost logic

### Agent Contract Tests

Verify:

- structured output schema
- allowed categories
- unknown IDs are rejected
- model output cannot override deterministic results

### API Tests

Verify:

```text
GET /health
POST /analyze
```

with:

- toy product
- electronics product
- unsupported product
- missing-information case

### End-to-End Test

At least one complete flow:

```text
product input
→ classification
→ rule routing
→ clarification
→ compliance
→ risk
→ cost
→ report
```

The same core flow should work through Streamlit and FastAPI.

---

## 30. Extension Points

Reserve interfaces now; do not implement unnecessary infrastructure.

### Storage

```text
JsonComplianceRepository
→ database repository
```

### Case Memory

```text
Session CaseStore
→ persistent CaseStore
```

### Models

```text
ModelGateway
→ multi-provider routing
```

### Policy Monitoring

```text
single approved source adapter
→ multiple source adapters
→ optional scheduler
```

### Product Coverage

```text
two categories
→ additional category policy packs
```

### Jurisdictions

```text
US MVP
→ jurisdiction-specific rule packs
```

### Costs

```text
compliance costs
→ tariffs
→ logistics
→ marketplace fees
→ landed-cost aggregator
```

### Delivery Interfaces

```text
Streamlit + FastAPI
→ production web frontend + public/private API
```

---

## 31. MVP Build Order

### P0-1 — Domain + Repository

Build:

- Pydantic models
- JSON repository
- startup validation
- case state

Acceptance:

- all six JSON files load
- IDs can be queried
- broken references fail clearly

### P0-2 — Classification

Acceptance:

- toy routes correctly
- electronics routes correctly
- unsupported stops safely
- uncertain asks clarification

### P0-3 — Compliance + Clarification

Build:

- evidence gate
- lifecycle gate
- trigger evaluation
- missing-fact detection
- clarification loop

Acceptance:

- unknown trigger facts never become false `NOT_APPLICABLE`

### P0-4 — Risk Engine

Acceptance:

- `LOW` is impossible while a relevant P0 blocker remains unresolved

### P0-5 — Cost Calculator

Acceptance:

- `QUOTE_REQUIRED` never becomes zero
- direct and planning costs remain distinguishable

### P0-6 — Core End-to-End Flow

Build:

```text
input
→ classification
→ clarification
→ compliance
→ risk
→ cost
→ report
```

### P0-7 — Streamlit UI

Connect the working core flow to the demo interface.

### P0-8 — FastAPI

Implement:

```text
GET /health
POST /analyze
```

Both must call the same core services as Streamlit.

### P0-9 — Strands Integration

Use Strands to orchestrate existing services.

Acceptance:

- business logic remains usable without rewriting it for the Agent

### P1 — Policy Update Workflow

Add candidate detection and human approve/reject.

### P1 — Docker

Dockerize after the local happy path is stable.

---

## 32. Architecture Acceptance Criteria

Architecture implementation passes when:

1. Compliance facts come only from approved structured data.
2. `VERIFIED + EFFECTIVE` gating is enforced in code.
3. Missing trigger facts cannot create a false green result.
4. `dual` products can evaluate both relevant category paths.
5. Case facts are structured instead of relying on endless raw chat history.
6. Model context is filtered to the current task.
7. Model output cannot silently modify trusted knowledge.
8. Risk calculation is deterministic.
9. Cost semantics preserve `DIRECT`, `PLANNING_ONLY`, `QUOTE_REQUIRED`, and `DISPLAY_ONLY`.
10. Findings resolve to rule IDs and source IDs.
11. Streamlit and FastAPI share the same core business services.
12. Real API keys never enter Git.
13. Storage, model provider, case memory, UI, and API can evolve independently.
14. One complete P0 product flow works before adding breadth or UI polish.

---

## 33. Final Architecture Decision

ImportReady AI MVP will use a **modular monolith**:

```text
Streamlit UI
+ FastAPI interface
+ Strands Agent orchestration
+ deterministic Python services
+ structured CaseState
+ approved local JSON knowledge
```

A modular monolith is one deployable application with clearly separated internal responsibilities.

This gives the hackathon MVP:

- fast development
- easy local testing
- interactive demo access
- API-based programmatic testing
- controlled model usage
- traceable compliance results

while keeping clear paths for post-hackathon expansion into persistent memory, databases, additional categories and jurisdictions, richer landed-cost calculation, multi-source policy monitoring, and production APIs.
