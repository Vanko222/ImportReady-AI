# P3.1 Multi-Provider Architecture — Implementation Plan (REVISED)

Status: **P3.1 plan APPROVED · implementation COMPLETED · human code review COMPLETED · offline
regression suite PASSED.** The architecture and decisions below are the approved design, unchanged.
This revision aligns the plan with the current verified project state and replaces the provider-level
whitelist with an **exact Provider + Model combination** policy.

Revision date: 2026-09-13 · Supersedes the first version of this file.

## Implementation result (P3.1 complete)

- Modified production file: `src/agent/model_factory.py`
- Modified test file: `tests/test_model_factory.py`
- Provider-specific tests: **34 passed / 0 failed** (`python -m pytest tests/test_model_factory.py -q`)
- Full offline suite: **436 passed / 0 failed** (`python -m pytest -q`)
- `RuntimeProviderConfig.credential` is protected from `repr`/`str` leakage with `repr=False`.
- **DeepSeek + `deepseek-flash` remains `EXPERIMENTAL`** until the separate live 10-gate provider
  certification is completed. No live provider certification has been performed or claimed.
- `verified_combinations()` currently exposes **no** Consumer model combinations.
- **Bedrock has no approved exact model combination yet**: the provider is registered, but onboarding
  still requires registering one exact model id as `EXPERIMENTAL`.
- **R-ELEC-002 remains unchanged.**
- **R-ELEC-012 remains explicitly deferred** — no transportation trigger, no transport inference.
- Registered policy objects reflect the implementation: `none` (VERIFIED, never exposed),
  `deepseek-chat` (`UNSUPPORTED`), `model_factory.py` at 359 lines (past the ~220-line
  placement-revisit trigger, but no new module was created in this task).

---

## A. Correct current baseline

| Item | Value | How verified |
|---|---|---|
| Full offline suite | **410 passed / 0 failed** | `python -m pytest -q` |
| HEAD | `5b892b3` "Implement applicability engine and Phase 2C integration" | `git log --oneline -3` |
| Working tree | Clean except this plan file (`?? P3_1_Multi_Provider_Architecture_Plan.md`) | `git status --short` |
| `src/agent/model_factory.py` | 53 lines, `build_model()` env-driven | file read |
| `tests/test_model_factory.py` | 134 lines, **8 tests** | file read |
| Installed: strands-agents | 1.55.1 | `importlib.metadata` |
| Installed: openai | 2.54.0 | `importlib.metadata` |
| Installed: boto3 / botocore | 1.43.91 / 1.43.91 (**transitive, not declared**) | `importlib.metadata` |
| Declared deps | `pydantic>=2.7,<3`, `pytest>=8.0,<10`, `strands-agents>=1.55,<2`, `openai>=1.68.0,<3.0.0` | `requirements.txt` |
| Canonical DeepSeek model | `deepseek-flash` | project state |
| Canonical DeepSeek base URL | `https://api.deepseek.com` | `model_factory.py:47` |
| Process credential variable | `API_KEY` (provider-neutral by design) | `model_factory.py:38` |

## B. Correct DeepSeek status

**The previous DeepSeek HTTP 400 compatibility problem is RESOLVED.** The unsupported
`params={"extra_body": {"thinking": ...}}` argument was removed from `build_model()`; that removal
is in the committed tree and is guarded by a regression assertion
(`tests/test_model_factory.py:87` — `assert "params" not in model.kwargs`).

Evidence already obtained for **DeepSeek + `deepseek-flash`**:

| # | Certification stage | Status |
|---|---|---|
| 1 | Minimal direct API request (`model` + `messages` only) → HTTP 200, assistant response returned | **PASS** |
| 2 | Real Strands model request through the application → HTTP 200 | **PASS** |
| 3 | Tool schema registration (the two Agent tools) | **PASS** |
| 4 | Tool selection — model chose `analyze_product` | **PASS** |
| 5 | Tool execution — `analyze_product` ran successfully | **PASS** |
| 6 | Canonical tool-result preservation — deterministic Applicability output produced; R-ELEC-002 gave the correct canonical result when the required USER facts were supplied | **PASS** |
| 7 | Full multi-turn → final-answer certification | **PENDING** |

**Remaining DeepSeek item.** A later multi-turn stage produced this warning:

> `reasoningContent is not supported in multi-turn conversations with the Chat Completions API.`

This does **not** re-open the 400 issue, and it does **not** break single-turn tool use. It means
the complete *tool-result → multi-turn → final-answer* path still needs to be certified before the
combination can be marked `VERIFIED`.

Explicit guidance for the plan's readers:

- `deepseek-flash` is the **current canonical test model**. Keep it.
- Do **not** recommend retrying `deepseek-chat`.
- Do **not** recommend returning to `deepseek-v4-flash`.
- Do **not** describe the 400 problem as unresolved.
- Do **not** mark the full `deepseek` + `deepseek-flash` combination as completely certified until
  the P3 certification suite passes end-to-end.

## C. Current architecture summary

**Single provider entry point.** `src/agent/model_factory.py` is the *only* place in `src/` that
reads provider configuration. A repo-wide search for `MODEL_PROVIDER|MODEL_ID|AWS_REGION|API_KEY|environ`
returns hits only in `model_factory.py` (lines 21–53) and `config.py` (which reads only
`IMPORTREADY_DATA_DIR`).

| Aspect | Current state |
|---|---|
| Selection | `build_model()` string-compares `MODEL_PROVIDER`: `""`/`none` → `None`; `bedrock`; `deepseek`; otherwise `ValueError` |
| Config entry | `MODEL_PROVIDER`, `MODEL_ID`, `AWS_REGION` (bedrock), `API_KEY` (deepseek) — all read inside `build_model()` |
| Call site | `app.py:387` `_build_model_or_none()` → `main():508`; `ValueError` → `ERROR [model_unavailable]`, exit 3 |
| Model usage | `app.py` treats the model as opaque `Any`: `Agent(model=model, …)` at `app.py:258` and `app.py:299`; `model is None` selects the offline deterministic path |
| Provider-neutral | `Orchestrator`, `AnalysisService`, `ApplicabilityEngine`, both Agent tools, `CaseState`, prompts, `config.py` — **zero** provider imports |
| Tests | `tests/test_model_factory.py` monkeypatches `strands.models.openai.OpenAIModel` / `strands.models.bedrock.BedrockModel` |
| Adapters available | Strands ships `anthropic`, `bedrock`, `gemini`, `litellm`, `llamaapi`, `llamacpp`, `mistral`, `ollama`, `openai`, `openai_responses`, `routing`, `sagemaker`, `writer` |
| Docs | `ARCHITECTURE.md:41` (provider adapters replaceable), `:138` (core must not depend on one provider), `:1191` (`ModelGateway → multi-provider routing`) |

**Key finding (unchanged):** ImportReady does not need to write provider clients — Strands already
supplies them. P3.1 is a **registry + policy + status** layer, not a new HTTP client layer.

## D. Existing provider coupling

All coupling is confined to **one production file**:

| # | Coupling | Location |
|---|---|---|
| 1 | Bedrock branch (env + constructor) | `model_factory.py:25–34` |
| 2 | DeepSeek branch + hardcoded literal `https://api.deepseek.com` | `model_factory.py:36–51` |
| 3 | Hardcoded supported-provider list inside the error string | `model_factory.py:53` — will drift as providers are added |
| 4 | Tests bind to Strands module paths | `tests/test_model_factory.py:51,76` |
| 5 | Doc/code drift: `ARCHITECTURE.md:796` documents `MODEL_API_KEY`; code uses `API_KEY` | documentation only |

Already provider-neutral, needing no change: orchestrator, analysis service, applicability engine,
the two Agent tools, `CaseState`/`ProductFact`, prompts, `config.py`, and `app.py`'s generic
`Agent(model=model)` usage.

## E. Revised Provider Registry design

**Conceptual shape (NOT implemented):**

```
Provider Registry
   └── provider descriptor
         ├── approved provider identity (provider_id, display name)
         ├── provider builder (constructs the Strands model object)
         ├── credential requirements (env var names / credential type)
         ├── endpoint strategy (base URL owned by ImportReady — not user-supplied)
         ├── required runtime configuration (e.g. AWS_REGION)
         └── approved model combinations
                └── exact combination record
                      ├── provider_id
                      ├── model_id            ← exact, no wildcards
                      ├── compatibility_status
                      ├── ui_exposed
                      ├── certification metadata / reference
                      └── notes
```

**The Registry owns**, and callers may not override: approved provider identity, the endpoint/base-URL
strategy, the allowed model IDs, the provider-specific constructor, the credential type/scope, and
compatibility/certification metadata.

`MODEL_ID` is a **lookup key into an approved combination** — never a free-form value that grants
access to an arbitrary model.

**Design constraint:** keep this minimal. No plugin loader, no dynamic discovery, no config DSL, no
per-provider class hierarchy beyond what the descriptor needs.

**Placement:** initially inside `src/agent/model_factory.py` (see section R for the comparison with a
separate module).

## F. Exact Provider + Model compatibility design

Certification and demo exposure operate on the **exact `(provider_id, model_id)` pair**, never on the
provider alone.

> `deepseek` is a provider. That does **not** mean every `deepseek` model is certified.
> `deepseek` + `deepseek-flash` is one independently-certified combination.

Proposed structures (P3.1 implements these; nothing is implemented yet):

```python
class CompatibilityStatus(str, Enum):
    VERIFIED     = "VERIFIED"       # full E2E certification passed + human sign-off
    EXPERIMENTAL = "EXPERIMENTAL"   # registered for developer/certification use; never consumer-exposed
    UNSUPPORTED  = "UNSUPPORTED"    # recognised but deliberately refused (informative error)


@dataclass(frozen=True)
class ModelCombination:
    provider_id: str
    model_id: str                       # exact
    compatibility_status: CompatibilityStatus
    ui_exposed: bool = False            # consumer-visible only when True AND VERIFIED
    certification_ref: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    display_name: str
    credential_scope: tuple[str, ...]   # which credential inputs are handed to this builder
    runtime_env: tuple[str, ...]        # required runtime configuration names
    base_url: str | None                # ImportReady-owned; never caller-supplied
    build: ProviderBuilder              # (combination, runtime_fields, credential) -> model object
    combinations: tuple[ModelCombination, ...]


PROVIDER_REGISTRY: dict[str, ProviderDescriptor] = {...}
COMBINATION_INDEX: dict[tuple[str, str], ModelCombination] = {...}   # derived


def resolve_combination(provider_id: str, model_id: str) -> ModelCombination | None: ...
def verified_combinations() -> list[dict]: ...        # consumer picker source
def provider_status_report() -> list[dict]: ...       # diagnostics; never contains secret values
```

Rules:

1. A missing combination resolves to `None` → treated as **not consumer-eligible**.
2. A provider-level status alone never authorises an arbitrary model id.
3. `base_url` comes from the descriptor; there is **no** caller-supplied base URL anywhere.
4. Multiple models per provider are supported, each with its own status and `ui_exposed` flag.
5. The offline/`none` entry is registered but **not** `ui_exposed` (section L).
6. The already-resolved `ModelCombination` is the **single source of truth for `model_id`**. A
   provider builder must not independently re-read, accept, or trust a free-form `MODEL_ID` after
   combination resolution.
7. **Builder contract (smallest clean form):**

   ```python
   ProviderBuilder = Callable[[ModelCombination, Mapping[str, str], str | None], object]

   # build(combination, runtime_fields, credential) -> model object
   #   combination    : the already-resolved registered combination (owns model_id)
   #   runtime_fields : provider-scoped NON-SECRET runtime values (e.g. {"AWS_REGION": "us-east-1"})
   #   credential     : provider-scoped secret, or None (e.g. AWS credential chain)
   ```

   The descriptor owns `base_url`; the builder captures it, so **no caller ever supplies a base
   URL** on any path.

   Flow: resolved `ModelCombination` + scoped runtime config + scoped credential → provider builder.

## G. CLI / development path (unchanged, env-driven)

Existing behaviour must keep working:

```text
MODEL_PROVIDER   MODEL_ID   API_KEY            (deepseek)
MODEL_PROVIDER   MODEL_ID   AWS_REGION         (bedrock)
MODEL_PROVIDER=none                            (offline deterministic path)
```

`build_model()` keeps its signature `() -> Model | None` and its current error semantics, so
`src/agent/app.py` requires **no change**. Intended revised behaviour:

1. `provider = env.get("MODEL_PROVIDER", "none").strip().lower()`
2. `""` / `none` → `None`
3. unknown provider → `ValueError` listing **registry keys** (derived, not hardcoded)
4. missing required runtime config → `ValueError` using the **same substrings** as today
   (`"MODEL_ID and AWS_REGION"`, `"API_KEY"`, `"MODEL_ID"`)
5. `(provider_id, model_id)` is resolved through `COMBINATION_INDEX`. An **unregistered combination
   is refused** — a registered provider does **not** authorise arbitrary `MODEL_ID` values.
6. `EXPERIMENTAL` → allowed (this is the developer / certification path)
7. `UNSUPPORTED` → refused with an informative error
8. `VERIFIED` → allowed. The env path does not require `ui_exposed`; that gate belongs to the
   consumer path (section H)
9. otherwise `descriptor.build(combination, runtime_fields, credential)` — the builder receives the
   **already-resolved combination** and never re-reads `MODEL_ID` (section F)

This path is for **local development, CLI testing, and certification work only**. It never mutates
the environment and never persists a credential.

## H. Session-safe Consumer / BYOK runtime path

A hosted Consumer UI **must not** mutate shared process environment variables per session:

```python
# FORBIDDEN in the hosted consumer path:
os.environ["API_KEY"] = user_entered_key      # shared process state → cross-session credential risk
```

Instead, credentials travel as an explicit, request-scoped object:

```python
@dataclass(frozen=True)
class RuntimeProviderConfig:
    provider_id: str
    model_id: str
    credential: str | None = None                 # BYOK secret; request-scoped only
    runtime_fields: Mapping[str, str] = field(default_factory=dict)   # e.g. {"AWS_REGION": ...}


def build_model_from_runtime_config(config: RuntimeProviderConfig):
    ...
```

`build_model_from_runtime_config` must:

1. resolve `(provider_id, model_id)` through `COMBINATION_INDEX`;
2. require `compatibility_status == VERIFIED` **and** `ui_exposed is True` — otherwise `ValueError`;
3. obtain `base_url` from the descriptor, never from the caller;
4. pass only that provider's declared `credential_scope` + `runtime_fields` to that provider's builder;
5. never write to `os.environ`;
6. never persist, log, return, or embed the credential anywhere.

**Scope decision (final — no contradiction).** P3.1 **does** implement the request-scoped
`RuntimeProviderConfig` (or an equivalently minimal named structure) together with
`build_model_from_runtime_config()` and its offline tests. P3.1 does **not** implement Consumer Web
UI wiring — the browser layer that would *call* this API belongs to the later UI workstream.
Implementing the API now is deliberate: it makes session-safe credential handling the only available
path, so a later `os.environ["API_KEY"] = ...` shortcut is never introduced.

## I. Competition Demo credential/model path

Two distinct judge-facing modes:

**Competition Demo Mode (default).** The judge opens the hosted browser UI; no Python, no Git, no
Docker, and **no personal API key required**. The server uses **one already-certified combination**
selected by configuration. The judge never sees a provider picker and never supplies a credential.

**BYOK Advanced Mode (optional).** The picker is populated **exclusively** from
`verified_combinations()` — exact `VERIFIED` + `ui_exposed` pairs. Arbitrary model IDs and arbitrary
base URLs are never offered. The user's key is used through the session-safe path in section H.

## J. VERIFIED / EXPERIMENTAL / UNSUPPORTED semantics

| Status | Meaning | Consumer (`ui_exposed`) | Developer/certification |
|---|---|---|---|
| `VERIFIED` | Passed **all** certification gates **and** has human sign-off recorded | Allowed only if `ui_exposed=True` | Allowed |
| `EXPERIMENTAL` | Registered so it can be tested; not certified | **Never exposed** | Allowed, deliberately |
| `UNSUPPORTED` | Recognised name, deliberately refused (e.g. a retired model id) | Never | Refused, with an informative error |

Transition rule: `EXPERIMENTAL → VERIFIED` only after the full checklist (section M) passes **and** a
human records the sign-off in the certification record (section N). There is no automatic promotion.
`UNSUPPORTED` entries exist so that errors are informative rather than "unknown provider".

## K. `ui_exposed` semantics

- `ui_exposed` is a property of an **exact model combination**, not of a provider.
- `verified_combinations()` returns only combinations where
  `compatibility_status == VERIFIED and ui_exposed is True`.
- Default is `ui_exposed = False`; exposure is an explicit, deliberate opt-in per combination.
- A `VERIFIED` combination may still have `ui_exposed=False` (e.g. certified but withheld from the
  consumer picker for cost or policy reasons).
- The consumer UI must never construct its picker from the environment or from free-text input.

## L. Offline / `none` behaviour

- Offline deterministic mode **remains functional and unchanged** (`MODEL_PROVIDER=none` → `None` →
  the deterministic path in `app.py`, including the fallback path).
- If `none` appears in the registry at all, it must be `ui_exposed=False`, and
  `verified_combinations()` must exclude it — `none` is an internal mode, not a provider a consumer
  selects.
- Offline mode must never be removed and must never be presented as a normal provider choice.

## M. Provider certification process

Two strictly separated layers.

**Layer 1 — offline, deterministic, in the default pytest suite.** No network. Extends
`tests/test_model_factory.py`:

- registry integrity: every descriptor carries the required fields; `COMBINATION_INDEX` is consistent
  with the descriptors' `combinations`
- status semantics: `UNSUPPORTED` refused; `EXPERIMENTAL` allowed on the **developer** path but
  **absent** from `verified_combinations()` and **refused** on the **consumer** path; `VERIFIED` +
  `ui_exposed=True` allowed on the consumer path; `VERIFIED` + `ui_exposed=False` refused there
- **unregistered combination refused on every path** — both `build_model()` and
  `build_model_from_runtime_config()`
- unknown-provider error derived from the registry keys
- unchanged missing-env errors and error substrings
- **builder contract**: the builder receives the resolved combination, never re-reads `MODEL_ID`,
  and no caller-supplied `base_url` reaches any path
- no-secret-leak in error strings; cross-provider credential isolation
- `none` / offline excluded from `verified_combinations()`

Certification of a **new** model always begins by explicitly registering its exact combination as
`EXPERIMENTAL` — there is no implicit or free-form path to a new model.

**Layer 2 — real-provider certification, opt-in and never part of the offline suite.** A separately
run mechanism (created only in a later, approved step) executes the checklist against a real
provider and produces the certification record. The registry status changes only by human decision
after that record exists.

Checklist (all required before `VERIFIED`):

| # | Gate | Concrete probe |
|---|---|---|
| 1 | Authentication / basic request | minimal `model` + `messages` completion → HTTP 200 |
| 2 | Structured classification | `_suggest_category` returns a valid `CategorySuggestionPayload` |
| 3 | Tool schema acceptance | request carrying both tool schemas accepted (no 400) |
| 4 | Tool selection | model emits an `analyze_product` tool call |
| 5 | Tool execution | tool returns `ok: True` |
| 6 | Tool-result continuation | model consumes the tool result on the following turn |
| 7 | Multi-turn completion | conversation finishes within the turn limit (≤6), no unsupported-field warnings |
| 8 | Final response generation | non-empty final text; `stop_reason` in the normal allowlist |
| 9 | Safe error handling | injected provider error → sanitized `agent_runtime_failure`; no secret leak; no crash |
| 10 | Canonical result preservation | agent-path `ApplicabilityResult` == offline-path result for identical inputs (the Phase 2B §48 invariant) |

ImportReady-specific invariants that must also be asserted during certification:

- R-ELEC-002 canonical behaviour unchanged (True → APPLICABLE / TRIGGER_SATISFIED;
  False → NOT_APPLICABLE / TRIGGER_NOT_SATISFIED; every one of its seven required attributes still
  mandatory)
- `PROPOSED` ≠ EFFECTIVE
- `WATCHLIST` ≠ EFFECTIVE
- `NOT_FOUND` ≠ "not required"
- missing information stays missing (never inferred, never treated as false)
- model output cannot overwrite the deterministic result
- credential-like input is still blocked by the existing safety boundary
- unknown / unsupported category remains safe (no verdict invented)

Gate 7 is precisely where the current DeepSeek `reasoningContent` multi-turn warning must be
resolved before `deepseek` + `deepseek-flash` can be marked `VERIFIED`.

## N. Certification record / evidence format

One record per **exact combination**, human-readable, **never containing secrets**:

| Field | Content |
|---|---|
| `provider_id` | e.g. `deepseek` |
| `model_id` | exact, e.g. `deepseek-flash` |
| `endpoint_strategy` | ImportReady-owned base URL, or AWS region for Bedrock |
| `certification_date` | ISO date |
| `strands_agents_version` | e.g. `1.55.1` |
| `provider_sdk_version` | e.g. `openai 2.54.0`, `boto3 1.43.91` |
| `gates` | per-step `PASS` / `FAIL` for the ten gates |
| `importready_invariants` | per-invariant `PASS` / `FAIL` |
| `overall_status` | `VERIFIED` / `EXPERIMENTAL` / `FAILED` |
| `human_sign_off` | reviewer identifier + date |
| `notes` | warnings, e.g. the multi-turn `reasoningContent` item |

Recommended storage: a dedicated documentation path (for example `docs/certification/<provider_id>__<model_id>.md`),
created only when a certification is actually performed. **Do not** store records in `data/` — that
tree is approved compliance research data with sha256 provenance and must not be mixed with runtime
engineering records. `ModelCombination.certification_ref` points at the record; the record itself is
a later, separately approved artifact.

## O. Bedrock integration strategy

- Design for Bedrock now; **assume no access yet** (account allowlisting has previously been
  pending).
- No live certification claim until real testing succeeds.
- Certification will use an **exact** Bedrock model ID; all Bedrock models are **not** automatically
  supported or exposed.
- The Bedrock combination starts as `EXPERIMENTAL` with `ui_exposed=False`.
- Bedrock credential handling uses the AWS credential chain (not the `API_KEY` variable); the descriptor's
  `credential_scope` is empty and `runtime_env` includes `AWS_REGION`.
- Flag for a later dependency decision: `boto3`/`botocore` are currently present only
  **transitively** (1.43.91). If the Bedrock adapter becomes load-bearing, declaring them explicitly
  in `requirements.txt` should be considered — that is a dependency change and requires separate
  approval. **No dependency change is made in this task.**

## P. Competition-provider integration strategy

```text
Competition API details received
        ↓
Add provider descriptor + exact model combination (EXPERIMENTAL, ui_exposed=False)
        ↓
Run the same certification checklist (section M)
        ↓
Human review + recorded sign-off (section N)
        ↓
VERIFIED  →  eligible for Competition Demo Mode
```

This must require **no** modification to the Applicability Engine, Risk Engine, Cost Engine, Agent
tools, `ProductFact`, repositories, or the compliance JSON data — because an adapter only produces a
model object that `app.py` already consumes generically.

**An OpenAI-compatible competition endpoint is not automatically compatible.** It must pass the same
certification gates (several providers accept the chat schema but differ on tool-calling, structured
output, or multi-turn behaviour — exactly the class of issue already seen with DeepSeek).

## Q. Credential isolation / security rules

A user-supplied or environment credential must **never**:

- mutate global `os.environ` (except the existing, deliberate CLI/dev path reading it)
- enter `.env`
- enter Git
- enter any database
- enter `CaseState` or `ProductFact`
- enter compliance data or evidence
- enter logs
- enter exception messages
- enter provider status reports

Additional rules:

1. **Provider A's credential must never be handed to provider B.** Each descriptor declares its own
   `credential_scope`; the factory passes only those keys, and only to that provider's builder.
2. `RuntimeProviderConfig` is request-scoped and is never stored, cached, or attached to case state.
3. Provider credentials are **not** canonical case data and must never become facts.
4. `provider_status_report()` and `verified_combinations()` expose status/metadata only — no
   credential material, not even redacted fragments.
5. Error paths continue to use the existing sanitisation (`redact_secrets`) so a credential cannot
   surface through `ERROR [model_unavailable]` or agent error messages.
6. No provider adapter may touch applicability logic, risk logic, cost logic, `ProductFact`,
   `CaseState` canonical facts, evidence, or rule lifecycle.

## R. Exact proposed implementation file boundary

### MUST change (P3.1 core)

| File | Change | Est. |
|---|---|---|
| `src/agent/model_factory.py` | Add `CompatibilityStatus`, `ModelCombination`, `ProviderDescriptor`, `ProviderBuilder`, `RuntimeProviderConfig`, `PROVIDER_REGISTRY`, `COMBINATION_INDEX`, `resolve_combination()`; refactor `build_model()`; add `build_model_from_runtime_config()`, `verified_combinations()`, `provider_status_report()`. Preserve the public signature, constructor kwargs and error substrings | ~+150 lines (53 → ~205) |
| `tests/test_model_factory.py` | Adapt the tests that rely on an **unregistered** `MODEL_ID` to a registered `EXPERIMENTAL` fixture combination (note below); keep the remaining tests green; add registry / exact-combination / status / `ui_exposed` / builder-contract / credential-isolation / no-secret-leak tests | ~+110 lines |

**Test adaptation note (required by the final policy).** `test_bedrock_selection` sets
`MODEL_ID="some-model"`, which is now an **unregistered** combination and must be refused. Adapt it
to a registered `EXPERIMENTAL` fixture combination (for example a test-only
`bedrock` + `<exact-test-model-id>` entry registered as `EXPERIMENTAL` in the test's own registry
fixture) while still asserting the exact constructor kwargs
(`{"model_id": ..., "region_name": "us-east-1"}`). `test_deepseek_selection` already uses the
registered `deepseek` + `deepseek-flash` combination and needs no policy change. Do **not** weaken
the policy by adding a permissive escape hatch in production code.

### MAY change (only when justified, later)

| File | Condition |
|---|---|
| `src/agent/app.py` | Only when a UI or CLI needs to *list* providers/combinations. The runtime path needs no change. |
| `ARCHITECTURE.md` | Document the registry, status vocabulary and exact-combination policy; resolve the `MODEL_API_KEY` vs `API_KEY` doc drift (fix the **doc**, not the code). |
| `.env.example` | Add provider variable names (no values). |
| `requirements.txt` | Only if Bedrock is certified and boto3/botocore must be declared explicitly — dependency change, approval required. |
| `docs/certification/*.md` | New certification records, created only at certification time (section N). |

### PROTECTED — must not change

`src/services/applicability.py` · `src/services/analysis.py` · `src/services/orchestrator.py` ·
`src/agent/tools.py` · `src/agent/prompts.py` · `src/state.py` · `src/models.py` ·
`src/repositories/*` · `src/config.py` · `data/*.json`

### Placement decision: keep in `model_factory.py` vs a new module

| Criterion | (a) Extend `model_factory.py` | (b) New `src/agent/providers.py` |
|---|---|---|
| Files touched | 1 production + 1 test | 2 production + 1 test |
| Existing import path | Preserved (`from src.agent.model_factory import build_model`) | Must be preserved via re-export or the import path changes |
| Diff size | Smaller | Larger |
| Separation of concerns | Slightly weaker at ~185 lines | Cleaner if the file keeps growing |
| Risk to the 8 existing tests | Lower | Slightly higher (import-path churn) |

**Recommendation: (a)** for P3.1 — smallest boundary and lowest risk. Revisit **(b)** only if
`model_factory.py` exceeds roughly 220 lines or a second consumer needs the registry independently.
No new module is to be created in this task.

## S. Protected files / invariants

Unchanged and non-negotiable:

- LLM = orchestration / natural-language understanding / explanation; **never** the compliance authority.
- Canonical compliance decisions remain deterministic.
- Protected behaviour: `ProductFact` boundaries, `evidence_status` semantics, `rule_status`
  semantics, `applicability_status` semantics, the Applicability Engine, repository/data authority,
  the human-confirmation boundary, and the safety / hallucination boundaries.
- The two Agent tool signatures stay exactly `analyze_product(product_description)` and
  `get_compliance_evidence(rule_id)`.
- **R-ELEC-002** keeps its existing implemented behaviour.
- **R-ELEC-012** remains explicitly **deferred**: do not implement or simplify its transportation
  trigger, and never infer `lithium battery == transportation applicability`.
- No provider adapter may touch applicability, risk, cost, `ProductFact`, canonical `CaseState`
  facts, evidence, or rule lifecycle.

## T. Breaking-change risks

1. **The stricter combination policy changes behaviour that existing tests rely on.**
   `test_bedrock_selection` currently sets `MODEL_ID="some-model"` (now an **unregistered**
   combination), and `test_deepseek_selection` asserts exact DeepSeek constructor kwargs. Under the
   final policy an unregistered combination is refused on **every** path. *Mitigation:* adapt those
   tests to a **registered `EXPERIMENTAL` fixture combination**, preserve the exact constructor
   kwargs and error substrings, and do **not** add a permissive escape hatch to production code.
2. **Combination enforcement could break the currently working DeepSeek run or the certification
   path if it were applied inconsistently.** *Mitigation:* the policy is uniform on every path —
   **unregistered combinations are refused everywhere**. The developer/certification path stays
   usable **not** by bypassing enforcement, but because **registered `EXPERIMENTAL` combinations are
   permitted** there (registered `VERIFIED` combinations are permitted as well). Only the consumer
   runtime adds the stricter requirement of `VERIFIED` **and** `ui_exposed=True`. A new model is
   therefore onboarded by registering its exact combination as `EXPERIMENTAL` — never by relaxing
   the check.
3. **`none`/offline leaking into the consumer picker.** *Mitigation:* `ui_exposed=False` plus
   explicit exclusion in `verified_combinations()`.
4. **Provider-level thinking ("DeepSeek is VERIFIED, so any DeepSeek model is fine").**
   *Mitigation:* status lives on the exact combination only.
5. **Cross-provider credential leakage** — the main new security risk. *Mitigation:* per-descriptor
   `credential_scope`, only-declared-keys handoff, isolation test, no logging/persistence, and no
   credential in status reports.
6. **`reasoningContent` multi-turn warning** may indicate a real gap in the final-answer path.
   *Mitigation:* gate 7 stays `PENDING` until certified; do not claim full certification.
7. **Bedrock access unconfirmed.** Never assume availability; keep the combination `EXPERIMENTAL`.
8. **boto3/botocore are transitive.** A future Strands change could silently break the Bedrock
   adapter; declaring them is a later, approval-gated dependency decision.
9. **Arbitrary base URL or model ID reaching the consumer path.** *Mitigation:* endpoint is
   ImportReady-owned per descriptor; consumer path requires an exact registered combination.
10. **Doc drift** (`MODEL_API_KEY` vs `API_KEY`): fix the documentation; renaming the runtime
    variable would itself be a breaking configuration change.
11. **A competition/OpenAI-compatible provider being assumed compatible.** *Mitigation:* mandatory
    certification (section P).

## U. Updated implementation order

1. Freeze current working `build_model()` behaviour, then adapt the tests that rely on an
   **unregistered** `MODEL_ID` to a registered `EXPERIMENTAL` fixture combination.
2. Add minimal registry/status structures (`CompatibilityStatus`, `ModelCombination`,
   `ProviderDescriptor`, `ProviderBuilder`, `PROVIDER_REGISTRY`, `COMBINATION_INDEX`).
3. Add the exact Provider + Model combination policy and `resolve_combination()` — an unregistered
   combination is refused on **every** path; a new model is onboarded by explicitly registering its
   exact combination as `EXPERIMENTAL`.
4. Add the session-safe `RuntimeProviderConfig` + `build_model_from_runtime_config()`
   (request-scoped credential; `VERIFIED` + `ui_exposed` only; never mutates `os.environ`).
   **This API is P3.1 scope.** The Consumer Web UI that would call it is **not**.
5. Add credential-isolation, exact-combination and builder-contract tests.
6. Add read-only, UI-safe helpers (`verified_combinations()`, `provider_status_report()`).
7. Run the full offline regression suite (expect 410 + new, 0 failures).
8. Create and approve the separate real-provider certification mechanism (Layer 2).
9. Certify **DeepSeek + `deepseek-flash`** end-to-end — its exact combination is registered as
   `EXPERIMENTAL` first; this pass resolves the pending multi-turn gate.
10. Certify the exact **Bedrock** model when account/model access becomes available.
11. Certify the **competition** provider/model when supplied.
12. Only then expose `VERIFIED` + `ui_exposed` combinations in the Consumer UI.

**None of these steps is started.** Steps 8–12 depend on approvals and on external availability.

## V. Explicit deferred items

- **No implementation** of any part of P3.1 (this is a plan).
- **R-ELEC-012** remains deferred; no transportation trigger, no transport inference.
- **Risk Engine**, **Cost Engine**, **Action Plan**, **What-if**, browser UI, API endpoint, and
  packaging are untouched and out of scope.
- **Bedrock** live certification — deferred pending access/allowlisting.
- **Competition provider** descriptor — deferred until the competition supplies API details.
- **Consumer Web UI wiring** — deferred to the UI workstream. P3.1 **does** implement the
  session-safe `RuntimeProviderConfig` / `build_model_from_runtime_config()` API and its offline
  tests; only the browser layer that would call it is out of scope.
- **Real-provider certification mechanism** (Layer 2 script) — deferred to its own approved step.
- **`boto3`/`botocore` explicit declaration** — deferred dependency decision; no dependency change
  now.
- **`ARCHITECTURE.md` / `.env.example` updates** — deferred documentation edits.
- **Certification records** — created only when certifications are actually performed.

## W. Confirmation that no production/test files were modified

> **Superseded — historical record of the pre-implementation state.** P3.1 is now implemented as
> described in "Implementation result (P3.1 complete)" at the top of this file. The statements below
> describe the tree before that implementation and no longer reflect it.

- `git status --short` shows **exactly one** entry: `?? P3_1_Multi_Provider_Architecture_Plan.md`
  (this plan file, untracked).
- `git diff --name-only` → **empty**. No tracked file is modified.
- No production file modified: `src/**` untouched (`model_factory.py` still 53 lines,
  `app.py`, `applicability.py`, `analysis.py`, `orchestrator.py`, `tools.py`, `prompts.py`,
  `state.py`, `models.py`, repositories, `config.py` all unchanged).
- No test file modified: `tests/**` untouched (`tests/test_model_factory.py` still 134 lines,
  8 tests; the offline suite remains **410 passed**).
- No file created other than this plan file; nothing renamed, moved, or deleted.
- No package installed, upgraded, or removed; no dependency change.
- No `git commit`, `push`, `fetch`, `pull`, `reset`, `checkout`, `rebase`, or `merge`; the plan file
  is **not staged**.
- `.env` was **not** read, printed, inspected, or modified. No API key, AWS credential, token, or
  secret was read or exposed.
- Read-only operations used: file reads (`model_factory.py`, `tests/test_model_factory.py`),
  `git log`, `git status`, and package-version introspection via `importlib.metadata`.

**Plan revised. Awaiting approval before any implementation.**
