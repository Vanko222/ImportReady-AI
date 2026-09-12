# Applicability Engine — Phase 2A Revised Implementation Plan

Documentation only. This document does not authorize implementation. Phase 2A begins only on explicit approval, and then only for the two new files named in section A.

## A. Exact Phase 2A files

| File | Status | Purpose |
|---|---|---|
| `src/services/applicability.py` | **NEW** | Applicability core: enums, result contract, canonical fact indexing + `data_type` validation, gates G0–G6, `TriggerSpec` table + partition validation, minimal deterministic evaluator, reason-code vocabulary |
| `tests/test_applicability.py` | **NEW** | Offline deterministic tests (section F) |

**No existing file is modified in Phase 2A.** Phase 2A is additive-only.

Explicitly untouched:

- `src/models.py`, `src/state.py`
- `src/services/analysis.py`, `src/services/orchestrator.py`, `src/services/classification.py`
- `src/agent/tools.py`, `src/agent/app.py`, `src/agent/prompts.py`
- `src/repositories/*`, `src/config.py`
- `data/*.json`, `requirements.txt`, `.env`, `.gitignore`
- research files, existing documentation files

Hard constraints:

- new dependencies: **none**
- new Agent tools: **none** — tool count stays exactly 2 (`analyze_product`, `get_compliance_evidence`)
- approved JSON edits: **none**
- **Phase 2A changes no existing runtime behavior.** Output remains
  `applicability = NOT_EVALUATED`, `risk = NOT_EVALUATED`, `cost = NOT_AVAILABLE`
- current offline regression baseline **105 passed / 0 failed** must remain green, plus the new tests

Integration direction (Phase 2B, not Phase 2A):

```
Repository ComplianceRule
→ ApplicabilityEngine
→ RuleApplicabilityResult
```

## B. Exact classes and functions

All in `src/services/applicability.py`, in this order. Service models use plain Pydantic
`BaseModel`, matching the convention already used in `src/services/analysis.py`.

### B1. Enums

```python
class ApplicabilityStatus(str, Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NEEDS_INFO = "NEEDS_INFO"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"

class ApplicabilityReasonCode(str, Enum):
    # G0 rule existence / canonical resolution
    UNKNOWN_RULE_ID           = "UNKNOWN_RULE_ID"
    # G1 evidence
    EVIDENCE_NOT_VERIFIED     = "EVIDENCE_NOT_VERIFIED"
    EVIDENCE_NOT_FOUND        = "EVIDENCE_NOT_FOUND"
    # G2 lifecycle
    LIFECYCLE_PROPOSED        = "LIFECYCLE_PROPOSED"
    LIFECYCLE_WATCHLIST       = "LIFECYCLE_WATCHLIST"
    LIFECYCLE_SUPERSEDED      = "LIFECYCLE_SUPERSEDED"
    LIFECYCLE_UNKNOWN         = "LIFECYCLE_UNKNOWN"
    # G3 review-only policy (CONV-GAP-001 / P1)
    P1_REVIEW_ONLY            = "P1_REVIEW_ONLY"
    # G4 structural
    NO_REQUIRED_ATTRIBUTES    = "NO_REQUIRED_ATTRIBUTES"
    # G5 fact validation / completeness
    CONTRADICTORY_FACTS       = "CONTRADICTORY_FACTS"
    INVALID_FACT_VALUE        = "INVALID_FACT_VALUE"
    MISSING_REQUIRED_FACTS    = "MISSING_REQUIRED_FACTS"
    UNTRUSTED_FACT_ORIGIN     = "UNTRUSTED_FACT_ORIGIN"   # supplemental only, never deciding
    # G6 trigger evaluation
    TRIGGER_LOGIC_NOT_MODELED = "TRIGGER_LOGIC_NOT_MODELED"
    TRIGGER_SPEC_INVALID      = "TRIGGER_SPEC_INVALID"
    TRIGGER_SATISFIED         = "TRIGGER_SATISFIED"
    TRIGGER_NOT_SATISFIED     = "TRIGGER_NOT_SATISFIED"

class TriggerOperator(str, Enum):
    EQ = "EQ"      # value == expected
    IN = "IN"      # value in expected (collection)

class TriggerCombine(str, Enum):
    ALL = "ALL"    # the only combine behavior required by the approved specs

class TriggerOutcome(str, Enum):
    TRIGGER_SATISFIED     = "TRIGGER_SATISFIED"
    TRIGGER_NOT_SATISFIED = "TRIGGER_NOT_SATISFIED"

class FactIssueCode(str, Enum):
    UNKNOWN_ATTRIBUTE_ID = "UNKNOWN_ATTRIBUTE_ID"
    INVALID_VALUE        = "INVALID_VALUE"
    CONTRADICTORY_VALUES = "CONTRADICTORY_VALUES"
    UNTRUSTED_ORIGIN     = "UNTRUSTED_ORIGIN"
```

Operator set is closed at exactly `EQ` and `IN`. `TriggerCombine` has exactly one member, `ALL`.
`TriggerOutcome` is closed at exactly two values, so a branch can never express a third
"undecided" outcome. There is **no OR combinator**: `ANY` is deliberately absent from Phase 2A and
may be introduced later only under separate approval if a genuinely necessary rule requires it.
No regex, no `eval`, no arithmetic, no threshold language, no string searching, no
natural-language evaluation, no custom scripting, no LLM.

Fixed outcome mapping:

```python
_OUTCOME_TO_STATUS = {
    TriggerOutcome.TRIGGER_SATISFIED:     ApplicabilityStatus.APPLICABLE,
    TriggerOutcome.TRIGGER_NOT_SATISFIED: ApplicabilityStatus.NOT_APPLICABLE,
}
```

Input-level to rule-level issue mapping (documented, single source):

```python
_ISSUE_TO_REASON = {
    FactIssueCode.INVALID_VALUE:        ApplicabilityReasonCode.INVALID_FACT_VALUE,
    FactIssueCode.CONTRADICTORY_VALUES: ApplicabilityReasonCode.CONTRADICTORY_FACTS,
    FactIssueCode.UNTRUSTED_ORIGIN:     ApplicabilityReasonCode.UNTRUSTED_FACT_ORIGIN,
    # FactIssueCode.UNKNOWN_ATTRIBUTE_ID has no per-rule reason code: global only.
}
```

### B2. Result-contract models

```python
class FactInputIssue(BaseModel):
    attribute_id: str          # the offending input's attribute id (never None)
    issue_code: FactIssueCode
    detail: str                # deterministic type/vocabulary description ONLY
                               # NEVER echoes the submitted raw value

class RuleApplicabilityResult(BaseModel):
    rule_id: str
    applicability_status: ApplicabilityStatus
    reason_codes: list[ApplicabilityReasonCode] = Field(default_factory=list)
    evidence_status: EvidenceStatus | None = None   # None only for the G0 unknown-rule case
    rule_status: RuleStatus | None = None           # None only for the G0 unknown-rule case
    required_attribute_ids: list[str] = Field(default_factory=list)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    evaluated_attribute_ids: list[str] = Field(default_factory=list)

class ApplicabilityResult(BaseModel):
    rules: list[RuleApplicabilityResult] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    input_issues: list[FactInputIssue] = Field(default_factory=list)
```

`evidence_status` and `rule_status` are optional **only** because an unknown rule id has no
canonical rule and therefore no canonical statuses. They are never `None` for a rule that
resolved through G0; synthesizing placeholder statuses would invent data.

### B3. Trigger-spec models

```python
class TriggerCondition(BaseModel):
    attribute_id: str
    operator: TriggerOperator
    expected: bool | str | list[str]     # EQ: scalar; IN: non-empty list of allowed values

class TriggerBranch(BaseModel):
    combine: TriggerCombine                              # must be ALL
    conditions: list[TriggerCondition] = Field(default_factory=list)   # exactly one in Phase 2A
    outcome: TriggerOutcome

class TriggerSpec(BaseModel):
    rule_id: str
    deciding_attribute_ids: list[str] = Field(default_factory=list)    # exactly one in Phase 2A
    rationale: str
    branches: list[TriggerBranch] = Field(default_factory=list)
```

Phase 2A spec form is restricted to **exactly one deciding attribute** and **exactly one
condition per branch**. This is what makes branch-partition validation sound (section B5), and it
is the smallest form that expresses all three approved specs. Multi-condition, multi-attribute or
OR-shaped specs are out of scope and require separate approval.

### B4. Canonical fact indexing and validation

```python
# Explicit, small, reviewable exceptions -- no shape heuristics.
_VOCABULARY_FREE_ATTRIBUTE_IDS: frozenset[str] = frozenset({"A-CMN-003", "A-TOY-024"})

# Canonical boolean domain. NEVER derived from textual allowed_values.
_BOOLEAN_CANONICAL_DOMAIN: frozenset = frozenset({True, False})

class FactIndex(BaseModel):
    usable: dict[str, ProductFact] = Field(default_factory=dict)   # USER origin, valid, single-valued
    contradictory: frozenset[str] = Field(default_factory=frozenset)
    invalid: frozenset[str] = Field(default_factory=frozenset)
    untrusted: frozenset[str] = Field(default_factory=frozenset)   # CLASSIFIER/DERIVED present
    unknown_attribute_ids: frozenset[str] = Field(default_factory=frozenset)
    issues: list[FactInputIssue] = Field(default_factory=list)     # deterministic order

    def canonical(self, attribute_id: str) -> ProductFact | None      # usable only
    def is_usable(self, attribute_id: str) -> bool
    def issue_codes_for(attribute_id: str) -> list[FactIssueCode]

def index_facts(repository: ComplianceRepository, facts: Sequence[ProductFact]) -> FactIndex

# Closed dispatch table. Unknown data_type -> invalid (fail safe).
_VALIDATORS: dict[str, Callable[[Any], str | None]]
def _validate_boolean(value)          -> bool only; rejects "yes"/"no"/1/0/None
def _validate_enum(attribute, value)  -> str in attribute.allowed_values
def _validate_multi_select(attribute, value)
                                      -> list[str]; members in allowed_values unless the
                                         attribute is in _VOCABULARY_FREE_ATTRIBUTE_IDS
def _validate_integer(value)          -> int and not bool
def _validate_number(value)           -> int|float and not bool
def _validate_decimal(value)          -> int|float and not bool
def _validate_text(value)             -> str
def _validate_date(value)             -> datetime.date and not datetime.datetime
def _validate_structured_text(value)  -> str (outer type only)
def _validate_structured_list(value)  -> list (outer type only)

def canonical_domain(attribute: ProductAttribute) -> frozenset | None
    # boolean                      -> {True, False}
    # enum                         -> frozenset(allowed_values)
    # multi_select (vocabulary)    -> frozenset(allowed_values)
    # anything else                -> None  (no closed canonical domain)
```

Validation policy, applied exactly:

| `data_type` | Required canonical Python type | `allowed_values` role |
|---|---|---|
| `boolean` | real `bool` | **ignored** — canonical domain is `{True, False}` |
| `enum` | `str` in `allowed_values` | closed vocabulary, enforced |
| `multi_select` | `list[str]`, members in `allowed_values` | enforced, **except** `_VOCABULARY_FREE_ATTRIBUTE_IDS` |
| `integer` | `int`, **not** `bool` | ignored |
| `number` / `decimal` | `int` or `float`, **not** `bool` | ignored |
| `text` | `str` | ignored |
| `date` | `datetime.date`, **not** `datetime.datetime` | ignored |
| `structured_text` | `str` (outer type only) | ignored — inner semantics not invented |
| `structured_list` | `list` (outer type only) | ignored — inner semantics not invented |
| *(unknown)* | — | **fail safe → invalid** |

`allowed_values` is **not** treated as a closed vocabulary for `text`, `number`, `integer`,
`decimal`, `date`, `structured_text`, or `structured_list`. Several approved entries are
descriptive hints rather than enumerations (e.g. `A-ELEC-005` = `["MHz"]`, `A-CMN-009` =
`["0 and above"]`, `A-CMN-002` = a prose sentence).

`isinstance(True, int)` is `True` in Python, so `integer`/`number`/`decimal` carry an explicit
`not isinstance(value, bool)` guard.

`_VOCABULARY_FREE_ATTRIBUTE_IDS` is exactly `{"A-CMN-003", "A-TOY-024"}` — the only two
`multi_select` attributes whose `allowed_values` is descriptive or empty (`A-CMN-003` holds one
prose string; `A-TOY-024` holds `[]`). It is an explicit pinned frozenset, not a heuristic.

**Missing canonical facts are represented by absence from `FactIndex.usable` — never by a
placeholder value.** Only `FactOrigin.USER` facts may enter `usable`. `FactIndex` may hold values
because it is an internal working index; `ApplicabilityResult` must not (section C).

#### Deterministic mixed-origin handling in `index_facts`

One explicit rule, applied per attribute — no optional behaviour:

1. Collect the submitted facts for the attribute, grouped by origin.
2. The **canonical fact** is the `FactOrigin.USER` facts only.
3. **Identical duplicate `USER` facts** collapse into one canonical fact. No issue.
4. **Conflicting `USER` facts** (two or more distinct values) → the attribute is marked
   `contradictory`, there is **no** canonical fact, and one `CONTRADICTORY_VALUES` input issue is
   recorded.
5. **`CLASSIFIER` / `DERIVED` facts never become canonical and never create a canonical
   contradiction.** A `USER` fact is never invalidated by an untrusted submission.
6. Exactly one `UNTRUSTED_ORIGIN` input issue is recorded for an attribute when untrusted
   submissions were ignored, defined as:
   - no usable `USER` fact exists **and** at least one `CLASSIFIER`/`DERIVED` fact exists, **or**
   - a usable `USER` fact exists **and** at least one `CLASSIFIER`/`DERIVED` fact carries a
     **different** value.
   Untrusted facts that merely repeat the canonical `USER` value are redundant and produce no
   issue, because no information was ignored.
7. Input issues are recorded once per `(attribute_id, issue_code)` pair and ordered
   deterministically by `attribute_id`, then `issue_code`.

### B5. Trigger-spec table and partition validation

```python
TRIGGER_SPECS: dict[str, TriggerSpec]                     # exactly 3 entries
_APPROVED_SPEC_RULE_IDS: frozenset[str] = frozenset(
    {"R-TOY-010", "R-TOY-011", "R-ELEC-002"}
)

def validate_trigger_specs(repository: ComplianceRepository) -> list[str]
def _spec_problems(repository, spec) -> list[str]
def _branch_matched_values(spec, branch) -> frozenset      # canonical deciding values matched
def valid_spec_rule_ids(repository) -> frozenset[str]      # specs that pass validation
```

`validate_trigger_specs` rules:

1. every `spec.rule_id` exists in the repository
2. the spec key set is exactly `_APPROVED_SPEC_RULE_IDS` (no unapproved specs)
3. every `condition.attribute_id` is in that rule's `rule.required_attribute_ids`
4. `deciding_attribute_ids` has exactly one entry, and it equals the branch condition attribute
5. `combine == TriggerCombine.ALL` and every branch has exactly one condition
6. every operator is `EQ` or `IN`; `EQ` expects a scalar; `IN` expects a non-empty list whose
   members are all in the deciding attribute's canonical domain
7. the deciding attribute has a closed canonical domain (`canonical_domain(...) is not None`);
   a deciding attribute on an open type makes the spec invalid
8. **branch partition** — with `D = canonical_domain(deciding_attribute)`:
   - **coverage:** the union of `_branch_matched_values(spec, branch)` over all branches equals `D`
     (no uncovered canonical value)
   - **disjointness:** the branch matched-value sets are pairwise disjoint (no overlapping branches)
   - **no conflicting overlap:** consequently no canonical value maps to two different outcomes
   - equivalently: every canonical value in `D` maps to **exactly one** branch outcome

Rule 8 is **data-type aware**: for boolean attributes `D` is `{True, False}` and is never compared
against textual `allowed_values` such as `"yes"` / `"no"` / `"unknown"`.

The partition property is a validation-time guarantee, not a runtime heuristic. A spec with an
uncovered value, an overlapping branch, or two branches with conflicting outcomes for the same
value is **invalid** and is disabled (section E).

### B6. Minimal evaluator

```python
def _evaluate_condition(index: FactIndex, condition: TriggerCondition) -> bool
def _evaluate_branch(index: FactIndex, branch: TriggerBranch) -> bool
def _resolve_outcome(index: FactIndex, spec: TriggerSpec) -> TriggerOutcome | None
    # collects ALL matching branches; requires exactly one match; returns its outcome.
    # returns None when zero or more than one branch matches.
```

**The evaluator does not use "first matching branch wins."** Because validated specs partition the
canonical domain, exactly one branch matches any canonical deciding value. The evaluator collects
every matching branch and:
- exactly one match → that branch's outcome
- zero matches or more than one match → `None`, which G6 maps to `REVIEW_REQUIRED` /
  `TRIGGER_SPEC_INVALID`

This makes ambiguous authored logic impossible to resolve silently: it fails closed instead.
There is no `UNDECIDED` state, because under the approved strict completeness rule G5 guarantees
every required attribute is present, canonical, valid and non-contradictory before G6 runs.

### B7. Engine

```python
@dataclass
class GateOutcome:
    status: ApplicabilityStatus
    reason_codes: list[ApplicabilityReasonCode]

class ApplicabilityEngine:
    def __init__(self, repository: ComplianceRepository) -> None:
        # self._repository = repository
        # self._conv_gap_review_only_ids: frozenset[str]   (computed once, see B8)
        # self._valid_spec_rule_ids: frozenset[str]        (computed once, see B5)

    @property
    def conv_gap_review_only_ids(self) -> frozenset[str]
    @property
    def valid_spec_rule_ids(self) -> frozenset[str]

    def is_review_only(self, rule: ComplianceRule) -> bool
    def evaluate_rule(self, rule_id: str, index: FactIndex) -> RuleApplicabilityResult
    def evaluate(self, rule_ids: Sequence[str],
                 facts: Sequence[ProductFact]) -> ApplicabilityResult

    # Gate order is fixed. Each returns None to continue, or a GateOutcome.
    # Every gate after G0 receives the CANONICAL rule only.
    def _g0_resolve_canonical(rule_id) -> tuple[ComplianceRule | None, GateOutcome | None]
    def _g1_evidence(canonical_rule) -> GateOutcome | None
    def _g2_lifecycle(canonical_rule) -> GateOutcome | None
    def _g3_review_only(canonical_rule) -> GateOutcome | None
    def _g4_structural(canonical_rule) -> GateOutcome | None
    def _g5_facts(canonical_rule, index) -> GateOutcome | None
    def _g6_trigger(canonical_rule, index) -> GateOutcome
```

**G0 resolves the canonical rule; the caller never supplies rule content.**

```
rule_id
→ repository.get_rule(rule_id)
→ canonical_rule
→ G1–G6 evaluate canonical_rule only
```

- `evaluate_rule` and `evaluate` accept **only `rule_id` strings**, never `ComplianceRule`
  objects. A caller cannot therefore smuggle mutated lifecycle, evidence, or
  `required_attribute_ids` content into the gates: matching on `rule_id` is not sufficient, so no
  caller-supplied rule object is ever trusted.
- If `repository.get_rule(rule_id)` returns `None`, G0 short-circuits with `REVIEW_REQUIRED` /
  `[UNKNOWN_RULE_ID]`, `evidence_status` and `rule_status` are `None`, and G1–G6 are never
  evaluated.
- Tests that need synthetic `UNVERIFIED` / `PROPOSED` / `SUPERSEDED` / `CONFLICT` states construct
  a **controlled test repository** (a small in-memory implementation of the existing
  `ComplianceRepository` `typing.Protocol`) whose `get_rule` returns synthetic canonical rules.
  Mutating a copy of a real rule and passing it to production code is explicitly not the design.
- The engine depends only on `ComplianceRepository` Protocol methods (`get_rule`,
  `get_attribute`, `get_known_gaps`, and so on), never on the concrete JSON repository's
  convenience properties.

### B8. Review-only policy

```python
def _conv_gap_review_only_ids(repository) -> frozenset[str]
    # frozenset(CONV-GAP-001.related_rule_ids) from repository.get_known_gaps()

# G3 decision, evaluated on the CANONICAL rule:
def is_review_only(self, rule: ComplianceRule) -> bool:
    return rule.mvp_priority == "P1" or rule.rule_id in self._conv_gap_review_only_ids
```

Explicit approved provenance:

- `CONV-GAP-001` is read from `repository.get_known_gaps()`; its `mvp_handling` is
  *"Keep P1 rules as review-only escalation records until a separately approved mapping is
  added; do not generate a definitive applicability result from missing fields."*
- `STATUS.md` records *"P1 rules remain review/escalation records where clarification mappings
  are incomplete."*
- G3 applies the conservative union (`CONV-GAP-001` related rule ids **or** canonical
  `mvp_priority == "P1"`). Tests assert the two sources agree in the current approved data, so any
  future divergence is reported rather than silently absorbed.
- The engine must **not** identify review-only rules from the coincidence that
  `clarification_question`, `missing_information_blocks_decision` and
  `runtime_status_if_missing` are all `None`; that is not a stable contract.
- If `CONV-GAP-001` is absent from the repository, the union degrades to the
  `mvp_priority == "P1"` test, which is still an explicit approved policy.

### B9. Deterministic explanations

```python
REASON_EXPLANATIONS: dict[ApplicabilityReasonCode, str]
ISSUE_EXPLANATIONS: dict[FactIssueCode, str]
def explain_reason_codes(codes: Sequence[ApplicabilityReasonCode]) -> list[str]
```

Deterministic module-level text only. No LLM text, no narrative reasoning in the canonical result.

## C. Final result contract

Lean and self-contained. **No raw user fact values, no LLM reasoning, no confidence, no
probability, no score, no free-form explanation.**

```python
class RuleApplicabilityResult(BaseModel):
    rule_id: str
    applicability_status: ApplicabilityStatus
    reason_codes: list[ApplicabilityReasonCode] = Field(default_factory=list)
    evidence_status: EvidenceStatus | None = None
    rule_status: RuleStatus | None = None
    required_attribute_ids: list[str] = Field(default_factory=list)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    evaluated_attribute_ids: list[str] = Field(default_factory=list)

class ApplicabilityResult(BaseModel):
    rules: list[RuleApplicabilityResult] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    input_issues: list[FactInputIssue] = Field(default_factory=list)
```

Rules:

1. `reason_codes` is ordered and **non-empty** for every result.
   **`reason_codes[0]` is always the deciding reason.**
2. `UNTRUSTED_FACT_ORIGIN` is **supplemental provenance, never a deciding status code.**
   Fact-level deciding precedence is strictly:

   ```
   CONTRADICTORY_FACTS  >  INVALID_FACT_VALUE  >  MISSING_REQUIRED_FACTS
   ```

   Worked example — only an untrusted fact exists for a required attribute:

   ```
   reason_codes = [MISSING_REQUIRED_FACTS, UNTRUSTED_FACT_ORIGIN]
   status       = from runtime_status_if_missing (MISSING_REQUIRED_FACTS decides)
   ```
3. `UNTRUSTED_FACT_ORIGIN` never outranks `MISSING_REQUIRED_FACTS` and is never placed at index 0.
4. `evidence_status` and `rule_status` come from the canonical repository rule and are retained so
   each result is self-contained and gate-testable without a join. They are `None` only for the G0
   unknown-rule case. `requirement`, `category`, `source_ids` and `clarification_question` are
   **not** duplicated — they already exist in `verified.compliance_information`.
5. `missing_attribute_ids` lists this rule's canonical `required_attribute_ids` that are not
   usable. Under strict completeness a definitive verdict implies `missing_attribute_ids == []`.
6. `evaluated_attribute_ids` lists the attributes the approved `TriggerSpec` actually read.
7. `counts` always contains all four status keys (zeros included) and sums to `len(rules)`.
8. `input_issues` is global, deterministic (sorted by `attribute_id`, then `issue_code`, one entry
   per pair), and is never broadcast into per-rule statuses.
9. `FactInputIssue.detail` describes defects by type/vocabulary only, e.g.
   `"expected bool for data_type boolean, got str"`. It **never** echoes the submitted value.
10. Mutable defaults are always `Field(default_factory=...)` — never bare `=[]` / `={}`.
    `FactIndex` uses `default_factory=frozenset` for its set fields.

## D. Final gate precedence

Order is fixed and total. The first gate that fires decides the status; later gates are not
evaluated for that rule. **Rule existence and canonical resolution is G0, and all later gates
operate on the canonical repository rule.**

```
G0 Canonical rule resolution
→ G1 Evidence
→ G2 Lifecycle
→ G3 Review-only policy
→ G4 Structural
→ G5 Fact validation / completeness
→ G6 Trigger evaluation
```

| Gate | Condition (on the canonical rule) | Status | `reason_codes` |
|---|---|---|---|
| **G0** | `repository.get_rule(rule_id)` returns `None` | `REVIEW_REQUIRED` | `[UNKNOWN_RULE_ID]` |
| G0 | canonical rule resolved | *continue with canonical rule* | — |
| **G1** | `evidence_status == VERIFIED` | *continue* | — |
| G1 | `== UNVERIFIED` or `== CONFLICT` | `REVIEW_REQUIRED` | `[EVIDENCE_NOT_VERIFIED]` |
| G1 | `== NOT_FOUND` | `REVIEW_REQUIRED` | `[EVIDENCE_NOT_FOUND]` |
| **G2** | `rule_status == EFFECTIVE` | *continue* | — |
| G2 | `== PROPOSED` | `REVIEW_REQUIRED` | `[LIFECYCLE_PROPOSED]` |
| G2 | `== WATCHLIST` | `REVIEW_REQUIRED` | `[LIFECYCLE_WATCHLIST]` |
| G2 | `== SUPERSEDED` | `REVIEW_REQUIRED` | `[LIFECYCLE_SUPERSEDED]` |
| G2 | `== UNKNOWN` | `REVIEW_REQUIRED` | `[LIFECYCLE_UNKNOWN]` |
| **G3** | `is_review_only(canonical_rule)` (CONV-GAP-001 union P1) | `REVIEW_REQUIRED` | `[P1_REVIEW_ONLY]` |
| **G4** | `required_attribute_ids == []` | `REVIEW_REQUIRED` | `[NO_REQUIRED_ATTRIBUTES]` |
| **G5** | any required attribute has contradictory USER facts | `REVIEW_REQUIRED` | `[CONTRADICTORY_FACTS, ...]` |
| G5 | any required attribute value fails canonical `data_type` validation | `REVIEW_REQUIRED` | `[INVALID_FACT_VALUE, ...]` |
| G5 | any required attribute is not usable (missing, or only untrusted) | `NEEDS_INFO` if `runtime_status_if_missing == "NEEDS_INFO"`; otherwise `REVIEW_REQUIRED` | `[MISSING_REQUIRED_FACTS, (UNTRUSTED_FACT_ORIGIN)]` |
| G5 | all required attributes usable | *continue* | — |
| **G6** | no `TriggerSpec` for `rule_id` | `REVIEW_REQUIRED` | `[TRIGGER_LOGIC_NOT_MODELED]` |
| G6 | spec exists but failed validation (including a non-partition spec) | `REVIEW_REQUIRED` | `[TRIGGER_SPEC_INVALID]` |
| G6 | exactly one branch matches, outcome `TRIGGER_SATISFIED` | `APPLICABLE` | `[TRIGGER_SATISFIED]` |
| G6 | exactly one branch matches, outcome `TRIGGER_NOT_SATISFIED` | `NOT_APPLICABLE` | `[TRIGGER_NOT_SATISFIED]` |
| G6 | zero or more than one branch matched (defensive; unreachable for validated specs) | `REVIEW_REQUIRED` | `[TRIGGER_SPEC_INVALID]` |

G5 detail — evaluated over **every** canonical `rule.required_attribute_ids`, not only deciding
attributes:

- per-attribute classification, in this order: contradictory → `CONTRADICTORY_FACTS`;
  invalid → `INVALID_FACT_VALUE`; not usable → `MISSING_REQUIRED_FACTS`
  (plus `UNTRUSTED_FACT_ORIGIN` when untrusted submissions were ignored for that attribute)
- decision precedence: `CONTRADICTORY_FACTS` > `INVALID_FACT_VALUE` > `MISSING_REQUIRED_FACTS`
- contributing codes are appended after the deciding code in that same fixed precedence order
- if `runtime_status_if_missing` is absent, fall back to `REVIEW_REQUIRED`
- defensive: a required attribute id that does not resolve in the repository is treated as
  not usable → `[MISSING_REQUIRED_FACTS]` with `REVIEW_REQUIRED` (unreachable in practice, because
  `repository.validate_references()` already guarantees rule→attribute references resolve)
- a valid `USER` fact plus an ignored conflicting `CLASSIFIER`/`DERIVED` fact leaves the attribute
  **usable**: an `UNTRUSTED_ORIGIN` global input issue is recorded, but G5 does not fail and the
  dependent rule can still reach a verdict

Gate-ordering consequences that are deliberate:

- **G0 first, and canonical.** An unknown rule fails safe before evidence, lifecycle, facts or
  trigger logic is consulted, and no caller-supplied rule content is ever trusted. Never
  `APPLICABLE`, never `NOT_APPLICABLE`, no exception raised.
- **G2 before G3.** `R-ELEC-018` (WATCHLIST) and `R-ELEC-019` (PROPOSED) are both P1, so both gates
  would fire; the specific lifecycle reason is reported instead of a generic P1 reason.
- **Strict completeness (D-1).** `APPLICABLE` and `NOT_APPLICABLE` are emitted only when **all**
  canonical `rule.required_attribute_ids` are present, `FactOrigin.USER`, valid and
  non-contradictory. A definitive verdict is never permitted while any required attribute is
  missing/untrusted/invalid/contradictory — even when the deciding attribute is present.
- **Only G6 can produce `APPLICABLE` / `NOT_APPLICABLE`.** There is no other code path.
- Only `EFFECTIVE` + `VERIFIED` rules can reach G6, so only they can reach a canonical verdict.

## E. Exact 3 TriggerSpecs in human-readable form

Exactly three. All use `combine = ALL`, one deciding attribute, and one condition per branch.
No other spec may appear without explicit approval.

### TS-1 — `R-TOY-010` Magnet safety for toys

- **Deciding attribute:** `A-TOY-012` `magnet_present` (boolean; canonical domain `{True, False}`)
- **required_attribute_ids:** `A-TOY-001, A-TOY-005, A-TOY-012, A-TOY-013, A-TOY-014`
- **Branches:**
  1. `ALL [ A-TOY-012 EQ True ]` → `TRIGGER_SATISFIED` (→ `APPLICABLE`)
  2. `ALL [ A-TOY-012 EQ False ]` → `TRIGGER_NOT_SATISFIED` (→ `NOT_APPLICABLE`)
- **Partition:** `{True, False}` maps onto exactly two disjoint branches; complete and unambiguous.
- **Strict rule applies:** a definitive verdict requires all five required attributes to pass G5.
- **Rationale:** the rule's trigger is "Toy contains magnets"; `magnet_present` is the
  rule-declared attribute whose meaning is exactly that subject matter.

### TS-2 — `R-TOY-011` Battery-operated toy requirements

- **Deciding attribute:** `A-TOY-015` `battery_operated` (boolean; canonical domain `{True, False}`)
- **required_attribute_ids:** `A-TOY-001, A-TOY-005, A-TOY-015, A-TOY-016, A-TOY-017`
- **Branches:**
  1. `ALL [ A-TOY-015 EQ True ]` → `TRIGGER_SATISFIED`
  2. `ALL [ A-TOY-015 EQ False ]` → `TRIGGER_NOT_SATISFIED`
- **Partition:** complete and disjoint.
- **Strict rule applies:** all five required attributes must pass G5.
- **Rationale:** trigger is "Toy is battery operated, including toys using button/coin cells".
  The `A-TOY-016` button-cell case is subsumed by `battery_operated = True` and is deliberately
  not a separate branch.

### TS-3 — `R-ELEC-002` FCC certification for intentional radiators

- **Deciding attribute:** `A-ELEC-002` `intentional_rf_transmitter`
  (boolean; canonical domain `{True, False}`)
- **required_attribute_ids:** `A-ELEC-002, A-ELEC-003, A-ELEC-007, A-ELEC-008, A-ELEC-009, A-ELEC-010, A-ELEC-021`
- **Branches:**
  1. `ALL [ A-ELEC-002 EQ True ]` → `TRIGGER_SATISFIED`
  2. `ALL [ A-ELEC-002 EQ False ]` → `TRIGGER_NOT_SATISFIED`
- **Partition:** complete and disjoint.
- **No `"unknown"` branch.** The canonical boolean domain has no "unknown" value. If no trusted
  canonical boolean fact exists, the attribute is **missing** → missing-fact handling via
  `runtime_status_if_missing` (`R-ELEC-002` → `REVIEW_REQUIRED`). A future CLI/UI may let the human
  choose "unknown", but the application boundary must represent that as **absence of a
  `ProductFact`**, never as `ProductFact(value="unknown")`.
- **Strict rule applies:** all seven required attributes must pass G5.
- **Rationale:** trigger is "Device intentionally generates and emits RF, including Bluetooth or
  Wi-Fi". `A-ELEC-003` (RF technologies), `A-ELEC-007` (module status) and `A-ELEC-021`
  (Covered List) describe how the rule applies, not whether it does, so they are not consulted by
  the spec — but under strict completeness they must still be supplied before a verdict.

### Deferred: `R-ELEC-012` Lithium battery UN 38.3 / US transport controls

`R-ELEC-012` is **removed from the approved Phase 2A TriggerSpec set**. It must remain:

```
TRIGGER_LOGIC_NOT_MODELED → REVIEW_REQUIRED
```

Reason, recorded as an approved decision: the rule's documented trigger requires that a
lithium-ion or lithium-metal cell/battery is *"offered for transportation"*. A predicate over
`A-ELEC-011` `battery_chemistry` alone would evaluate only part of that natural-language trigger,
and the current structured facts do not explicitly establish the transportation condition.
Emitting `APPLICABLE` / `NOT_APPLICABLE` from chemistry alone would therefore encode a partial
trigger, so no transport assumption is invented. Modelling it requires a separately approved
predicate over the transportation facts.

`R-ELEC-012` must **not** be replaced by another rule without separate approval.

Also deferred, unchanged from the earlier plan: `R-ELEC-017`, `R-TOY-012`, `R-TOY-014`.

**Runtime fail-safe for invalid specs.** Validation is not merely a test concern. During
`ApplicabilityEngine.__init__` the spec table is validated once (including branch-partition
validation) and the passing set is stored in `self._valid_spec_rule_ids`. Specs that fail
validation — including overlapping, conflicting, or non-covering specs — are **disabled**. If a
rule's spec is disabled, G6 returns `REVIEW_REQUIRED` / `TRIGGER_SPEC_INVALID`. An invalid spec must
never crash the Agent and never produce `APPLICABLE` or `NOT_APPLICABLE`. This is a few lines of
lookup, not a validation/config framework.

## F. Exact Phase 2A tests

`tests/test_applicability.py`. All tests are offline and deterministic: no model, no network, no
API calls. Canonical typed facts are constructed directly (the `--facts-file` boundary arrives in
Phase 2B), so `bool`/`int`/`float`/`str`/`date` values are passed as real Python objects.

Two repositories are used, and the distinction matters:

- **`JsonComplianceRepository`** (the real approved dataset) for dataset-level assertions and for
  the three approved specs.
- **`StubRepository`** — a small in-memory implementation of the existing
  `ComplianceRepository` `typing.Protocol`, defined in the test file and populated with synthetic
  `ComplianceRule` / `ProductAttribute` / `KnownGap` objects. It is used for every state the
  approved data does not contain: `UNVERIFIED`, `CONFLICT`, `NOT_FOUND`, `PROPOSED`, `WATCHLIST`,
  `SUPERSEDED`, `UNKNOWN`, unknown rule ids, and malformed specs.

Synthetic states are **never** produced by mutating a copy of a real rule and relying on
production code to trust it. The engine accepts only `rule_id` strings and resolves canonical
rules from the repository, so `StubRepository.get_rule` is the single controlled injection point.

Comparisons use **structural model equality** (`result == expected`, or `model_dump(mode="json")`
dicts) — never brittle byte-identical JSON strings.

### F.1 Spec integrity and partition validation (A)

- `T-S1` — `validate_trigger_specs(real_repo) == []`.
- `T-S2` — `set(TRIGGER_SPECS) == {"R-TOY-010","R-TOY-011","R-ELEC-002"}` exactly — three specs,
  no unapproved specs, and `R-ELEC-012` is absent.
- `T-S3` — every `spec.rule_id` resolves in the repository.
- `T-S4` — every condition attribute is in that rule's `required_attribute_ids`.
- `T-S5` — `deciding_attribute_ids` has exactly one entry and equals the branch condition attribute.
- `T-S6` — every operator is `EQ` or `IN`; `TriggerCombine` has exactly one member, `ALL`, and
  every branch uses it. `TriggerCombine.ANY` does not exist.
- `T-S7` — every branch has exactly one condition; every `EQ` expects a scalar; every `IN` expects
  a non-empty list whose members are in the deciding attribute's canonical domain.
- `T-S8` — every deciding attribute has a non-`None` `canonical_domain(...)`.
- `T-S9` — boolean partition uses the canonical domain: for TS-1/TS-2/TS-3 the union of matched
  values equals `{True, False}`, and no comparison is made against textual `"yes"`/`"no"`.
- `T-S10` — `_APPROVED_SPEC_RULE_IDS` equals the `TRIGGER_SPECS` key set.
- `T-S11` — **coverage rejection:** a synthetic spec leaving one canonical value uncovered yields
  non-empty problems and is excluded from `valid_spec_rule_ids`.
- `T-S12` — **overlap rejection:** a synthetic spec with two branches matching the same canonical
  value yields non-empty problems and is excluded.
- `T-S13` — **conflicting-overlap rejection:** a synthetic spec where two overlapping branches
  carry different outcomes yields non-empty problems and is excluded.
- `T-S14` — **duplicate-outcome overlap rejection:** two overlapping branches with the *same*
  outcome are still rejected (overlap is invalid regardless of outcome agreement).
- `T-S15` — a synthetic spec whose deciding attribute is an open type (`A-ELEC-005`, number)
  yields non-empty problems and is excluded.
- `T-S16` — a synthetic spec whose condition attribute is not in `required_attribute_ids` yields
  non-empty problems and is excluded.
- `T-S17` — a synthetic spec with an unapproved `rule_id` yields non-empty problems.
- `T-S18` — `_branch_matched_values` returns the correct canonical value set for `EQ` (singleton)
  and `IN` (declared list).

### F.2 G0 canonical resolution and rule-existence precedence (B)

- `T-G0-1` — `evaluate(["R-FAKE-999"], facts)` gives `REVIEW_REQUIRED` / `[UNKNOWN_RULE_ID]`.
- `T-G0-2` — with an unknown id **and** a real modelled rule in the same call, the unknown entry is
  `[UNKNOWN_RULE_ID]` while the real entry is evaluated normally, proving no contamination.
- `T-G0-3` — no exception is raised for an unknown or malformed rule id (including `""`).
- `T-G0-4` — an unknown rule is never `APPLICABLE` / `NOT_APPLICABLE`.
- `T-G0-5` — `evidence_status is None` and `rule_status is None` only for the unknown-rule result;
  every resolved rule has both populated.
- `T-G0-6` — **canonical-rule enforcement, structural:** `evaluate_rule` and `evaluate` accept only
  `rule_id` strings; neither signature accepts a `ComplianceRule`, so caller-supplied rule content
  cannot reach any gate.
- `T-G0-7` — **canonical-rule enforcement, behavioural:** a caller that has a locally altered
  `ComplianceRule` object cannot influence the outcome, because no such object is accepted; the
  verdict follows `StubRepository.get_rule`'s canonical content. Verified by returning a
  `PROPOSED` canonical rule from `StubRepository` and observing `[LIFECYCLE_PROPOSED]` even though
  a "fixed" copy exists in the test.

### F.3 Evidence gate (C)

All `UNVERIFIED` / `CONFLICT` / `NOT_FOUND` fixtures come from `StubRepository`.

- `T-E1` — `UNVERIFIED` evidence gives `REVIEW_REQUIRED` / `[EVIDENCE_NOT_VERIFIED]`.
- `T-E2` — `CONFLICT` evidence gives `REVIEW_REQUIRED` / `[EVIDENCE_NOT_VERIFIED]`.
- `T-E3` — `NOT_FOUND` evidence gives `REVIEW_REQUIRED` / `[EVIDENCE_NOT_FOUND]`, and is **never**
  interpreted as `NOT_APPLICABLE`.
- `T-E4` — each of the three, even with a satisfied trigger and complete valid facts, is never
  `APPLICABLE`.
- `T-E5` — dataset invariant: the approved dataset currently has zero non-`VERIFIED` rules.

### F.4 Lifecycle gate (D)

All four non-`EFFECTIVE` fixtures come from `StubRepository`.

- `T-L1` — `PROPOSED` gives `REVIEW_REQUIRED` / `[LIFECYCLE_PROPOSED]`.
- `T-L2` — `WATCHLIST` gives `REVIEW_REQUIRED` / `[LIFECYCLE_WATCHLIST]`.
- `T-L3` — `SUPERSEDED` gives `REVIEW_REQUIRED` / `[LIFECYCLE_SUPERSEDED]`.
- `T-L4` — `UNKNOWN` gives `REVIEW_REQUIRED` / `[LIFECYCLE_UNKNOWN]`.
- `T-L5` — `EFFECTIVE` continues past G2 (reaches later gates).
- `T-L6` — live check on the approved data: `R-ELEC-018` → `LIFECYCLE_WATCHLIST`;
  `R-ELEC-019` → `LIFECYCLE_PROPOSED`.

### F.5 P1 review-only policy (E)

- `T-P1` — dataset invariant: `CONV-GAP-001.related_rule_ids` equals the
  `mvp_priority == "P1"` set in the approved data, so the union's two sources agree.
- `T-P2` — all 8 review-only rules are `REVIEW_REQUIRED`, with
  `reason_codes[0] in {P1_REVIEW_ONLY, LIFECYCLE_PROPOSED, LIFECYCLE_WATCHLIST}`;
  never `APPLICABLE` / `NOT_APPLICABLE`.
- `T-P3` — lifecycle reason outranks generic P1: `R-ELEC-018` and `R-ELEC-019` report lifecycle
  codes, not `P1_REVIEW_ONLY`.
- `T-P4` — the other six P1 rules report `P1_REVIEW_ONLY`.
- `T-P5` — all 8 stay `REVIEW_REQUIRED` even when every required attribute is supplied as a valid
  `USER` fact.
- `T-P6` — review-only detection is policy-based: a synthetic `StubRepository` rule that is
  `EFFECTIVE`, `VERIFIED`, `mvp_priority == "P1"` and has complete metadata is still
  `REVIEW_REQUIRED` (it does not depend on the three `None` fields).
- `T-P7` — the `CONV-GAP-001` source alone is sufficient: a `StubRepository` without
  `CONV-GAP-001` still applies the P1 policy, and a `StubRepository` where only `CONV-GAP-001`
  lists a non-P1 rule still forces `P1_REVIEW_ONLY` for it.

### F.6 Strict required-fact completeness (F, D-1)

- `T-C1` — parameterized over all three specs: with every required attribute supplied as valid
  `USER` facts, the deciding value `True` yields `APPLICABLE`.
- `T-C2` — parameterized over all three specs: deciding value `False` yields `NOT_APPLICABLE`.
- `T-C3` — parameterized over all three specs **and each required attribute in turn**: removing
  that one attribute (while the deciding attribute stays present and satisfied) yields **no**
  `APPLICABLE` and **no** `NOT_APPLICABLE`; the result matches the rule's approved
  `runtime_status_if_missing`.
- `T-C4` — the same sweep with the removed attribute supplied as `CLASSIFIER` instead of `USER`.
- `T-C5` — the same sweep with the removed attribute supplied with an invalid value.
- `T-C6` — whenever a definitive verdict is emitted, `missing_attribute_ids == []` (invariant).
- `T-C7` — the strict rule applies to all seven of `R-ELEC-002`'s required attributes and to all
  five of `R-TOY-010` / `R-TOY-011`'s required attributes.

### F.7 Missing never means negative (G)

- `T-I1` — sweep all 41 approved rules with zero facts: no result is `APPLICABLE` and no result is
  `NOT_APPLICABLE`.
- `T-I2` — sweep all 41 rules with zero facts: `reason_codes[0]` is always a non-verdict code.
- `T-I3` — for each of the three specs, supplying only the deciding attribute never yields
  `APPLICABLE` / `NOT_APPLICABLE`.

### F.8 Canonical `data_type` validation (H)

- `T-V1` — `boolean`: accepts `True` / `False`; rejects `"yes"`, `"no"`, `"unknown"`, `1`, `0`, `None`.
- `T-V2` — `enum`: accepts a declared value; rejects an undeclared string and a non-string.
- `T-V3` — `multi_select`: accepts a list of declared members; rejects a mixed valid/invalid list
  and a bare string.
- `T-V4` — `integer`: accepts `3`; rejects `3.5`, `"3"`, `True`.
- `T-V5` — `number`: accepts `1.2` and `3`; rejects `True`, `"1.2"`.
- `T-V6` — `decimal`: accepts `1.2`; rejects `True`, `"1.2"`.
- `T-V7` — `text`: accepts `"China"`; rejects `123`.
- `T-V8` — `date`: accepts a real `datetime.date`; rejects `"2026-01-01"` and a `datetime.datetime`.
- `T-V9` — `structured_text`: accepts a string; rejects a list.
- `T-V10` — `structured_list`: accepts a list; rejects a string. Inner contents are not validated.
- `T-V11` — unknown `data_type` → invalid, fail safe, no exception.
- `T-V12` — open types are not vocabulary-checked: `A-ELEC-005` (number) accepts `1.2`,
  `A-CMN-002` (text) accepts `"China"`, `A-CMN-009` (integer) accepts `7`.
- `T-V13` — `_VOCABULARY_FREE_ATTRIBUTE_IDS == frozenset({"A-CMN-003","A-TOY-024"})` exactly;
  `A-CMN-003=["California"]` is accepted while `A-CMN-004=["Amazon","Nope"]` is rejected.
- `T-V14` — `canonical_domain` returns `{True, False}` for `A-TOY-012` (and other booleans),
  `frozenset(allowed_values)` for `A-ELEC-011`, and `None` for `A-ELEC-005`.

### F.9 Mixed-origin fact behavior (I)

Using `A-TOY-015` (deciding attribute of `R-TOY-011`) with the other required attributes supplied
as valid `USER` facts. Issue behaviour is **deterministic**, never optional.

1. `T-M1` — `USER=True` only → usable canonical fact; satisfied path reachable.
2. `T-M2` — `CLASSIFIER=True` only → canonical fact missing; no `usable` entry;
   `reason_codes == [MISSING_REQUIRED_FACTS, UNTRUSTED_FACT_ORIGIN]`; status follows
   `MISSING_REQUIRED_FACTS` via `runtime_status_if_missing` (`R-TOY-011` → `NEEDS_INFO`);
   exactly one `UNTRUSTED_ORIGIN` input issue for `A-TOY-015`.
3. `T-M3` — `DERIVED=True` only → identical fail-safe behavior and identical issue.
4. `T-M4` — `USER=True` + `CLASSIFIER=False` → `USER=True` remains canonical; **no** contradiction;
   `APPLICABLE`; **exactly one** `UNTRUSTED_ORIGIN` input issue for `A-TOY-015` is recorded
   (the ignored untrusted submission), and the dependent rule's status is unchanged by it.
5. `T-M5` — `USER=True` + `DERIVED=False` → same as `T-M4`.
6. `T-M6` — `USER=True` + `USER=True` → collapses to one canonical fact; no issue.
7. `T-M7` — `USER=True` + `USER=False` → `CONTRADICTORY_FACTS` → `REVIEW_REQUIRED`;
   `reason_codes[0] == CONTRADICTORY_FACTS`; one `CONTRADICTORY_VALUES` issue; the attribute is
   not `usable`.
8. `T-M8` — `USER=True` + `CLASSIFIER=True` (same value) → canonical `USER=True`; **no** issue,
   because no untrusted information was ignored.
9. `T-M9` — untrusted facts never appear in `FactIndex.usable`.
10. `T-M10` — `UNTRUSTED_FACT_ORIGIN` is never at `reason_codes[0]` in any fixture.
11. `T-M11` — a conflicting untrusted fact alone never creates a canonical contradiction:
    `contradictory` does not contain the attribute in `T-M4` / `T-M5`.
12. `T-M12` — issue de-duplication: multiple untrusted facts for one attribute still produce exactly
    one `UNTRUSTED_ORIGIN` issue for that `(attribute_id, issue_code)` pair.

### F.10 Invalid / unknown fact isolation (J)

- `T-X1` — unknown `A-FAKE-999` produces a global `UNKNOWN_ATTRIBUTE_ID` input issue and does **not**
  make all rules `REVIEW_REQUIRED`; a full 41-rule before/after comparison shows no status change.
- `T-X2` — `R-TOY-011` still yields `NOT_APPLICABLE` from its own complete valid facts while an
  unrelated unknown attribute is present.
- `T-X3` — an unrelated invalid fact changes no rule's status.
- `T-X4` — an unrelated contradictory fact changes no rule's status.
- `T-X5` — a bad fact that a rule does depend on changes that rule only, and only that rule.
- `T-X6` — dependent-rule effects are exactly the rules whose canonical `required_attribute_ids`
  contain the affected attribute.

### F.11 Input-issue privacy (K)

- `T-K1` — `FactInputIssue.detail` never contains the submitted raw value's string form
  (checked across invalid, contradictory and unknown-attribute fixtures).
- `T-K2` — `detail` for an invalid boolean is exactly the type/vocabulary description,
  e.g. it contains `boolean` and `str` and nothing from the input.
- `T-K3` — no `RuleApplicabilityResult` or `ApplicabilityResult` field contains a user-supplied
  value (structural assertion that the contract stays value-free).

### F.12 Trigger outcomes (L)

- `T-T1` — `R-TOY-010`: complete valid facts, `A-TOY-012=True` → `APPLICABLE` /
  `[TRIGGER_SATISFIED]`; `=False` → `NOT_APPLICABLE` / `[TRIGGER_NOT_SATISFIED]`.
- `T-T2` — `R-TOY-011`: same shape via `A-TOY-015`.
- `T-T3` — `R-ELEC-002`: `A-ELEC-002=True` → `APPLICABLE`; `=False` → `NOT_APPLICABLE`;
  missing → missing-fact handling (`REVIEW_REQUIRED` per `runtime_status_if_missing`);
  `"unknown"` supplied as a value is **rejected by validation as invalid**, never treated as a
  canonical boolean and never producing `APPLICABLE` / `NOT_APPLICABLE`.
- `T-T4` — `evaluated_attribute_ids == spec.deciding_attribute_ids` whenever a verdict is emitted.
- `T-T5` — a verdict is reachable only through the unique matching branch of a validated spec.
- `T-T6` — `_resolve_outcome` returns the branch outcome for a canonical value, and `None` when
  zero or multiple branches match (multi-match exercised by bypassing validation with a synthetic
  spec object, then asserting G6 returns `TRIGGER_SPEC_INVALID`).

### F.13 Unmodeled and deferred rules (M)

- `T-U1` — `R-CMN-001` with all required attributes supplied as valid `USER` facts →
  `REVIEW_REQUIRED` / `[TRIGGER_LOGIC_NOT_MODELED]`.
- `T-U2` — same for `R-TOY-004` and `R-ELEC-003` (unmodeled P0 rules).
- `T-U3` — **`R-ELEC-012` is deferred**: with all six of its required attributes supplied as valid
  `USER` facts, including `A-ELEC-011 = "lithium ion"`, the result is `REVIEW_REQUIRED` /
  `[TRIGGER_LOGIC_NOT_MODELED]` — never `APPLICABLE` and never `NOT_APPLICABLE`.
- `T-U4` — `R-ELEC-017`, `R-TOY-012`, `R-TOY-014` are absent from `TRIGGER_SPECS` and return
  `[TRIGGER_LOGIC_NOT_MODELED]`.
- `T-U5` — unmodeled/deferred rules are never `APPLICABLE` / `NOT_APPLICABLE`.

### F.14 Invalid TriggerSpec runtime fail-safe (N)

- `T-N1` — with a deliberately broken spec injected into the table (uncovered value, overlapping
  branches, or conflicting-overlap), `valid_spec_rule_ids` excludes it and the engine returns
  `REVIEW_REQUIRED` / `[TRIGGER_SPEC_INVALID]` for that rule.
- `T-N2` — the same broken spec never raises an exception.
- `T-N3` — the same broken spec never produces `APPLICABLE` / `NOT_APPLICABLE`, even with complete
  valid facts whose deciding value would otherwise satisfy a branch.
- `T-N4` — engine construction succeeds with an invalid spec present (specs are disabled, not fatal).
- `T-N5` — a non-partition spec is not resolved by branch order: reversing the branch order of an
  overlapping spec does not change the outcome (still `TRIGGER_SPEC_INVALID`).

### F.15 Gate precedence (O)

- `T-O1` — unknown rule id + complete valid facts for other rules present →
  `[UNKNOWN_RULE_ID]` for the unknown entry only.
- `T-O2` — `UNVERIFIED` canonical evidence + satisfied trigger + complete facts →
  `[EVIDENCE_NOT_VERIFIED]`.
- `T-O3` — canonical `PROPOSED` + P1 + complete facts → `[LIFECYCLE_PROPOSED]`.
- `T-O4` — canonical `WATCHLIST` + P1 + complete facts → `[LIFECYCLE_WATCHLIST]`.
- `T-O5` — contradictory fact + trigger true → `[CONTRADICTORY_FACTS]`.
- `T-O6` — invalid + missing → `[INVALID_FACT_VALUE, MISSING_REQUIRED_FACTS]`,
  `reason_codes[0] == INVALID_FACT_VALUE`.
- `T-O7` — untrusted-only deciding attribute →
  `[MISSING_REQUIRED_FACTS, UNTRUSTED_FACT_ORIGIN]`,
  `reason_codes[0] == MISSING_REQUIRED_FACTS`.
- `T-O8` — canonical rule with empty `required_attribute_ids` and complete metadata →
  `[NO_REQUIRED_ATTRIBUTES]`.
- `T-O9` — a table-driven check that `reason_codes[0]`'s implied status equals the result status.
- `T-O10` — a valid `USER` fact plus an ignored untrusted conflict still reaches the verdict
  (`NOT_APPLICABLE` or `APPLICABLE`) and records exactly one `UNTRUSTED_ORIGIN` issue, proving the
  issue does not change the dependent rule's status.

### F.16 Determinism (P)

- `T-D1` — the same canonical facts supplied in a different input order produce an
  `ApplicabilityResult` that is `==` (structural model equality).
- `T-D2` — `rules` ordering is stable and independent of fact input order.
- `T-D3` — `reason_codes` ordering is stable across repeated runs and permutations.
- `T-D4` — `missing_attribute_ids` ordering is stable (canonical `required_attribute_ids` order,
  de-duplicated; the aggregate union is deterministic).
- `T-D5` — `input_issues` ordering is stable (sorted by `attribute_id`, then `issue_code`) and
  contains one entry per `(attribute_id, issue_code)` pair.
- `T-D6` — `counts` contains all four status keys and sums to `len(rules)`.
- `T-D7` — no test relies on byte-identical JSON strings.

### F.17 No-runtime-change regression (Q)

- `T-R1` — the existing orchestrator path still yields
  `unknown.not_evaluated["applicability"] == "NOT_EVALUATED"`.
- `T-R2` — `AnalysisResult` still has **no** `applicability` attribute (guards against accidental
  Phase 2B leakage into Phase 2A).
- `T-R3` — `unknown.not_evaluated["risk"] == "NOT_EVALUATED"` and
  `unknown.not_evaluated["cost"] == "NOT_AVAILABLE"`, unchanged.
- `T-R4` — Agent tools are still exactly 2, and `get_compliance_evidence` remains scoped to the
  current analysis.
- `T-R5` — `analyze_product` exposes only a `product_description` parameter: no fact/attribute/value
  parameter exists through which an Agent could inject canonical facts.
- `T-R6` — existing classification and human-review behavior unchanged (`agent_generated` still
  `REVIEW_REQUIRED`; an agent suggestion still never becomes `RESOLVED`).
- `T-R7` — the full existing suite (105 tests) plus all Phase 2A tests pass, 0 failures.
- `T-R8` — no test makes a real model/API call (offline only).

Estimated total: about 80–92 test functions, several parameterized.

## G. Estimated change size

| Component | Est. lines |
|---|---|
| `src/services/applicability.py` — enums + fixed mappings | ~65 |
| — result-contract models | ~55 |
| — trigger-spec models | ~30 |
| — `FactIndex` + `index_facts` (incl. deterministic mixed-origin rules) + 10 `data_type` validators + `canonical_domain` | ~150 |
| — `TRIGGER_SPECS` (3) + `validate_trigger_specs` / `_spec_problems` / `_branch_matched_values` / `valid_spec_rule_ids` (incl. partition checks) | ~100 |
| — minimal evaluator (`_evaluate_condition` / `_evaluate_branch` / `_resolve_outcome`) | ~30 |
| — engine (`GateOutcome`, G0 canonical resolution, G1–G6, `evaluate_rule`, `evaluate`) | ~140 |
| — explanations + helpers | ~40 |
| **`src/services/applicability.py` subtotal** | **~430–525** |
| `tests/test_applicability.py` (incl. `StubRepository` fixture) | **~470–590** |
| Existing files modified | **0** |
| New dependencies | **0** |
| New Agent tools | **0** |
| Approved JSON modified | **0** |

One new module rather than several: the fact-validation layer, the spec table and the engine are
cohesive, and none of them is reusable elsewhere yet. The size is driven mainly by the ten
`data_type` validators, the deterministic mixed-origin rules, the six gates, and the spec
partition validation. If the module exceeds roughly 560 lines during implementation, that will be
reported rather than silently split or padded.

## H. Remaining risks and confirmed decisions

### Confirmed decisions

- **D-1 — STRICT required-attribute completeness is APPROVED for Phase 2A.**
  Before an approved `TriggerSpec` may produce `APPLICABLE` or `NOT_APPLICABLE`, **all** canonical
  `rule.required_attribute_ids` must be present, `FactOrigin.USER`, valid and non-contradictory.
  If any is unavailable, untrusted, invalid or contradictory, no definitive verdict is emitted;
  the approved `missing_information_blocks_decision` / `runtime_status_if_missing` metadata is used
  to produce `NEEDS_INFO` or `REVIEW_REQUIRED`. A verdict is **never** permitted on the basis of
  `deciding_attribute_ids` alone while other required attributes are missing. Treating some
  required attributes as "only how, not whether" would itself be a new regulatory interpretation.
  Distinguishing deciding vs supporting attributes is explicitly out of scope for Phase 2A.
- **D-2 — invalid canonical fact value → `REVIEW_REQUIRED` / `INVALID_FACT_VALUE` is APPROVED.**
  `NEEDS_INFO` is not used for invalid typed input: this is an input defect needing correction,
  not a pure knowledge gap.
- **D-3 — Phase 2B is ONE atomic integration change**, containing: the human fact boundary,
  the `ProductFact(origin=USER)` flow, `Orchestrator` / `AnalysisService` integration, the new
  top-level `AnalysisResult.applicability` block, **removal of
  `unknown.not_evaluated["applicability"]`**, preservation of the risk/cost placeholders, the
  Agent/system prompt wording update, and the existing `analyze_product` tool consuming canonical
  applicability. Phase 2C is real DeepSeek applicability regression and real-model wording
  validation, with no major architecture work unless a real defect is found.
- **`R-ELEC-012` is deferred from Phase 2A and must remain `TRIGGER_LOGIC_NOT_MODELED` →
  `REVIEW_REQUIRED`.** Its documented trigger includes the condition that a lithium-ion or
  lithium-metal battery is *"offered for transportation"*, but a `battery_chemistry`-only predicate
  would evaluate only part of the natural-language trigger; the current structured facts do not
  explicitly prove the transportation condition. No transport assumption is invented. A separately
  approved predicate over the transportation facts is required before it can be modelled.
  The approved Phase 2A spec set is exactly `R-TOY-010`, `R-TOY-011`, `R-ELEC-002`, and
  `R-ELEC-012` must not be replaced by another rule without separate approval.
- **G0 resolves the canonical repository rule; no caller-supplied rule content is trusted.**
  `rule_id → repository lookup → canonical_rule → G1–G6 use canonical_rule only`. The engine's
  entry points accept only `rule_id` strings. Matching a caller-supplied object on `rule_id` is not
  sufficient and is not the design. Synthetic `UNVERIFIED` / `PROPOSED` / `SUPERSEDED` states are
  exercised through a controlled test repository implementing the existing `ComplianceRepository`
  Protocol, never by mutating a real rule copy.
- **TriggerSpec branches must be a partition.** Validation verifies both coverage and that each
  canonical deciding value maps to **exactly one** branch outcome. Uncovered values, overlapping
  branches, and conflicting overlapping outcomes are rejected. The evaluator does **not** use
  "first matching branch wins": it requires exactly one match and otherwise fails closed with
  `REVIEW_REQUIRED` / `TRIGGER_SPEC_INVALID`. Malformed overlapping specs are disabled at runtime
  and can never produce a verdict.
- **No `ANY` combinator in Phase 2A.** `TriggerCombine` has exactly one member, `ALL`, which is the
  only combine behaviour the approved specs require. No operators or combinators are added for
  speculative future extensibility; a future approved rule may introduce `ANY` under separate
  approval.
- **Mixed-origin behaviour is deterministic, not optional.** A valid `USER` fact remains canonical;
  a conflicting `CLASSIFIER`/`DERIVED` fact never creates a canonical contradiction; the ignored
  untrusted submission is **always** recorded as exactly one global `UNTRUSTED_ORIGIN` input issue
  for that attribute; and it does not change the dependent rule's status when a valid `USER` fact
  exists. Untrusted facts that merely repeat the canonical value produce no issue.
- **Only `FactOrigin.USER` is trusted.** `CLASSIFIER` and `DERIVED` may be recorded but must not
  create `APPLICABLE`, must not create `NOT_APPLICABLE`, must not override a `USER` fact, and must
  not create a contradiction against a valid `USER` fact.
- **Canonical boolean has no "unknown" value.** The canonical domain is exactly `{True, False}`.
  The approved textual `allowed_values` (`"yes"`, `"no"`, `"unknown"`) are a data-representation
  detail and must not weaken the canonical type or be used for coverage comparison. A human
  choosing "unknown" is represented by the boundary as absence of a `ProductFact`.
- **`UNKNOWN_RULE_ID` is G0** — checked before evidence, lifecycle, facts and trigger logic.
  Unknown rule → `REVIEW_REQUIRED`, never `APPLICABLE`/`NOT_APPLICABLE`, no exception.
- **`UNTRUSTED_FACT_ORIGIN` is supplemental provenance, never a deciding reason.** Fact-level
  deciding precedence is `CONTRADICTORY_FACTS` > `INVALID_FACT_VALUE` > `MISSING_REQUIRED_FACTS`.
- **Invalid TriggerSpecs fail safe at runtime**: validated once at engine construction (including
  partition validation), disabled if invalid, `REVIEW_REQUIRED` / `TRIGGER_SPEC_INVALID`, never a
  crash, never a verdict.
- **No approved JSON edits.** `rule.required_attribute_ids` is the authoritative rule→attribute
  relation. `product_taxonomy.triggered_rule_ids` is not the canonical trigger source.
- **The 7 reverse-only `triggered_rule_ids` mismatches are deferred** as a future data-quality
  issue: `A-ELEC-001→R-TOY-001`, `A-ELEC-006→R-ELEC-010`, `A-ELEC-006→R-ELEC-011`,
  `A-ELEC-011→R-ELEC-017`, `A-TOY-007→R-TOY-005`, `A-TOY-011→R-TOY-003`,
  `A-TOY-017→R-TOY-012`.
- **No Agent tool expansion.** Exactly 2 tools remain; no `evaluate_applicability` /
  `evaluate_compliance` tool. Applicability runs internally through
  `analyze_product → AnalysisService → ApplicabilityEngine`.
- **Phase 2A changes no existing runtime behavior.** `AnalysisResult` is untouched; the runtime
  still reports `applicability = NOT_EVALUATED`, `risk = NOT_EVALUATED`, `cost = NOT_AVAILABLE`.
- **`ComplianceFinding` is not extended.** The engine operates on original repository
  `ComplianceRule` objects; internal rule metadata is not pushed into Agent-visible findings.
- **Option C is primary, Option A is the escape hatch.** An explicit source-controlled Python
  `TriggerSpec` table plus a small generic deterministic evaluator; a rule-specific Python predicate
  remains available for rules too complex for the simple form. No `data/rule_triggers.json` and no
  contamination of approved research JSON with newly authored runtime logic.
- **No result-model changes to `src/models.py` or `src/state.py`.** Existing `ProductFact`,
  `FactOrigin`, `ComplianceRule` and the repository are reused; no second incompatible fact model.

### Remaining risks

1. **The three specs are authored regulatory interpretation.** Even minimal predicates encode
   "subject-matter presence" reasoning. Mitigation: in-code `rationale`, single
   `deciding_attribute_ids` traceability, partition assertions, and a test pinning the exact
   approved spec set. Residual risk is real and needs human review; it cannot be removed by
   engineering. The `R-ELEC-012` deferral is the concrete evidence that this risk is being taken
   seriously rather than papered over.
2. **Deciding-attribute choice is still a judgment call for the remaining three specs.**
   `R-ELEC-002` keys on `intentional_rf_transmitter` and ignores the Covered List attribute.
   Documented per spec; changing one is a one-line spec edit.
3. **Strict completeness makes verdicts deliberately hard to reach.** Under D-1, `R-ELEC-002`
   requires seven supplied facts and `R-TOY-010` / `R-TOY-011` require five before any can produce
   a verdict. Only three of 41 rules are modelled at all, so most real runs will correctly report
   `NEEDS_INFO`/`REVIEW_REQUIRED`. This is intended behaviour, not a defect, but it will make the
   engine look "inert" in early manual testing.
4. **`allowed_values` is not a genuine vocabulary for two `multi_select` attributes.** `A-CMN-003`
   holds one prose string and `A-TOY-024` holds `[]`. Naive member validation would reject valid
   facts. Mitigated by the pinned `_VOCABULARY_FREE_ATTRIBUTE_IDS` frozenset and test `T-V13`.
   Inner structure of `structured_text` / `structured_list` is intentionally not validated, because
   validating it would invent semantics.
5. **P1 review-only policy is a union of two approved sources.** `CONV-GAP-001.related_rule_ids`
   (via `repository.get_known_gaps()`) union canonical `mvp_priority == "P1"`. The union is
   conservative: a future P1 rule outside `CONV-GAP-001` is still forced to `REVIEW_REQUIRED`, and
   test `T-P1` fires to surface the divergence. A single-source policy is available if preferred.
6. **The engine now depends on `get_rule` returning canonical content for the rule id.** That is
   the intended trust boundary, but it means a repository whose `get_rule` is inconsistent with its
   own `rules` collection would produce verdicts from the `get_rule` view. `JsonComplianceRepository`
   builds both from the same validated index, so this is not reachable with the approved data; it is
   recorded because the controlled test repository reproduces exactly this seam.
7. **Date typing creates a Phase 2B dependency.** `_validate_date` requires a real `datetime.date`;
   the `--facts-file` boundary must parse ISO-8601 strings and must reject bare strings. Only
   `A-CMN-015` is affected and it is not a deciding attribute of any Phase 2A spec, so Phase 2A is
   unaffected.
8. **Boolean-domain quirk is a recorded data-quality item.** Approved boolean attributes carry
   textual `allowed_values` (`"yes"`, `"no"`, sometimes `"unknown"`) while the canonical engine
   domain is `{True, False}`. Coverage validation must therefore be data-type aware. No canonical
   type is weakened to accommodate the representation.
9. **Partition validation restricts the spec form.** Phase 2A allows exactly one deciding attribute
   and one condition per branch. A future rule needing multi-attribute or OR-shaped logic requires
   an approved design change rather than a table entry. This is deliberate: it keeps partition
   soundness provable and prevents ambiguous authored logic from ever producing a verdict.
10. **`input_issues` may be noisy** if a user supplies many unknown attribute ids. Accepted: it is a
    global list, never broadcast into per-rule statuses, bounded by input size, and de-duplicated
    per `(attribute_id, issue_code)` pair.
11. **Agent context size.** Mitigated by the lean contract: no raw values, no per-rule explanation
    prose. Worst case is 41 lean result objects; `REASON_EXPLANATIONS` is applied at the presentation
    layer, not embedded per result.
12. **The 105-test baseline is the acceptance bar.** Phase 2A adds two new files only; `T-R7`
    requires the full existing suite to remain green with no runtime change.
13. **A defensive branch is unreachable by construction.** A required attribute id that does not
    resolve in the repository is treated as not usable (`MISSING_REQUIRED_FACTS` +
    `REVIEW_REQUIRED`), but `repository.validate_references()` already guarantees rule→attribute
    references resolve at load time, so this path cannot be exercised with the approved data.
