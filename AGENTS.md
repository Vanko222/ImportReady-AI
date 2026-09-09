# AGENTS.md

# ImportReady AI — Agent Rules

## 1. Mission & Priority

Build the smallest reliable, demo-ready MVP within the hackathon deadline.

Priority:

1. Safety and factual correctness
2. Current task acceptance criteria
3. Keep existing working features functional
4. Minimal focused changes
5. Speed and token/API cost efficiency

Do not expand scope or optimize beyond the current task unless explicitly requested.

---

## 2. MVP Scope

Current scope:

- Market: USA
- Platforms: Amazon / eBay
- Products:
  - Children's toys
  - Small electronics
- Core flow:
  - Understand product information
  - Identify compliance checks
  - Ask for missing information
  - Estimate compliance-related cost impact
  - Generate an Import Readiness Report

Out of scope unless explicitly requested:

- Global markets
- Automatic HS Code determination
- Automatic customs declaration
- Tariff avoidance
- Supplier credit investigation

---

## 3. Grounding & Hallucination Control

Never present assumptions as facts.

Important information must be treated as one of:

- Confirmed
- Inferred
- Estimated
- Unknown / unverified

For regulations, certifications, compliance requirements, costs, fees,
legal claims, or other external facts:

- Never invent missing information.
- Never guess exact numbers.
- Use approved project data or verified sources.
- Clearly label estimates.
- If evidence is insufficient, say so.

Actual files, code execution, tests, logs, and tool output take precedence
over previous assumptions.

---

## 4. Memory Hygiene

Do not store guesses, failed experiments, temporary observations,
or unverified conclusions as project facts.

Persistent project documentation should contain only verified information.

If new evidence conflicts with existing documentation:

1. Identify the conflict.
2. Verify the correct information.
3. Update documentation only after confirmation.

Use:

- AGENTS.md → stable agent rules
- PRODUCT.md → confirmed product definition
- ARCHITECTURE.md → confirmed technical architecture
- STATUS.md → current state, progress, blockers
- DECISIONS.md → confirmed important decisions

---

## 5. Task & Context Discipline

Before editing, determine:

- Objective
- Relevant files
- Acceptance criteria
- Required validation

Read only the files needed for the task.
Expand repository search only when necessary.
Do not repeatedly read unchanged files or scan the full repository without reason.

Prefer the smallest working change.

Do not:

- Add unrelated features
- Perform unrelated refactors
- Rewrite working code unnecessarily
- Modify unrelated files
- Change architecture or public behavior without approval
- Add speculative abstractions for future use

This is a hackathon MVP. Prefer simple and reliable solutions.

---

## 6. Dependencies & Environment

Do not add or upgrade dependencies unless necessary for the current task.

Avoid replacing working libraries or installing large toolchains without clear value.

Primary environment:

- Windows
- PowerShell
- VS Code
- Git
- Docker
- DeepSeek Harness

Do not assume macOS/Linux host commands unless explicitly working inside
a confirmed Linux environment.

---

## 7. Validation & Debugging

Validation must match change risk:

- Low risk: targeted check
- Medium risk: relevant tests + affected workflow
- High risk: broader integration validation

A task is complete only with objective evidence such as:

- Passing test
- Successful build
- Successful command
- Verified application behavior
- Relevant logs/output

Do not repeatedly rerun successful checks without new changes.

When debugging:

1. Read the actual error.
2. Identify the likely cause.
3. Apply the smallest reasonable fix.
4. Validate again.

Do not randomly modify multiple areas.

If essentially the same strategy fails 2–3 times, stop, summarize evidence,
and request guidance or propose a different approach.

---

## 8. Git, Safety & Secrets

Keep diffs small and task-focused.

Do not automatically:

- Commit
- Force push
- Rewrite Git history
- Delete branches
- Perform destructive resets
- Revert unrelated user changes

Never expose or commit:

- API keys
- Tokens
- Passwords
- Credentials

Use environment variables or `.env` where appropriate and keep secrets out of Git.

Work only inside the current workspace unless explicitly authorized.

---

## 9. Human Decision Boundary

Stop and ask before materially changing:

- Product scope
- Core architecture
- Technology stack
- Paid services or significant API cost
- Data model
- Security model
- Deployment strategy
- Confirmed product behavior

Routine implementation details do not require approval.

---

## 10. Efficiency, Completion & Reporting

Use only the context needed to complete the task correctly.

Avoid:

- Repeating project background
- Dumping large unchanged files
- Repeated repository scans
- Long completion explanations
- Endless optimization or retry loops

Stop when:

- Acceptance criteria are met
- Required validation passes
- No critical regression exists
- No blocker prevents the task from working

Final report:

### Changed
- Files changed
- What changed

### Validation
- What was tested
- Result

### Remaining
- Known blocker/risk, or "None"

Keep the report concise.

When uncertainty affects correctness, scope, security, cost, or architecture:
verify first; if still uncertain, ask instead of guessing.