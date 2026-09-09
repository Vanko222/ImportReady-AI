# ImportReady AI — Agent Rules

## 1. Priority

Follow this order:

1. Safety and factual correctness
2. Current task acceptance criteria
3. Preserve working functionality
4. Minimal task-scoped changes
5. Speed and token/API efficiency

Do not expand scope, optimize, or refactor beyond the current task unless asked.

## 2. Grounding & Hallucination Control

Never present assumptions as facts.

Treat important information as:

- Confirmed
- Inferred
- Estimated
- Unknown / unverified

For compliance rules, certifications, legal claims, costs, fees, and other external facts:

- Never invent missing information.
- Never guess exact numbers.
- Use approved project data or verified sources.
- Clearly label estimates and uncertainty.
- If evidence is insufficient, say so.

Actual files, code, tests, logs, and tool output override previous assumptions.

## 3. Memory Hygiene

Do not store guesses, failed experiments, temporary observations, or unverified conclusions as project facts.

Persistent project files should contain only verified information.

If new evidence conflicts with existing documentation:

1. Identify the conflict.
2. Verify the correct information.
3. Update only after confirmation.

Use project files by role:

- AGENTS.md → stable agent rules
- PRODUCT.md → confirmed product definition
- STATUS.md → current progress and blockers
- DECISIONS.md → confirmed key decisions
- ARCHITECTURE.md → confirmed technical architecture

## 4. Task & Context Discipline

Before editing, determine:

- Objective
- Relevant files
- Acceptance criteria
- Required validation

Read only files needed for the task.
Expand search only when necessary.
Do not repeatedly read unchanged files or scan the full repository without reason.

Make the smallest working change.

Do not:

- Add unrelated features
- Modify unrelated files
- Perform unrelated refactors
- Rewrite working code unnecessarily
- Change architecture, dependencies, or public behavior without need
- Add speculative abstractions

Prefer simple, reliable MVP solutions.

## 5. Dependencies, Environment & Safety

Primary host environment: Windows + PowerShell.

Do not assume macOS/Linux host commands unless explicitly working inside a confirmed Linux environment.

Do not add or upgrade dependencies unless necessary.

Never expose or commit:

- API keys
- Tokens
- Passwords
- Credentials

Use environment variables or `.env` where appropriate.

Work only inside the current workspace unless explicitly authorized.

## 6. Validation, Debugging & Stop Rules

Validation must match risk:

- Low risk → targeted check
- Medium risk → relevant tests + affected workflow
- High risk → broader integration validation

A task is complete only with objective evidence such as:

- Passing test
- Successful build
- Successful command
- Verified runtime behavior
- Relevant logs/output

Do not repeatedly rerun successful checks without new changes.

When debugging:

1. Read the actual error.
2. Identify the likely cause.
3. Apply the smallest reasonable fix.
4. Validate again.

Do not randomly modify multiple areas.

If essentially the same strategy fails 2–3 times:

- Stop
- Summarize evidence
- Propose a different approach or request guidance

Stop once acceptance criteria pass.
Do not continue polishing or optimizing unless asked.

## 7. Git & Change Safety

Keep diffs small and reviewable.

Do not automatically:

- Commit
- Force push
- Rewrite Git history
- Delete branches
- Perform destructive resets
- Revert unrelated user changes

Do not perform destructive actions without explicit approval.

## 8. Human Decision Boundary

Ask before materially changing:

- Product scope
- Core architecture
- Technology stack
- Paid services or significant API cost
- Data model
- Security model
- Deployment strategy
- Confirmed product behavior

Routine implementation details do not require approval.

## 9. Token & Cost Efficiency

Prefer:

- Targeted file reads
- Small edits
- Stable project context
- Concise tool output
- Short final reports

Avoid:

- Repeating project background
- Dumping large unchanged files
- Repeated repository scans
- Long completion explanations
- Endless retry loops
- Unnecessary high-effort model use

## 10. Completion Report

Return only:

### Changed
- Files changed
- What changed

### Validation
- What was tested
- Result

### Remaining
- Blocker/risk, or "None"

When uncertainty affects correctness, scope, security, cost, or architecture:
verify first; if still uncertain, ask instead of guessing.