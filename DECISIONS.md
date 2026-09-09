# ImportReady AI — Project Decisions

## D001 — MVP First

**Decision:** Build the smallest complete, demo-ready MVP before adding optional features.

**Reason:** The hackathon development window is limited. A complete end-to-end demo is more important than feature breadth.

---

## D002 — Limited Product Scope

**Decision:** The MVP supports:

- USA market
- Amazon / eBay
- Children's toys
- Small electronic products

Features outside the confirmed MVP scope will not be added unless explicitly approved.

---

## D003 — Agent-Oriented Product

**Decision:** ImportReady AI is an agent workflow, not a simple compliance chatbot.

The Agent should:

1. Understand the purchasing goal
2. Determine relevant checks
3. Ask for missing information
4. Use tools
5. Convert compliance risk into actionable cost and purchasing advice

---

## D004 — Grounded Compliance Data

**Decision:** Compliance requirements and cost figures must not be invented by the language model.

Use approved project data or verified sources.

Uncertain information must be labeled as estimated or unverified.

---

## D005 — Development Roles

**Decision:**

- User → Product owner and final decision-maker
- ChatGPT main chat → Project controller and technical/product decision support
- DeepSeek Harness → Primary coding agent
- ChatGPT Work → Bounded research, analysis, review, and document tasks
- Git → Source/version control
- Project Markdown files → Persistent project context

---

## D006 — Implementation Strategy

**Decision:** Prefer simple, local, reversible implementations.

Avoid:

- Premature abstraction
- Unnecessary refactoring
- Complex infrastructure
- Building custom agent frameworks/plugins during the hackathon

---

## D007 — Validation Strategy

**Decision:** Validation effort should match change risk.

Agent completion reports alone are not proof of success.
Important changes require objective validation such as tests, builds, commands, or verified runtime behavior.

---

## D008 — Git Discipline

**Decision:** Keep changes small and reviewable.

Coding agents should not automatically commit or perform destructive Git operations without explicit approval.