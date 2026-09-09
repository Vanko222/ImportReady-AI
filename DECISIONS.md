# ImportReady AI — Project Decisions

## D001 — MVP Scope

Build the smallest complete, demo-ready MVP first.

Supported:

- USA market
- Amazon / eBay
- Children's toys
- Small consumer electronics

Wooden blocks and Bluetooth earphones are demo examples, not fixed supported products.

If a product is outside the supported categories, stop safely instead of inventing compliance information.

---

## D002 — Core Agent Flow

ImportReady AI is an agent workflow, not a simple compliance chatbot.

Core flow:

Product Input
→ Classification
→ Supported Category Check
→ Compliance Lookup
→ User Clarification if needed
→ Cost Calculation
→ Import Readiness Report

---

## D003 — Grounded Compliance Data

Compliance rules, certifications, policy details, and cost figures must not rely on model memory alone.

Use approved project data or verified sources.

Unknown or estimated information must be clearly labeled.

---

## D004 — Human-in-the-Loop

Two human checkpoints are required:

1. Missing product information
   → Agent asks user
   → User confirms
   → Analysis continues

2. Policy update
   → Human triggers policy check
   → Agent prepares candidate update
   → Human Approves / Rejects
   → Only approved data enters the knowledge base

The Agent must not autonomously modify trusted compliance data.

---

## D005 — Policy Update MVP

For the MVP, one approved official policy source is sufficient.

Prove the complete review-and-update loop first.

Large-scale crawling, multi-source monitoring, and fully autonomous policy updates are out of scope.

---

## D006 — Development Responsibilities

- User → Product owner and final decision-maker
- ChatGPT main chat → Project controller
- ChatGPT Work → Policy research and bounded research/review
- DeepSeek Harness → Primary coding agent
- Git → Source/version control
- Project Markdown files → Persistent project context

Policy research and coding remain separated:
Work researches → main controller reviews → Harness implements approved data.

---

## D007 — Engineering Rules

Prefer simple, local, reversible implementations.

Avoid unnecessary infrastructure, abstractions, refactoring, or dependency changes.

Validation must match risk and use objective evidence.

Coding agents must not automatically commit, rewrite Git history, or perform destructive Git operations.