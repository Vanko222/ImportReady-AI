# ImportReady AI — Product Definition

## Product

**ImportReady AI**

AI Agent for cross-border import risk assessment.

Target user: a small US e-commerce seller sourcing products from China before placing a bulk order.

The user wants to know:

1. Whether the product meets US market requirements
2. Whether important certifications or documents are missing
3. How compliance gaps may affect actual cost

## MVP Scope

- Market: USA
- Platforms: Amazon / eBay
- Supported categories:
  - Children's toys
  - Small consumer electronics

Demo examples:

- Children's wooden blocks
- Bluetooth earphones

These are examples only. The MVP should accept other products within the two supported categories.

## Core Product Flow

User Input
→ Product Classification
→ Supported Category Check
→ Compliance Lookup
→ Clarification if needed
→ Cost Calculation
→ Import Readiness Report

Explanation:

The user enters a product such as a Bluetooth speaker or wooden toy.

The Agent first determines its category.

If the category is supported, the Agent retrieves relevant compliance information and continues the analysis.

If unsupported, the system stops and states that compliance information for that category is not currently available.

## Agent Behavior

The Agent should:

- Understand the purchasing goal
- Classify the product
- Determine whether the category is supported
- Retrieve compliance information from approved project data
- Ask the user when important information is missing
- Use tools for calculations
- Convert compliance risks into actionable cost and purchasing advice

The Agent must not invent compliance information.

## Compliance Knowledge Base

Compliance information must come from an approved knowledge base rather than model memory alone.

Policy records should support fields such as:

- Product category
- Requirement
- Source
- Source URL
- Effective date
- Last checked date
- Verification status

## Policy Update Workflow

Human clicks "Check Policy Updates"
→ Agent checks an approved official source
→ Agent creates candidate updates
→ Human reviews
→ Approve / Reject
→ Approved data enters the trusted knowledge base

Explanation:

The Agent may discover and organize possible policy changes, but it cannot directly modify trusted compliance data.

A human must approve the change first.

For the MVP, one official policy source is sufficient if time is limited.

## Human-in-the-Loop

Two forms of human participation are required:

1. **User clarification**
   - Agent asks for missing product information.
   - User confirms before analysis continues.

2. **Policy review**
   - Agent proposes policy updates.
   - Human approves or rejects them before the knowledge base changes.

## Final Output

The Import Readiness Report should include:

- Risk Level: Low / Medium / High
- Detected Risks
- Missing requirements or documents
- Estimated Cost Impact
- Recommended Actions
- Relevant sources where available

Estimates must be clearly labeled.

## Out of Scope

The MVP does not include:

- Global market coverage
- Automatic HS Code determination
- Automatic customs declaration
- Tariff avoidance
- Supplier credit investigation
- Fully autonomous policy updates
- Large-scale multi-source crawling

## MVP Principle

Prove one complete flow:

Classify the product
→ Check whether it is supported
→ Retrieve trusted compliance information
→ Ask for missing information
→ Calculate cost impact
→ Generate a useful report
→ Support human-reviewed policy updates