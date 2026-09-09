# ImportReady AI — Product Definition

## Product

**ImportReady AI**

AI Agent for Cross-border Import Risk Assessment.

## Target User

A small US e-commerce seller sourcing products from China before placing a bulk order.

The user wants to know:

1. Whether the product meets US market requirements
2. Whether required certifications or documents are missing
3. How missing requirements may affect the real landed cost

## MVP Market & Scope

- Target market: USA
- Sales platforms: Amazon / eBay
- Supported product categories:
  - Children's toys
  - Small electronic products

## Core User Flow

1. User enters product and purchasing information.
2. Agent identifies relevant compliance requirements.
3. If important information is missing, the Agent asks the user for clarification.
4. Agent uses tools to calculate purchasing and compliance-related cost impact.
5. Agent generates an Import Readiness Report.

## User Input

Possible input includes:

- Product
- Target market
- Sales platform
- Quantity
- Unit price
- Existing documents
- Certification status

## Agent Behavior

The Agent should:

- Understand the user's purchasing goal
- Determine which compliance checks are relevant
- Ask questions when information is insufficient
- Use tools instead of guessing calculations
- Convert compliance risk into understandable cost impact

The product is not simply a regulation Q&A system.

## Tools

MVP tools include:

- Compliance Knowledge Base
- Cost Calculator
- Product Classification

## Final Output

The Import Readiness Report should include:

- Risk Level: Low / Medium / High
- Detected Risks
- Estimated Cost Impact
- Recommended Actions

Compliance and cost estimates must be clearly presented as estimates where applicable.
Users should confirm important certification or testing costs with the relevant professional provider.

## Demo Cases

Primary demo product cases:

- Children's wooden blocks
- Bluetooth earphones

## Out of Scope

The MVP does not include:

- Global market coverage
- Automatic HS Code determination
- Automatic customs declaration
- Tariff avoidance strategies
- Supplier credit investigation

## MVP Principle

Build a focused demo that proves this core value:

The Agent understands the purchasing goal, identifies relevant compliance risks,
asks for missing information, uses tools, and converts those risks into actionable
cost and purchasing decisions.