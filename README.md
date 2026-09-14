# ImportReady AI

ImportReady AI is an AI-powered U.S. import compliance assistant for cross-border sellers, importers, small businesses, and trade professionals.

Users describe a product in natural language. A Strands Agent helps identify the product category and extract explicitly stated product information. After human confirmation, deterministic compliance engines evaluate applicability, risk, recommended actions, cost references, missing information, and what-if scenarios.

## MVP Scope

- Children's toys
- Small consumer electronics

## Key Features

- Natural-language product intake
- Human-confirmed AI category and fact extraction
- Strands Agents tool workflow
- Deterministic compliance analysis
- Risk and recommended-action engines
- Compliance cost references
- What-if analysis
- Evidence and source traceability
- English / Chinese interface
- Competition Demo and BYOK modes

## Architecture

User
→ Streamlit Web UI
→ Human Confirmation
→ Strands Agent + DeepSeek
→ analyze_product / get_compliance_evidence
→ Deterministic Compliance Engines
→ Safe Narrative Composer
→ Response Guard
→ Customer Compliance Report

AWS deployment:

Docker
→ Amazon ECR
→ Amazon ECS / Fargate
→ Public HTTPS Demo

## Technology Stack

- Python
- Streamlit
- Strands Agents SDK
- DeepSeek deepseek-v4-flash
- Docker
- Amazon ECR
- Amazon ECS / Fargate
- AWS Systems Manager Parameter Store
- GitHub

## Local Setup

1. Create a Python virtual environment.
2. Install dependencies from requirements.txt.
3. Run:

python -m streamlit run app.py

Then open:

http://localhost:8501

## Environment Variables

API_KEY=<provider API key>
MODEL_PROVIDER=deepseek
MODEL_ID=deepseek-v4-flash

Never commit API keys or .env files.

## Security and Guardrails

- AI-generated product facts require human confirmation.
- Missing information is not automatically converted to False.
- AI cannot override canonical applicability, risk, cost, actions, or what-if results.
- Proposed or watchlist rules are not treated as current effective obligations.
- BYOK credentials are session-only and clearable.

## Limitations

ImportReady AI is an MVP planning and decision-support system.

It does not cover every U.S. import regulation and does not constitute legal advice, formal import approval, or a complete landed-cost calculation.

## License

MIT
