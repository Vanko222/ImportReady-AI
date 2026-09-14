# ImportReady AI — AWS Deployment (ECS Express Mode)

Deployment preparation for the approved Consumer MVP. This document contains **no
credentials**: the provider key is supplied at runtime by AWS, never stored here, in the
image, or in the repository.

## 1. Target architecture

```
Amazon ECR (private repository)
  └─ Amazon ECS Express Mode service
       └─ AWS Fargate task (this container image)
            └─ public Application Load Balancer
                 └─ AWS-managed HTTPS public URL
```

Express Mode provisions and manages the load balancer, listener/TLS certificate, target
group, Auto Scaling and the public DNS name; you supply the image, the container port, a
health-check path and the environment.

| Item | Value |
| --- | --- |
| Image | `<account>.dkr.ecr.<region>.amazonaws.com/importready-ai:<tag>` |
| Container port | `8501` |
| Health check path | `/_stcore/health` (Streamlit's own endpoint; no provider access) |
| Entrypoint | `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true` |
| Compute | Fargate, 1 vCPU / 2 GB is a reasonable starting point for the Streamlit MVP |
| Provider | `deepseek` / `deepseek-v4-flash` (`VERIFIED`, `ui_exposed=True`) |

## 2. Runtime environment variables

Set these on the ECS task definition (Express Mode "Environment variables"). **The
provider key must be an AWS secret**, not a plain variable.

| Name | Kind | Value / source |
| --- | --- | --- |
| `MODEL_PROVIDER` | non-secret variable | `deepseek` |
| `MODEL_ID` | non-secret variable | `deepseek-v4-flash` |
| `API_KEY` | **secret** | AWS Secrets Manager, injected via the task definition `secrets` valueFrom (or an SSM SecureString parameter). Never a plain task environment value, never in the image. |
| `IMPORTREADY_DATA_DIR` | optional, non-secret | Only if the approved data set is mounted elsewhere; defaults to `/app/data` inside the image. |

No other configuration is required: the image already sets `STREAMLIT_SERVER_ADDRESS=0.0.0.0`,
`STREAMLIT_SERVER_PORT=8501` and `STREAMLIT_SERVER_HEADLESS=true`, and `.streamlit/config.toml`
(theme + `gatherUsageStats = false`) is copied into the image at `/app/.streamlit/config.toml`.

Note: the shared `requirements.txt` also pins `pytest` (a development dependency). It is
harmless for the MVP — no test files are shipped — and can be split into a runtime-only
requirements file in a later round.

The application keeps its existing gate: the Competition Demo path accepts a model only when
the exact registered combination is `VERIFIED`; BYOK continues to work independently, with the
key held in session state only.

## 3. Build and push the image (ECR workflow)

```powershell
# 0) One-time: create the private repository
aws ecr create-repository --repository-name importready-ai --region <region>

# 1) Build locally (helper script; add -Push to publish)
powershell -File scripts/aws_build_image.ps1 -Tag deploy
#    or manually:
docker build -t importready-ai:deploy .

# 2) Authenticate, tag, push (manual equivalent)
$registry = "<account>.dkr.ecr.<region>.amazonaws.com"
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin $registry
docker tag importready-ai:deploy "$registry/importready-ai:deploy"
docker push "$registry/importready-ai:deploy"
```

Tag each release with an immutable identifier as well (for example the Git short SHA:
`importready-ai:deploy` plus `importready-ai:a1b2c3d`) so a rollback has a concrete target.

## 4. Create the ECS Express Mode service

1. ECS console → **Create service** → choose **Express mode**.
2. Image URI: the ECR URI pushed above.
3. Container port: `8501`. Health check path: `/_stcore/health`.
4. Environment variables and secret: as in §2.
5. Task size/logging: keep the default CloudWatch log group so container output is
   inspectable.
6. Create the service. Express Mode provisions the ALB, HTTPS listener/certificate and public
   URL; note the resulting `https://…` address.

CLI users can drive the same flow with the Express Mode create operation of the current AWS
CLI (`aws ecs create-express-gateway-service …`); confirm the flag names against the current
AWS documentation before running, since Express Mode is evolving.

Notes:

* Streamlit uses a WebSocket connection. The Express Mode managed ALB handles the upgrade; if
  a hand-rolled ALB is used instead, enable target-group stickiness and WebSocket support.
* Do not set `browser.serverAddress`/`serverAddress` to `localhost`: the public HTTPS URL is
  resolved from the request, and Streamlit's CORS/XSRF protections stay enabled.

## 5. Rollback concept

* **Fast rollback (recommended):** deploy the previous immutable image tag and create/update
  the Express service with that URI. Keep at least the current and previous tags in ECR.
* **Task-definition rollback:** ECS keeps previous task definition revisions — repoint the
  service to the last revision that was healthy.
* **Config rollback:** environment variables and the secret reference live on the task
  definition, so a bad variable change is reverted by the same revision switch. The secret
  itself is rotated in Secrets Manager, not redeployed.
* Rollback validation: `GET <public-url>/_stcore/health` returns `ok` and the product intake
  page loads.

## 6. Public E2E checklist (after deployment)

1. `GET https://<public-url>/_stcore/health` → `ok` (ALB target healthy, task running).
2. Root page loads over HTTPS with no mixed-content or WebSocket errors in the browser
   console.
3. Default page shows **one** intake box + **Analyze Product** only (no questionnaire, no
   "AI understood" before submitting).
4. Paste a plain product description → Analyze Product → an AI category suggestion appears
   (Competition Demo provider resolved from the runtime env/secret).
5. Correct the category to a different one → the previous details are discarded and the
   selector follows the new category.
6. Confirm the bundle → canonical analysis runs → the natural-language customer report
   renders (no rule ids in the default report).
7. **Technical details** expander shows canonical rule results, evidence and the attribute
   questionnaire; **Explore a scenario** runs the What-if engine.
8. Language `English ⇄ 中文` and Appearance `Light ⇄ Dark` still work.
9. BYOK mode: paste a key → the picker offers only the verified provider; Clear Key removes
   it; the key never appears in the report, technical details or logs.
10. Sensitive input: pasting a credential-shaped string into the intake box is rejected
    safely before any provider call.
11. Start Over clears the product state and keeps language/theme.
12. CloudWatch logs contain no credential material.

## 7. Cleanup reminder (avoid lingering cost)

Express Mode creates several billable resources. When the demo is finished, delete them:

1. **ECS service** — deleting the Express Mode service also removes its managed load balancer,
   listener/certificate association, target group and scaling policy.
2. **ECR repository** (and its images) if the deployment is not being kept.
3. **Secrets Manager secret** (schedule deletion, e.g. 7-day recovery window) or the SSM
   parameter.
4. **CloudWatch log group** created for the task.
5. Confirm no orphaned ALB, target group, NAT/Elastic IP or Fargate tasks remain in the
   region.

## 8. Local container verification status

The image is prepared for the platform, but **the local `docker build` / `docker run` /
container health smoke test was intentionally skipped by human decision** (the Docker daemon
is not reachable from this session, and the sandbox escalation was declined). The equivalent
offline evidence that *was* produced:

* the exact container start command runs locally and serves `/_stcore/health` → `200 ok` and
  `/` → `200`;
* `python -m pytest -q` → 0 failed;
* a static build-context audit (`.dockerignore` rules applied to the repository tree) shows
  the required files ship and no secret/development artefact does.

Run the container smoke test on a machine with a working Docker daemon before the first
deployment:

```powershell
docker build -t importready-ai:deploy .
docker run --rm -d --name importready-ai-deploy-test -p 8502:8501 importready-ai:deploy
Invoke-WebRequest -UseBasicParsing http://localhost:8502/_stcore/health   # -> ok
Invoke-WebRequest -UseBasicParsing http://localhost:8502                  # -> 200
docker stop importready-ai-deploy-test
```
