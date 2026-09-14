# ImportReady AI - consumer Streamlit MVP
# Target: Amazon ECR -> Amazon ECS Express Mode -> Fargate -> public ALB (managed HTTPS URL)
#
# Build:  docker build -t importready-ai:deploy .
# Run:    docker run --rm -p 8501:8501 \
#           -e MODEL_PROVIDER=deepseek -e MODEL_ID=deepseek-v4-flash -e API_KEY=<runtime secret> \
#           importready-ai:deploy
#
# No credential is baked into this image. API_KEY / BYOK keys are supplied by the
# runtime environment only (ECS task environment + secrets); nothing is copied from
# a local .env or credential file, and no secret is set as an image ENV.

FROM python:3.11-slim

# Deterministic, unbuffered container behaviour. Streamlit's bind address/port/headless
# mode are also exported so any plain `streamlit run app.py` inside the image is
# already deployment-correct; the CMD below states them explicitly as well.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# The only OS package needed: curl, used by the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so this layer stays cached while only application code changes.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Application code. .dockerignore keeps .env, .git, virtualenvs, caches, tests,
# development reports and local certification evidence out of the build context.
COPY .streamlit/ ./.streamlit/
COPY app.py ./
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501

# Streamlit's own health endpoint; no provider or network access is involved.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Production start command: no development reload mode.
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
