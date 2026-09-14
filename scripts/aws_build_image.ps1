<#
.SYNOPSIS
    Build (and optionally push) the ImportReady AI container image for ECS Express Mode.

.DESCRIPTION
    Repeatable, explicit helper for the deployment step. It never reads .env and never
    stores credentials: the Docker build uses the repository context, and a push only
    happens when -Push is passed explicitly. AWS credentials (when pushing) come from
    the standard AWS CLI credential chain - never from this script or the repository.

.EXAMPLE
    # Build only
    powershell -File scripts/aws_build_image.ps1 -Tag deploy

.EXAMPLE
    # Build, then push to ECR (requires an authenticated AWS CLI)
    powershell -File scripts/aws_build_image.ps1 -Tag deploy -Push -Region us-east-1 -AccountId 123456789012
#>
[CmdletBinding()]
param(
    [string]$Tag = "deploy",
    [string]$RepositoryName = "importready-ai",
    [string]$Region = $env:AWS_REGION,
    [string]$AccountId = $env:AWS_ACCOUNT_ID,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$localImage = "${RepositoryName}:${Tag}"

Write-Host "==> Building image $localImage from $repoRoot"
# The build context is the repository root; .dockerignore excludes every secret and
# development artefact, so no .env or credential file can enter the image.
Push-Location $repoRoot
try {
    docker build -t $localImage .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "==> Image built: $localImage"
Write-Host "    Run locally:  docker run --rm -p 8501:8501 $localImage"
Write-Host "    Health check: http://localhost:8501/_stcore/health"

if (-not $Push) {
    Write-Host "==> Push skipped (pass -Push to publish to ECR)."
    exit 0
}

if (-not $Region) { throw "-Region (or AWS_REGION) is required when pushing." }
if (-not $AccountId) { throw "-AccountId (or AWS_ACCOUNT_ID) is required when pushing." }

$registry = "${AccountId}.dkr.ecr.${Region}.amazonaws.com"
$remoteImage = "${registry}/${RepositoryName}:${Tag}"

Write-Host "==> Authenticating Docker against $registry"
aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { throw "ECR login failed with exit code $LASTEXITCODE" }

Write-Host "==> Tagging and pushing $remoteImage"
docker tag $localImage $remoteImage
if ($LASTEXITCODE -ne 0) { throw "docker tag failed with exit code $LASTEXITCODE" }
docker push $remoteImage
if ($LASTEXITCODE -ne 0) { throw "docker push failed with exit code $LASTEXITCODE" }

Write-Host "==> Pushed: $remoteImage"
Write-Host "    Use this image URI when creating the ECS Express Mode service."
