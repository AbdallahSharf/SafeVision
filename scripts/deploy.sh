#!/usr/bin/env bash
# =============================================================================
# SafeVision — Manual deployment to Google Compute Engine
# =============================================================================
#
# Prerequisites:
#   1. gcloud CLI installed & authenticated
#   2. Docker installed locally
#   3. GCE VM already created (see README.md for setup)
#   4. Artifact Registry repository created
#
# Usage:
#   ./scripts/deploy.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-me-west1}"
REPO="safevision"
IMAGE="safevision"
TAG="${1:-latest}"
VM_NAME="${GCE_VM_NAME:-safevision-vm}"
VM_ZONE="${GCE_VM_ZONE:-me-west1-c}"

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "🔨 Building Docker image …"
docker build -t "${FULL_IMAGE}" .

echo "📦 Pushing to Artifact Registry …"
docker push "${FULL_IMAGE}"

echo "🚀 Deploying to GCE VM (${VM_NAME}) …"
gcloud compute ssh "${VM_NAME}" \
    --zone="${VM_ZONE}" \
    --project="${PROJECT_ID}" \
    --command="
        # Authenticate Docker with Artifact Registry
        gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

        # Pull latest image
        docker pull ${FULL_IMAGE}

        # Stop old container (if running)
        docker stop safevision 2>/dev/null || true
        docker rm safevision 2>/dev/null || true
        docker system prune -af

        # Ensure unauthorized faces directory exists
        sudo mkdir -p /opt/safevision/unauthorized_faces
        sudo chmod 775 /opt/safevision/unauthorized_faces

        # Start new container
        docker run -d \
            --name safevision \
            --restart unless-stopped \
            --network host \
            --gpus all \
            -v /opt/safevision/unauthorized_faces:/opt/safevision/unauthorized_faces \
            --env-file /opt/safevision/.env \
            ${FULL_IMAGE}

        echo '✅ SafeVision deployed successfully!'
    "

echo ""
echo "✅ Deployment complete!"
echo "   Stream URL: http://\$(gcloud compute instances describe ${VM_NAME} --zone=${VM_ZONE} --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):8080/stream"
