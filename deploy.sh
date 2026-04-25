#!/bin/bash

# Configuration
IMAGE_NAME="bitu-confluence-mcp:v1"
CONTAINER_NAME="bitu-confluence-mcp"

# Proxy for build-time dependency installation (Allow override via env vars)
HTTP_PROXY="${HTTP_PROXY:-http://10.255.249.100:3128}"
HTTPS_PROXY="${HTTPS_PROXY:-http://10.255.249.100:3128}"
NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,.viettelcyber.com}"

# Using values from the working reference MCP
CONFLUENCE_HOST="confluence.viettelcyber.com"
CONFLUENCE_IP="10.255.244.100"
DNS_SERVER="10.255.244.100"

echo "Building Docker image: $IMAGE_NAME..."
docker build --no-cache --network host \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  --build-arg NO_PROXY="$NO_PROXY" \
  -t "$IMAGE_NAME" .

echo "Starting container: $CONTAINER_NAME..."
# Remove existing container if it exists
docker rm -f $CONTAINER_NAME 2>/dev/null || true

docker run -d \
  -p 8000:8000 \
  --name $CONTAINER_NAME \
  --dns $DNS_SERVER \
  --dns 8.8.8.8 \
  --add-host ${CONFLUENCE_HOST}:${CONFLUENCE_IP} \
  --env-file .env \
  $IMAGE_NAME

echo "Deployment complete. Server is running on port 8000."
