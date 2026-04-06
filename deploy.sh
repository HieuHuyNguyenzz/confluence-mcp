#!/bin/bash

# Configuration
IMAGE_NAME="bitu-confluence-mcp:v1"
CONTAINER_NAME="bitu-confluence-mcp"
# Using values from the working reference MCP
CONFLUENCE_HOST="confluence.viettelcyber.com"
CONFLUENCE_IP="10.255.244.100"
DNS_SERVER="10.255.244.100"

echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .

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
