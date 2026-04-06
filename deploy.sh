#!/bin/bash

# Configuration
IMAGE_NAME="bitu-confluence-mcp:v1"
CONTAINER_NAME="bitu-confluence-mcp"
HTTP_PROXY="http://10.255.249.100:3128"
HTTPS_PROXY="http://10.255.249.100:3128"
NO_PROXY="localhost,127.0.0.1"

echo "Building Docker image: $IMAGE_NAME..."
docker build \
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
  -e HTTP_PROXY="$HTTP_PROXY" \
  -e HTTPS_PROXY="$HTTPS_PROXY" \
  -e NO_PROXY="$NO_PROXY" \
  --env-file .env \
  $IMAGE_NAME

echo "Deployment complete. Server is running on port 8000."
