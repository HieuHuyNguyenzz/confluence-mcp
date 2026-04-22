FROM python:3.12-slim

# Build-time proxy arguments
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Set as environment variables for pip and other tools
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY

WORKDIR /app

# Copy project files first
COPY pyproject.toml .
COPY src/confluence_mcp /app/confluence_mcp

# Upgrade pip and install build dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir .

# Unset proxy for runtime to avoid issues with internal services
ENV http_proxy=
ENV https_proxy=
ENV no_proxy=

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "-m", "confluence_mcp.server", "--transport", "streamable-http", "--port", "8000"]
