FROM python:3.12-slim

WORKDIR /app

# Copy project files first
COPY pyproject.toml .
COPY src/confluence_mcp /app/confluence_mcp

# Upgrade pip and install build dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "-m", "confluence_mcp.server", "--transport", "streamable-http", "--port", "8000"]
