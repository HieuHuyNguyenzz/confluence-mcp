FROM python:3.12-slim

WORKDIR /app

# Copy project files first
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "-m", "confluence_mcp.server", "--transport", "streamable-http", "--port", "8000"]
