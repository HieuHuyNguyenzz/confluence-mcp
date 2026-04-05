FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

COPY src/ ./src/

EXPOSE 8000

CMD ["python", "-m", "confluence_mcp.server", "--transport", "streamable-http", "--port", "8000"]
