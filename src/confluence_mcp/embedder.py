"""Embedding client for external embedding API server."""

from typing import Any

import httpx


class EmbeddingClient:
    """Client for calling external embedding API.

    Configurable endpoint, method, and payload format.
    Adjust _build_payload() and _parse_response() to match your API.
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/embed",
        api_key: str = "",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text chunk.

        Returns embedding vector as list of floats.
        """
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text chunks in one API call.

        Returns list of embedding vectors.
        """
        if not texts:
            return []

        payload = self._build_payload(texts)
        result = await self._post(self.endpoint, payload)
        return self._parse_response(result, len(texts))

    async def _post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # Override these methods to match your embedding API format
    # ============================================================

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        """Build request payload for embedding API.

        Default format: {"texts": ["chunk1", "chunk2", ...]}

        Modify this to match your API's expected input.
        Common alternatives:
        - OpenAI: {"input": texts, "model": "text-embedding-3-small"}
        - Single: {"text": texts[0]}
        - Custom: {"documents": texts, "options": {...}}
        """
        return {"texts": texts}

    def _parse_response(self, result: dict, expected_count: int) -> list[list[float]]:
        """Parse embedding API response into list of vectors.

        Default format: {"embeddings": [[0.1, 0.2, ...], ...]}

        Modify this to match your API's response format.
        Common alternatives:
        - OpenAI: {"data": [{"embedding": [...]}, ...]}
        - Simple: {"embeddings": [[...], [...]]}
        - Direct: [[0.1, ...], [0.2, ...]]  (top-level array)
        - Single: {"embedding": [...]}
        """
        if "embeddings" in result:
            embeddings = result["embeddings"]
        elif "data" in result:
            embeddings = [item.get("embedding", []) for item in result["data"]]
        elif "embedding" in result:
            embeddings = [result["embedding"]]
        elif isinstance(result, list):
            embeddings = result
        else:
            raise ValueError(
                f"Unknown embedding response format. "
                f"Keys: {list(result.keys())}. "
                f"Override _parse_response() in EmbeddingClient to match your API."
            )

        if len(embeddings) != expected_count:
            raise ValueError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}"
            )

        return embeddings
