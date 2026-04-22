"""Embedding API client for vectorizing text."""

import asyncio
import logging
import httpx
from typing import List, Dict, Any

log = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, url: str, batch_size: int = 32):
        self.url = url
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(timeout=60.0)
        self._dim = None

    async def close(self):
        await self._client.aclose()

    async def get_embedding_dim(self) -> int:
        test_vec = await self._get_embedding(["test"])
        self._dim = len(test_vec[0])
        log.info(f"Embedding dimension: {self._dim}")
        return self._dim

    async def _get_embedding(self, texts: List[str], retry: int = 3) -> List[List[float]]:
        for attempt in range(retry):
            try:
                resp = await self._client.post(
                    self.url,
                    json={"input": texts},
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 429:
                    wait_time = 2 ** attempt
                    log.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = 2 ** attempt
                    log.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                log.warning(f"Attempt {attempt + 1}/{retry} failed: {e}")
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embedded_chunks = []
        total = len(chunks)
        
        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c["content"] for c in batch]
            
            log.info(f"Embedding batch {i // self.batch_size + 1}/{(total + self.batch_size - 1) // self.batch_size} ({len(texts)} texts)...")
            
            try:
                vectors = await self._get_embedding(texts)
                for j, chunk in enumerate(batch):
                    chunk["embedding"] = vectors[j]
                    embedded_chunks.append(chunk)
            except Exception as e:
                log.error(f"Batch embedding failed: {e}")
                continue
            
            await asyncio.sleep(0.1)
        
        return embedded_chunks
