"""Sync Confluence data to Milvus Vector Database."""

import os
import asyncio
import json
import hashlib
import logging
import time
from typing import Any, List, Dict
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

from confluence_mcp.client import ConfluenceClient
from confluence_mcp.processor import process_page

load_dotenv()

# ─────────────────────────── LOGGER ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────── CONFIG ────────────────────────────

MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "confluence_knowledge")

# Embedding API for Vector Storage
EMBEDDING_URL = os.getenv("EMBEDDING_API_URL")

# OpenAI Config for LLM Chunking
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# LLM Chunking Prompt
LLM_CHUNKING_PROMPT = """Bạn là chuyên gia phân tích và xử lý tài liệu. Nhiệm vụ của bạn là chia văn bản dưới đây thành các đoạn nhỏ (chunks) theo ngữ nghĩa.

NGUYÊN TẮC CHIA:
1. Mỗi chunk phải là một đơn vị thông tin HOÀN CHỈNH về mặt ngữ nghĩa
2. Không chia cắt giữa chừng một đoạn văn, danh sách, hoặc bảng
3. Các tiêu đề phụ nên thuộc về chunk của phần nội dung theo sau nó
4. Giới hạn mỗi chunk từ 200-1000 từ

YÊU CẦU TRẢ LỜI:
Trả về duy nhất một mảng JSON các chuỗi văn bản, không giải thích gì thêm.

Văn bản cần chia:
{text}"""

# Batch embedding config
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Default metadata since enrichment is skipped
DEFAULT_METADATA = {
    "domain": "general",
    "keywords": "",
    "column_type": "overview",
    "aggregation": "detail",
}

# ─────────────────────────── STATE MANAGER ────────────────────────────

class SyncStateManager:
    def __init__(self, state_file: str = "sync_state_milvus.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Error loading state file: {e}")
        return {}

    def is_synced(self, page_id: str) -> bool:
        return page_id in self.state

    def mark_synced(self, page_id: str):
        self.state[page_id] = "synced"
        self._save_state()

    def _save_state(self):
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            log.error(f"Error saving state file: {e}")

# ─────────────────────────── MILVUS CLIENT ────────────────────────────

class MilvusClient:
    def __init__(self, url: str, collection_name: str, token: str = None):
        self.url = url
        self.token = token
        self.collection_name = collection_name
        self._dim = None
        self._connect()

    def _connect(self):
        if self.token:
            connections.connect("default", uri=self.url, token=self.token)
            log.info(f"Connected to Milvus at {self.url} (with token)")
        else:
            connections.connect("default", uri=self.url)
            log.info(f"Connected to Milvus at {self.url}")

    def set_dimension(self, dim: int):
        self._dim = dim

    def ensure_collection(self):
        if not self._dim:
            raise ValueError("Dimension not set. Run embedding test first.")
        
        if utility.has_collection(self.collection_name):
            log.info(f"Collection '{self.collection_name}' already exists.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="group", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="column_type", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="aggregation", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1000),
        ]
        schema = CollectionSchema(fields, description="Confluence Knowledge Base")
        collection = Collection(name=self.collection_name, schema=schema)
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        )
        log.info(f"Created collection '{self.collection_name}' with dim={self._dim}")

    def upload_batch(self, batch: List[Dict[str, Any]]):
        collection = Collection(self.collection_name)
        collection.load()
        
        ids = [c["id"] for c in batch]
        vectors = [c["embedding"] for c in batch]
        scalar_data = {
            "content": [c["content"][:65534] for c in batch],
            "group": [c["group"] for c in batch],
            "domain": [c["domain"] for c in batch],
            "column_type": [c["column_type"] for c in batch],
            "aggregation": [c["aggregation"] for c in batch],
            "keywords": [c["keywords"] for c in batch],
            "source_file": [c["source_file"] for c in batch],
            "source_title": [c["source_title"] for c in batch],
            "heading": [c["heading"] for c in batch],
        }
        collection.insert(ids=ids, vectors=vectors, scalar_data=scalar_data)
        collection.flush()

# ─────────────────────────── EMBEDDING CLIENT ────────────────────────────

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

# ─────────────────────────── PIPELINE ────────────────────────────

class CustomEmbeddings(Embeddings):
    def __init__(self, url: str):
        self.url = url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        resp = requests.post(
            self.url,
            json={"input": texts},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def embed_query(self, text: str) -> List[float]]:
        return self.embed_documents([text])[0]


class LLMChunker:
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str,
        openai_model: str,
        prompt: str,
        max_chunk_size: int = 8000,
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
            temperature=0,
        )
        self.prompt = prompt
        self.max_chunk_size = max_chunk_size

    def _split_for_llm(self, text: str) -> List[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        
        splits = []
        for i in range(0, len(text), self.max_chunk_size):
            splits.append(text[i:i + self.max_chunk_size])
        return splits

    async def chunk_text(self, text: str) -> List[str]:
        initial_splits = self._split_for_llm(text)
        all_chunks = []
        
        for i, segment in enumerate(initial_splits):
            prompt_with_text = self.prompt.format(text=segment)
            
            try:
                response = await self.llm.abound(prompt_with_text)
                content = response.content
                
                try:
                    import json
                    chunks = json.loads(content)
                    if isinstance(chunks, list):
                        all_chunks.extend(chunks)
                    else:
                        all_chunks.append(segment)
                except json.JSONDecodeError:
                    import re
                    matches = re.findall(r'\[.*\]', content, re.DOTALL)
                    if matches:
                        try:
                            chunks = json.loads(matches[0])
                            all_chunks.extend(chunks)
                        except:
                            all_chunks.append(segment)
                    else:
                        all_chunks.append(segment)
                        
            except Exception as e:
                log.warning(f"LLM chunking failed for segment {i}: {e}")
                all_chunks.append(segment)
            
            await asyncio.sleep(0.2)
        
        return all_chunks


async def chunk_text(
    text: str,
    page_id: str,
    title: str,
    space_key: str,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    if use_llm and OPENAI_API_KEY:
        log.info("Using LLM Chunking for better context preservation...")
        try:
            chunker = LLMChunker(
                openai_api_key=OPENAI_API_KEY,
                openai_base_url=OPENAI_BASE_URL,
                openai_model=OPENAI_MODEL,
                prompt=LLM_CHUNKING_PROMPT,
            )
            splits = await chunker.chunk_text(text)
            log.info(f"LLM chunking produced {len(splits)} chunks")
        except Exception as e:
            log.warning(f"LLM chunking failed: {e}, falling back to basic splitting")
            splits = text.split("\n\n")
    else:
        splits = text.split("\n\n")
    
    chunks = []
    for i, content in enumerate(splits):
        content = content.strip()
        if not content:
            continue
            
        uid = hashlib.sha1(
            f"{page_id}::{i}::{content[:80]}".encode()
        ).hexdigest()[:16]
        
        chunk = {
            "id": uid,
            "content": content,
            "source_file": f"{space_key}/{title}.md",
            "source_title": title,
            "group": space_key,
            "heading": "",
            **DEFAULT_METADATA
        }
        chunks.append(chunk)
    return chunks

async def sync_single_space(c_client, m_client, e_client, s_key, sem, state_manager, stats):
    try:
        log.info(f"--- Syncing space: {s_key} ---")
        pages = await c_client.get_all_pages_paginated(s_key)
        log.info(f"Found {len(pages)} pages in {s_key}.")
        
        for p in pages:
            p_id = p.get("id")
            if state_manager.is_synced(p_id):
                log.debug(f"Skipping already synced page: {p_id}")
                continue
                
            async with sem:
                try:
                    # 1. Extraction
                    p_data = await process_page(c_client, p, s_key, True, sem)
                    title = p_data["title"]
                    full_text = p_data["content"]
                    
                    if p_data["attachments"]:
                        att_txt = "\n\n### Attachments\n"
                        for a in p_data["attachments"]:
                            att_txt += f"**{a['filename']}**:\n{a['content']}\n\n"
                        full_text += att_txt
                    
                    # 2. Chunking
                    chunks = await chunk_text(full_text, p_id, title, s_key)
                    if not chunks:
                        continue
                    
                    # 3. Batch Embedding
                    embedded_chunks = await e_client.embed_chunks(chunks)
                    
                    # 4. Storage
                    if embedded_chunks:
                        for i in range(0, len(embedded_chunks), 100):
                            batch = embedded_chunks[i:i + 100]
                            m_client.upload_batch(batch)
                    
                    state_manager.mark_synced(p_id)
                    log.info(f"Successfully synced page: {title} ({len(embedded_chunks)} chunks)")
                    stats["success"] += 1
                    stats["chunks"] += len(embedded_chunks)
                    
                except Exception as e:
                    stats["failure"] += 1
                    log.error(f"Failed to sync page {p.get('title', p_id)}: {e}")
                
        log.info(f"Completed sync for space: {s_key}")
    except Exception as e:
        log.error(f"Critical error syncing space {s_key}: {e}")

async def main():
    c_url = os.getenv("CONFLUENCE_URL")
    c_tok = os.getenv("CONFLUENCE_API_TOKEN")
    v_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"
    
    if not all([c_url, c_tok, EMBEDDING_URL]):
        log.error("Missing required env vars: CONFLUENCE_URL, CONFLUENCE_API_TOKEN, EMBEDDING_API_URL")
        return

    c_client = ConfluenceClient(base_url=c_url, api_token=c_tok, verify_ssl=v_ssl)
    m_client = MilvusClient(url=MILVUS_URL, collection_name=COLLECTION_NAME, token=MILVUS_TOKEN)
    e_client = EmbeddingClient(url=EMBEDDING_URL, batch_size=BATCH_SIZE)
    state_manager = SyncStateManager()
    
    dim = await e_client.get_embedding_dim()
    m_client.set_dimension(dim)
    m_client.ensure_collection()
    
    sem = asyncio.Semaphore(10)
    stats = {"success": 0, "failure": 0, "chunks": 0}
    t_s_key = os.getenv("SYNC_SPACE_KEY")
    
    try:
        if t_s_key:
            log.info(f"Targeted sync mode: space {t_s_key}")
            await sync_single_space(c_client, m_client, e_client, t_s_key, sem, state_manager, stats)
        else:
            log.info("Full system sync mode: Crawling all spaces...")
            spaces = await c_client.list_spaces()
            if not spaces:
                log.info("No spaces found.")
                return
            for s in spaces:
                sk = s.get("key") if isinstance(s, dict) else s
                if sk:
                    await sync_single_space(c_client, m_client, e_client, sk, sem, state_manager, stats)
        
        log.info("\n--- Sync Summary (Milvus) ---")
        log.info(f"Successfully processed: {stats['success']} pages")
        log.info(f"Total chunks embedded: {stats['chunks']}")
        log.info(f"Failed: {stats['failure']} pages")
        log.info("----------------------------\n")
    finally:
        await c_client.close()
        await e_client.close()

if __name__ == "__main__":
    asyncio.run(main())