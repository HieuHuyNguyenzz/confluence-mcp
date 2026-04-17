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
LLM_CHUNKING_PROMPT = """Reasoning: low

# Role:
Bạn là một hệ thống **chunking văn bản thông minh**.

### Nhiệm vụ:
Nhận đầu vào là **một văn bản dài** và chia nhỏ nó thành các đoạn (**chunk**) theo các quy tắc sau:

---

### Quy tắc chia chunk:
1. **Độ dài mỗi chunk**: từ **100–200 từ** (có thể điều chỉnh tùy nhu cầu).
2. **Không cắt giữa câu** — luôn giữ nguyên các câu đầy đủ.
3. **Giữ cấu trúc logic**:
   - Nếu văn bản có **các tiêu đề, mục, hoặc đoạn rõ ràng** (`#`, `##`, `###`), hãy ưu tiên chia theo các phần đó.
   - Một **tiêu đề** và **nội dung liên quan** phải nằm cùng trong một chunk (trừ khi vượt quá giới hạn từ).
4. Nếu văn bản **ngắn hơn giới hạn chunk**, hãy giữ nguyên.
5. **Giữ nguyên các ký tự định dạng tiêu đề** như `#`, `##`, `###`.
6. **Xóa các ký tự vô nghĩa** (như khoảng trắng thừa, ký tự đặc biệt không cần thiết).
7. Nếu có **bảng, danh sách hoặc dữ liệu dạng bảng**, hãy đưa nguyên bảng vào **một chunk riêng biệt**.
8. **Không thay đổi, thêm, hoặc bỏ nội dung gốc**, trừ việc chia nhỏ theo quy tắc trên.

---

### 🧩 Cấu trúc đầu ra:
Kết quả phải được xuất theo định dạng **JSON** như sau:
```json
{
  "chunks": [
    {
      "content": "Title + '\\n' + đoạn văn bản + '\\n' + context",
      "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    },
    {
      "content": "Title + '\\n' + đoạn văn bản + '\\n' + context",
      "keywords": ["keywordA", "keywordB", "keywordC", "keywordD", "keywordE"]
    }
  ]
}
```

### Yêu cầu chi tiết cho từng trường:
- **content**:
  - Bắt đầu bằng title tổng (thường ở dòng đầu tiên của chunk đầu tiên).
  - Giữ nguyên heading (1, 2, 3...) tương ứng.
  - Bao gồm toàn bộ đoạn văn bản thuộc chunk.
  - Cuối mỗi chunk, thêm một đoạn context ngắn gọn, tóm tắt nội dung hoặc ngữ cảnh của đoạn đó trong toàn bộ tài liệu (Bắt đầu bằng: Đoạn này .....).
  - → Mục tiêu: cải thiện khả năng tìm kiếm và gợi nhớ nội dung.
- **keywords**:
  - Trích xuất 5–10 từ khóa quan trọng nhất của chunk (tùy theo độ dài).
  - Từ khóa nên phản ánh chủ đề chính, khái niệm trọng tâm, hoặc thuật ngữ đặc trưng của đoạn văn.
  - Viết thường, không trùng lặp.

### Lưu ý:
- Không thêm bất kỳ văn bản, ký tự, hay bình luận nào ngoài định dạng JSON trên.
- Không dịch, không diễn giải lại nội dung.
- Đảm bảo từng chunk độc lập và có ngữ cảnh riêng.

Văn bản cần chunk:
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
            log.info(f"Dropping existing collection '{self.collection_name}'...")
            utility.drop_collection(self.collection_name)

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
        
        data = [
            {
                "id": c["id"],
                "vector": c["embedding"],
                "content": c["content"][:65534],
                "group": c["group"],
                "domain": c["domain"],
                "column_type": c["column_type"],
                "aggregation": c["aggregation"],
                "keywords": ", ".join(c["keywords"]) if isinstance(c["keywords"], list) else c["keywords"],
                "source_file": c["source_file"],
                "source_title": c["source_title"],
                "heading": c["heading"],
            }
            for c in batch
        ]
        collection.insert(data=data)
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

    def embed_query(self, text: str) -> List[float]:
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

    def _parse_json_response(self, content: str) -> Any:
        """Robustly extract JSON from LLM response."""
        import json
        import re
        
        # 1. Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = content.strip()
        
        # 2. Try direct load
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
        # 3. Try extracting content between first { and last }
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end+1])
        except json.JSONDecodeError:
            pass
            
        return None

    def _split_for_llm(self, text: str) -> List[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        
        splits = []
        for i in range(0, len(text), self.max_chunk_size):
            splits.append(text[i:i + self.max_chunk_size])
        return splits

    async def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        initial_splits = self._split_for_llm(text)
        all_chunks = []
        
        for i, segment in enumerate(initial_splits):
            prompt_with_text = self.prompt.format(text=segment)
            
            try:
                response = await self.llm.ainvoke(prompt_with_text)
                content = response.content
                
                data = self._parse_json_response(content)
                
                if data is not None:
                    if isinstance(data, dict) and "chunks" in data:
                        chunks = data["chunks"]
                        if isinstance(chunks, list):
                            all_chunks.extend(chunks)
                        else:
                            all_chunks.append({"content": segment, "keywords": []})
                    elif isinstance(data, list):
                        all_chunks.extend(data)
                    else:
                        all_chunks.append({"content": segment, "keywords": []})
                else:
                    log.warning(f"LLM returned non-JSON content for segment {i}. Content: {content[:200]}...")
                    all_chunks.append({"content": segment, "keywords": []})
                        
            except Exception as e:
                log.warning(f"LLM chunking failed for segment {i}: {e}")
                all_chunks.append({"content": segment, "keywords": []})
            
            await asyncio.sleep(0.2)
        
        return all_chunks


async def chunk_text(
    text: str,
    unique_id: str,
    title: str,
    space_key: str,
    source_type: str = "page",
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
    for i, split_data in enumerate(splits):
        if isinstance(split_data, dict):
            content = split_data.get("content", "").strip()
            keywords = split_data.get("keywords", [])
        else:
            content = split_data.strip()
            keywords = []
            
        if not content:
            continue
            
        uid = hashlib.sha1(
            f"{unique_id}::{i}::{content[:80]}".encode()
        ).hexdigest()[:16]
        
        if source_type == "attachment":
            source_file = title
        else:
            source_file = f"{space_key}/{title}.md"
        
        chunk = {
            "id": uid,
            "content": content,
            "source_file": source_file,
            "source_title": title,
            "group": space_key,
            "heading": "",
            **DEFAULT_METADATA,
            "keywords": keywords if keywords else DEFAULT_METADATA.get("keywords", "")
        }
        chunks.append(chunk)
    return chunks

async def sync_single_space(c_client, m_client, e_client, s_key, sem, state_manager, stats):
    try:
        log.info(f"--- Syncing attachments in space: {s_key} ---")
        pages = await c_client.get_all_pages_paginated(s_key)
        log.info(f"Found {len(pages)} pages in {s_key} to check for attachments.")
        
        for p in pages:
            p_id = p.get("id")
            if state_manager.is_synced(p_id):
                log.debug(f"Skipping already synced page attachments: {p_id}")
                continue
                
            # Process attachments for this page
            attachments = await c_client.get_all_attachments_paginated(p_id)
            if attachments:
                page_title = p.get("title", "Untitled")
                log.info(f"Processing {len(attachments)} attachments for page {page_title}...")
                
                for att in attachments:
                    att_filename = att.get("title", "")
                    att_download_path = att.get("_links", {}).get("download")
                    
                    if not att_filename or not att_download_path:
                        continue
                    
                    async with sem:
                        try:
                            # 1. Download attachment
                            att_bytes = await c_client.download_attachment(att_download_path)
                            
                            # 2. Extract text to Markdown
                            from confluence_mcp.extractor import FileExtractor
                            att_text = FileExtractor.extract(att_filename, att_bytes)
                            
                            if not att_text or len(att_text.strip()) < 10:
                                continue
                            
                            # 3. Process attachment content: Chunk -> Embed -> Save
                            att_chunks = await chunk_text(
                                att_text,
                                f"{p_id}_{att_filename}",
                                att_filename,
                                s_key,
                                source_type="attachment"
                            )
                            
                            if att_chunks:
                                embedded_att_chunks = await e_client.embed_chunks(att_chunks)
                                if embedded_att_chunks:
                                    for i in range(0, len(embedded_att_chunks), 100):
                                        batch = embedded_att_chunks[i:i + 100]
                                        m_client.upload_batch(batch)
                                    stats["chunks"] += len(embedded_att_chunks)
                                    log.info(f"Synced attachment: {att_filename}")
                                    
                        except Exception as att_e:
                            log.warning(f"Failed to sync attachment {att_filename}: {att_e}")
            
            state_manager.mark_synced(p_id)
            stats["success"] += 1
            
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