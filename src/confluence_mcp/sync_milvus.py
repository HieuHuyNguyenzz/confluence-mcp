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
LLM_CHUNKING_PROMPT = \"\"\"Bạn là một hệ thống chunking văn bản thông minh chuyên dụng cho RAG.

### Nhiệm vụ:
Nhận đầu vào là một văn bản và chia nó thành các đoạn (chunks) nhỏ sao cho mỗi chunk giữ được ý nghĩa trọn vẹn và có ngữ cảnh rõ ràng.

### Quy tắc chia chunk:
1. **Độ dài**: Mỗi chunk khoảng 100-300 từ.
2. **Tính toàn vẹn**: Không cắt giữa câu. Luôn giữ nguyên các câu đầy đủ.
3. **Cấu trúc**: Ưu tiên chia theo tiêu đề (#, ##, ###) hoặc đoạn văn rõ ràng.
4. **Định dạng**: Giữ nguyên các ký tự định dạng tiêu đề. 
5. **Nội dung**: Không thay đổi, thêm, hoặc bỏ nội dung gốc.
6. **Làm giàu**: 
   - Mỗi chunk phải có 5-10 từ khóa quan trọng.
   - Cuối mỗi chunk, thêm một câu tóm tắt ngữ cảnh ngắn (Bắt đầu bằng: "Đoạn này...").

### 🧩 Định dạng đầu ra (JSON):
Bạn PHẢI trả về kết quả dưới định dạng JSON thuần túy, bắt đầu bằng `{` và kết thúc bằng `}`. Không thêm lời dẫn hoặc markdown block.

Ví dụ:
Input: \"# Hướng dẫn cài đặt\\nBước 1: Tải file. Bước 2: Chạy installer. Bước 3: Cấu hình IP.\"
Output:
{{
  \"chunks\": [
    {{
      \"content\": \"# Hướng dẫn cài đặt\\nBước 1: Tải file. Bước 2: Chạy installer. Bước 3: Cấu hình IP. (Đoạn này hướng dẫn các bước cài đặt cơ bản)\",
      \"keywords\": [\"cài đặt\", \"hướng dẫn\", \"installer\", \"IP\"]
    }}
  ]
}}

Văn bản cần chunk:
<text>
{text}
</text>\"\"\"

LLM_SUMMARY_PROMPT = \"\"\"Bạn là một chuyên gia phân tích tài liệu cấp cao.

### Nhiệm vụ:
Hãy tạo một bản tóm tắt cô đọng và toàn diện cho tài liệu dưới đây để làm ngữ cảnh toàn cục (global context).

### Yêu cầu:
1. **Mục đích**: Tài liệu này viết về cái gì? Giải quyết vấn đề gì?
2. **Thực thể**: Liệt kê các khái niệm, thuật ngữ, sản phẩm then chốt.
3. **Cấu trúc**: Mô tả ngắn gọn luồng thông tin.
4. **Phong cách**: Súc tích, khách quan, tiếng Việt, 100-200 từ.
5. **Định dạng**: Không chào hỏi, không giải thích, chỉ trả về nội dung bản tóm tắt.

Văn bản cần tóm tắt:
<text>
{text}
</text>\"\"\"

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
            FieldSchema(name="global_context", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
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
                "global_context": c.get("global_context", ""),
                "parent_id": c.get("parent_id", ""),
                "chunk_type": c.get("chunk_type", "child"),
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

    async def generate_document_summary(self, text: str) -> str:
        """Generate a global summary for the entire document to be used as context for chunks."""
        # Safe limit for a single LLM call to avoid hitting context window limits 
        # although model limit is 64k, we keep it reasonable
        limit = 100000 
        if len(text) > limit:
            log.info(f"Document too large ({len(text)} chars), summarizing in parts...")
            # Simplified recursive summarization: split into 2 parts, summarize each, then summarize result
            mid = len(text) // 2
            part1 = await self.generate_document_summary(text[:mid])
            part2 = await self.generate_document_summary(text[mid:])
            text = f"Part 1 Summary:\n{part1}\n\nPart 2 Summary:\n{part2}"
        
        try:
            prompt = LLM_SUMMARY_PROMPT.replace("{text}", text)
            response = await self.llm.ainvoke(prompt)
            summary = response.content.strip()
            log.info("Successfully generated global document summary.")
            return summary
        except Exception as e:
            log.warning(f"Failed to generate global summary: {e}")
            return ""

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

    async def chunk_text(self, text: str, global_summary: str = "") -> List[Dict[str, Any]]:
        initial_splits = self._split_for_llm(text)
        
        async def process_segment(i, segment):
            prompt_with_text = self.prompt.replace("{text}", segment)
            async with LLM_SEMAPHORE:
                try:
                    response = await self.llm.ainvoke(prompt_with_text)
                    content = response.content
                    data = self._parse_json_response(content)
                    if data is not None:
                        if isinstance(data, dict) and "chunks" in data:
                            chunks = data["chunks"]
                            if isinstance(chunks, list):
                                processed = []
                                for c in chunks:
                                    if isinstance(c, dict):
                                        original_content = c.get("content", "")
                                        c["content"] = f"--- GLOBAL CONTEXT ---\n{global_summary}\n\n--- CONTENT ---\n{original_content}"
                                        processed.append(c)
                                    else:
                                        processed.append({"content": segment, "keywords": []})
                                return processed
                        elif isinstance(data, list):
                            processed = []
                            for c in data:
                                if isinstance(c, dict):
                                    original_content = c.get("content", "")
                                    c["content"] = f"--- GLOBAL CONTEXT ---\n{global_summary}\n\n--- CONTENT ---\n{original_content}"
                                    processed.append(c)
                                else:
                                    processed.append({"content": segment, "keywords": []})
                            return processed
                    log.warning(f"LLM returned non-JSON content for segment {i}. Content: {content[:200]}...")
                    return [{"content": segment, "keywords": []}]
                except Exception as e:
                    log.warning(f"LLM chunking failed for segment {i}: {e}")
                    return [{"content": segment, "keywords": []}]

        tasks = [process_segment(i, segment) for i, segment in enumerate(initial_splits)]
        results = await asyncio.gather(*tasks)
        
        all_chunks = []
        for res in results:
            all_chunks.extend(res)
        return all_chunks


async def chunk_text(
    text: str,
    unique_id: str,
    title: str,
    space_key: str,
    global_summary: str = "",
    source_type: str = "page",
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    PARENT_CHUNK_SIZE = 2000  # Reduced to avoid 504 Gateway Timeout
    
    if not use_llm or not OPENAI_API_KEY:
        # Fallback to simple splitting if LLM is disabled or key missing
        splits = text.split("\n\n")
        chunks = []
        for i, split in enumerate(splits):
            content = split.strip()
            if not content: continue
            uid = hashlib.sha1(f"{unique_id}::{i}::{content[:80]}".encode()).hexdigest()[:16]
            chunks.append({
                "id": uid,
                "content": content,
                "source_file": title if source_type == "attachment" else f"{space_key}/{title}.md",
                "source_title": title,
                "group": space_key,
                "heading": "",
                **DEFAULT_METADATA,
                "keywords": [],
                "global_context": global_summary,
                "parent_id": "",
                "chunk_type": "child"
            })
        return chunks

    log.info("Implementing Hierarchical (Parent-Child) Chunking...")
    try:
        chunker = LLMChunker(
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL,
            openai_model=OPENAI_MODEL,
            prompt=LLM_CHUNKING_PROMPT,
        )
        
        # 1. Split into Parent Segments
        parent_segments = [text[i:i + PARENT_CHUNK_SIZE] for i in range(0, len(text), PARENT_CHUNK_SIZE)]
        all_final_chunks = []
        
        for i, p_text in enumerate(parent_segments):
            # Create Parent Chunk
            p_uid = hashlib.sha1(f"{unique_id}::parent::{i}".encode()).hexdigest()[:16]
            parent_chunk = {
                "id": p_uid,
                "content": p_text,
                "source_file": title if source_type == "attachment" else f"{space_key}/{title}.md",
                "source_title": title,
                "group": space_key,
                "heading": "",
                **DEFAULT_METADATA,
                "keywords": [],
                "global_context": global_summary,
                "parent_id": "",
                "chunk_type": "parent"
            }
            all_final_chunks.append(parent_chunk)
            
            # 2. Generate Child Chunks from this Parent
            child_splits = await chunker.chunk_text(p_text, global_summary=global_summary)
            
            for j, c_data in enumerate(child_splits):
                if isinstance(c_data, dict):
                    content = c_data.get("content", "").strip()
                    keywords = c_data.get("keywords", [])
                else:
                    content = str(c_data).strip()
                    keywords = []
                
                if not content: continue
                
                c_uid = hashlib.sha1(f"{p_uid}::child::{j}::{content[:80]}".encode()).hexdigest()[:16]
                all_final_chunks.append({
                    "id": c_uid,
                    "content": content,
                    "source_file": parent_chunk["source_file"],
                    "source_title": title,
                    "group": space_key,
                    "heading": "",
                    **DEFAULT_METADATA,
                    "keywords": keywords if keywords else DEFAULT_METADATA.get("keywords", ""),
                    "global_context": global_summary,
                    "parent_id": p_uid,
                    "chunk_type": "child"
                })
                
        log.info(f"Hierarchical chunking produced {len(all_final_chunks)} total chunks (parents + children)")
        return all_final_chunks

    except Exception as e:
        log.warning(f"Hierarchical chunking failed: {e}, falling back to basic splitting")
        splits = text.split("\n\n")
        chunks = []
        for i, split in enumerate(splits):
            content = split.strip()
            if not content: continue
            uid = hashlib.sha1(f"{unique_id}::{i}::{content[:80]}".encode()).hexdigest()[:16]
            chunks.append({
                "id": uid,
                "content": content,
                "source_file": title if source_type == "attachment" else f"{space_key}/{title}.md",
                "source_title": title,
                "group": space_key,
                "heading": "",
                **DEFAULT_METADATA,
                "keywords": [],
                "global_context": global_summary,
                "parent_id": "",
                "chunk_type": "child"
            })
        return chunks

async def process_page_attachments(p, c_client, m_client, e_client, s_key, sem, chunker, stats):
    """Helper to process all attachments of a page concurrently."""
    p_id = p.get("id")
    attachments = await c_client.get_all_attachments_paginated(p_id)
    if not attachments:
        return
        
    page_title = p.get("title", "Untitled")
    log.info(f"Processing {len(attachments)} attachments for page {page_title}...")
    
    async def process_att(att):
        att_filename = att.get("title", "")
        att_download_path = att.get("_links", {}).get("download")
        if not att_filename or not att_download_path:
            return
            
        async with sem:
            try:
                att_bytes = await c_client.download_attachment(att_download_path)
                from confluence_mcp.extractor import FileExtractor
                att_text = FileExtractor.extract(att_filename, att_bytes)
                if not att_text or len(att_text.strip()) < 10:
                    return
                
                global_summary = await chunker.generate_document_summary(att_text)
                att_chunks = await chunk_text(
                    att_text,
                    f"{p_id}_{att_filename}",
                    att_filename,
                    s_key,
                    global_summary=global_summary,
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

    await asyncio.gather(*[process_att(att) for att in attachments])

async def sync_single_space(c_client, m_client, e_client, s_key, sem, state_manager, stats):
    try:
        log.info(f"--- Syncing attachments in space: {s_key} ---")
        pages = await c_client.get_all_pages_paginated(s_key)
        log.info(f"Found {len(pages)} pages in {s_key} to check for attachments.")
        
        chunker = LLMChunker(
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL,
            openai_model=OPENAI_MODEL,
            prompt=LLM_CHUNKING_PROMPT,
        )
        
        # Process pages concurrently in batches to avoid overwhelming the system
        batch_size = 5
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            tasks = []
            for p in batch:
                p_id = p.get("id")
                if not state_manager.is_synced(p_id):
                    tasks.append(process_page_attachments(p, c_client, m_client, e_client, s_key, sem, chunker, stats))
            
            if tasks:
                await asyncio.gather(*tasks)
                # Mark all pages in this batch as synced
                for p in batch:
                    state_manager.mark_synced(p.get("id"))
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