"""Sync Confluence data to Milvus Vector Database."""

import os
import asyncio
import json
import hashlib
import logging
from typing import Any, List, Dict
from dotenv import load_dotenv

from confluence_mcp.client import ConfluenceClient
from confluence_mcp.knowledge.milvus_client import MilvusClient
from confluence_mcp.knowledge.embedding_client import EmbeddingClient
from confluence_mcp.knowledge.llm_chunker import LLMChunker

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
EMBEDDING_URL = os.getenv("EMBEDDING_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_SEMAPHORE = asyncio.Semaphore(5)
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

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

# ─────────────────────────── PIPELINE ────────────────────────────

async def chunk_text(
    text: str,
    unique_id: str,
    title: str,
    space_key: str,
    global_summary: str = "",
    source_type: str = "page",
    use_llm: bool = True,
    chunker: LLMChunker = None,
) -> List[Dict[str, Any]]:
    PARENT_CHUNK_SIZE = 2000
    
    if not use_llm or not OPENAI_API_KEY or not chunker:
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
        parent_segments = [text[i:i + PARENT_CHUNK_SIZE] for i in range(0, len(text), PARENT_CHUNK_SIZE)]
        all_final_chunks = []
        
        for i, p_text in enumerate(parent_segments):
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
            
            # Process child chunks with semaphore
            async with LLM_SEMAPHORE:
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
        
        log.info(f"Hierarchical chunking produced {len(all_final_chunks)} total chunks")
        return all_final_chunks
    except Exception as e:
        log.warning(f"Hierarchical chunking failed: {e}, falling back to basic splitting")
        return await chunk_text(text, unique_id, title, space_key, global_summary, source_type, use_llm=False)

async def process_page_attachments(p, c_client, m_client, e_client, s_key, sem, chunker, stats):
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
                    source_type="attachment",
                    chunker=chunker
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
        )
        
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
