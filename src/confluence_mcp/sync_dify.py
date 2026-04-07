"""Sync Confluence data to Dify Knowledge Base."""

import os
import asyncio
from typing import Any, List
import httpx
from dotenv import load_dotenv
from confluence_mcp.client import ConfluenceClient
from confluence_mcp.processor import process_page, process_attachment
from confluence_mcp.converter import convert_page_to_markdown, generate_page_path
from confluence_mcp.extractor import FileExtractor

load_dotenv()

class DifyClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0, connect=20.0),
        )

    async def close(self):
        await self._client.aclose()

    async def create_document_from_text(self, dataset_id: str, txt: str, title: str):
        endpoint = f"/datasets/{dataset_id}/document/create_by_text"
        payload = {
            "name": title,
            "text": txt,
            "indexing_technique": "high_quality",
            "process_mode": "automatic",
        }
        resp = await self._client.post(endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()

async def sync_single_space(c_client, d_client, s_key, d_id, sem):
    try:
        print(f"--- Syncing space: {s_key} ---")
        pages = await c_client.get_all_pages_paginated(s_key)
        print(f"Found {len(pages)} pages in {s_key}.")
        for p in pages:
            p_data = await process_page(c_client, p, s_key, True, sem)
            t = p_data["title"]
            c_txt = p_data["content"]
            if p_data["attachments"]:
                att_txt = "\n\n### Attachments\n"
                for a in p_data["attachments"]:
                    att_txt += f"**{a['filename']}**:\n{a['content']}\n\n"
                c_txt += att_txt
            try:
                await d_client.create_document_from_text(d_id, c_txt, t)
                print(f"Synced page: {t}")
            except Exception as e:
                print(f"Failed to sync page {t}: {e}")
        print(f"Completed sync for space: {s_key}")
    except Exception as e:
        print(f"Critical error syncing space {s_key}: {e}")

async def main():
    c_url = os.getenv("CONFLUENCE_URL")
    c_tok = os.getenv("CONFLUENCE_API_TOKEN")
    v_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"
    d_url = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")
    d_key = os.getenv("DIFY_API_KEY")
    d_id = os.getenv("DIFY_DATASET_ID")
    t_s_key = os.getenv("SYNC_SPACE_KEY")
    if not all([c_url, c_tok, d_key, d_id]):
        print("Error: Missing required env vars")
        return
    c_client = ConfluenceClient(base_url=c_url, api_token=c_tok, verify_ssl=v_ssl)
    d_client = DifyClient(base_url=d_url, api_key=d_key)
    sem = asyncio.Semaphore(10)
    try:
        if t_s_key:
            print(f"Targeted sync mode: space {t_s_key}")
            await sync_single_space(c_client, d_client, t_s_key, d_id, sem)
        else:
            print("Full system sync mode: Crawling all spaces...")
            spaces = await c_client.list_spaces()
            if not spaces:
                print("No spaces found.")
                return
            print(f"Found {len(spaces)} spaces. Starting global sync...")
            for s in spaces:
                sk = s.get("key")
                if sk:
                    await sync_single_space(c_client, d_client, sk, d_id, sem)
        print("\nAll sync tasks completed successfully.")
    finally:
        await c_client.close()
        await d_client.close()

if __name__ == "__main__":
    asyncio.run(main())
