"""Sync Confluence attachments to Dify Workflow."""

import os
import asyncio
from typing import Any, List, Dict
import httpx
from dotenv import load_dotenv
from confluence_mcp.client import ConfluenceClient

load_dotenv()

class DifyWorkflowClient:
    def __init__(self, base_url: str, api_key: str, user_id: str = "confluence_sync_bot"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300.0, connect=20.0), # Longer timeout for file uploads
        )

    async def close(self):
        await self._client.aclose()

    async def upload_file(self, file_bytes: bytes, filename: str) -> str:
        """Uploads a file to Dify and returns the file_id."""
        endpoint = "/files/upload"
        files = {"file": (filename, file_bytes)}
        try:
            resp = await self._client.post(endpoint, files=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("id")
        except httpx.HTTPStatusError as e:
            print(f"Dify Upload error: {e.response.status_code} - {e.response.text}")
            raise

    async def run_workflow(self, filename: str, file_id: str):
        """Triggers the Dify Workflow with the uploaded file."""
        endpoint = "/workflows/run"
        # Dify Workflow file inputs usually expect an object with upload_file_id
        payload = {
            "inputs": {
                "filename": filename,
                "file": [
                    {
                        "transfer_method": "local_file",
                        "upload_file_id": file_id
                    }
                ]
            },
            "response_mode": "blocking",
            "user": self.user_id
        }
        try:
            resp = await self._client.post(endpoint, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"Dify Workflow error: {e.response.status_code} - {e.response.text}")
            raise

async def sync_single_space(c_client, d_client, s_key, sem):
    try:
        print(f"--- Processing attachments in space: {s_key} ---")
        pages = await c_client.get_all_pages_paginated(s_key)
        print(f"Found {len(pages)} pages in {s_key}.")
        
        for p in pages:
            p_id = p.get("id")
            # Get attachments for this page
            result = await c_client.get_page_attachments(p_id)
            attachments = result.get("results", [])
            if not attachments:
                continue
                
            print(f"Page {p_id} has {len(attachments)} attachments. Processing...")
            
            for att in attachments:
                filename = att.get("title") or att.get("filename")
                download_path = att.get("_links", {}).get("download")
                # We only want to process if we have a filename and a download path
                if not filename or not download_path:
                    continue
                
                async with sem:
                    try:
                        # 1. Download from Confluence
                        file_bytes = await c_client.download_attachment(download_path)
                        if not file_bytes:
                            print(f"Failed to download {filename}, skipping.")
                            continue
                            
                        # 2. Upload to Dify
                        file_id = await d_client.upload_file(file_bytes, filename)
                        
                        # 3. Run Workflow
                        await d_client.run_workflow(filename, file_id)
                        print(f"Successfully processed file: {filename}")
                        
                    except Exception as e:
                        print(f"Error processing file {filename} on page {p_id}: {e}")
                        
        print(f"Completed processing for space: {s_key}")
    except Exception as e:
        print(f"Critical error processing space {s_key}: {e}")

async def main():
    c_url = os.getenv("CONFLUENCE_URL")
    c_tok = os.getenv("CONFLUENCE_API_TOKEN")
    v_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"
    d_url = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")
    d_key = os.getenv("DIFY_WORKFLOW_API_KEY")
    d_user = os.getenv("DIFY_USER_ID", "confluence_sync_bot")
    t_s_key = os.getenv("SYNC_SPACE_KEY")
    
    if not all([c_url, c_tok, d_key]):
        print("Error: Missing required env vars (CONFLUENCE_URL, CONFLUENCE_API_TOKEN, DIFY_WORKFLOW_API_KEY)")
        return
        
    c_client = ConfluenceClient(base_url=c_url, api_token=c_tok, verify_ssl=v_ssl)
    d_client = DifyWorkflowClient(base_url=d_url, api_key=d_key, user_id=d_user)
    # Use a smaller semaphore for file uploads to avoid overloading Dify
    sem = asyncio.Semaphore(5)
    
    try:
        if t_s_key:
            print(f"Targeted sync mode: space {t_s_key}")
            await sync_single_space(c_client, d_client, t_s_key, sem)
        else:
            print("Full system sync mode: Crawling all spaces for attachments...")
            spaces = await c_client.list_spaces()
            if not spaces:
                print("No spaces found.")
                return
            print(f"Found {len(spaces)} spaces. Starting global sync...")
            for s in spaces:
                sk = s.get("key") if isinstance(s, dict) else s
                if sk:
                    await sync_single_space(c_client, d_client, sk, sem)
        print("\nAll attachment sync tasks completed successfully.")
    finally:
        await c_client.close()
        await d_client.close()

if __name__ == "__main__":
    asyncio.run(main())
