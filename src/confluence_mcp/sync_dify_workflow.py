"""Sync Confluence attachments to Dify Workflow."""

import os
import asyncio
import tempfile
from typing import Any, List, Dict
import httpx
from dotenv import load_dotenv
from confluence_mcp.client import ConfluenceClient
from confluence_mcp.extractor import FileExtractor

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

    async def upload_file(self, file_source: bytes | str, filename: str) -> str:
        """Uploads a file to Dify and returns the file_id.
        file_source can be bytes or a path to a file.
        """
        endpoint = "/files/upload"
        
        if isinstance(file_source, str):
            # file_source is a path
            with open(file_source, "rb") as f:
                files = {"file": (filename, f)}
                try:
                    resp = await self._client.post(endpoint, files=files)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("id")
                except httpx.HTTPStatusError as e:
                    print(f"Dify Upload error: {e.response.status_code} - {e.response.text}")
                    raise
        else:
            # file_source is bytes
            files = {"file": (filename, file_source)}
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
                "file": {
                    "type": "document",
                    "transfer_method": "local_file",
                    "upload_file_id": file_id
                }
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

async def sync_single_space(c_client, d_client, s_key, sem, stats):
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
                    temp_file = None
                    temp_md_file = None
                    try:
                        # 1. Download from Confluence to temp file
                        fd, temp_file = tempfile.mkstemp()
                        os.close(fd) # Close fd, we'll use the path
                        
                        await c_client.download_attachment(download_path, save_path=temp_file)
                        
                        # Convert .doc to Markdown if necessary
                        current_filename = filename
                        current_source = temp_file
                        
                        if filename.lower().endswith(".doc"):
                            print(f"Converting {filename} to markdown...")
                            with open(temp_file, "rb") as f:
                                file_bytes = f.read()
                            extracted_text = FileExtractor.extract(filename, file_bytes)
                            
                            current_filename = os.path.splitext(filename)[0] + ".md"
                            fd_md, temp_md_file = tempfile.mkstemp(suffix=".md")
                            os.close(fd_md)
                            with open(temp_md_file, "w", encoding="utf-8") as f:
                                f.write(extracted_text)
                            
                            current_source = temp_md_file
                            print(f"Converted to {current_filename}")
                        
                        # 2. Upload to Dify using streaming path
                        file_id = await d_client.upload_file(current_source, current_filename)
                        
                        # 3. Run Workflow
                        await d_client.run_workflow(current_filename, file_id)
                        stats["success"] += 1
                        print(f"Successfully processed file: {current_filename}")
                        
                        # Avoid overloading
                        delay = float(os.getenv("SYNC_DELAY", "120"))
                        await asyncio.sleep(delay)
                        
                    except Exception as e:
                        stats["failure"] += 1
                        ext = os.path.splitext(filename)[1]
                        stats["failed_formats"].add(ext)
                        print(f"Error processing file {filename} on page {p_id}: {e}")
                    finally:
                        if temp_file and os.path.exists(temp_file):
                            os.remove(temp_file)
                        if temp_md_file and os.path.exists(temp_md_file):
                            os.remove(temp_md_file)
                        
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
    
    stats = {"success": 0, "failure": 0, "failed_formats": set()}
    
    if not all([c_url, c_tok, d_key]):
        print("Error: Missing required env vars (CONFLUENCE_URL, CONFLUENCE_API_TOKEN, DIFY_WORKFLOW_API_KEY)")
        return
        
    c_client = ConfluenceClient(base_url=c_url, api_token=c_tok, verify_ssl=v_ssl)
    d_client = DifyWorkflowClient(base_url=d_url, api_key=d_key, user_id=d_user)
    # Use a configurable semaphore to avoid overloading Dify/LLM
    concurrency = int(os.getenv("SYNC_CONCURRENCY", "1"))
    sem = asyncio.Semaphore(concurrency)
    
    try:
        if t_s_key:
            print(f"Targeted sync mode: space {t_s_key}")
            await sync_single_space(c_client, d_client, t_s_key, sem, stats)
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
                    await sync_single_space(c_client, d_client, sk, sem, stats)
        
        summary = []
        summary.append("\n--- Sync Summary ---")
        summary.append(f"Successfully processed: {stats['success']} files")
        summary.append(f"Failed: {stats['failure']} files")
        if stats['failed_formats']:
            summary.append(f"Failed formats: {', '.join(stats['failed_formats']) if stats['failed_formats'] else 'None'}")
        summary.append("-------------------\n")
        summary.append("\nAll attachment sync tasks completed successfully.")
        
        summary_text = "\n".join(summary)
        print(summary_text)
        
        with open("sync_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)
        print("\nSummary saved to sync_summary.txt")
    finally:
        await c_client.close()
        await d_client.close()

if __name__ == "__main__":
    asyncio.run(main())
