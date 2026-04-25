"""Common processing logic for Confluence pages and attachments."""

import asyncio
from typing import Any
from confluence_mcp.client import ConfluenceClient
from confluence_mcp.converter import convert_page_to_markdown, generate_page_path
from confluence_mcp.extractor import FileExtractor

def is_binary_type(media_type: str) -> bool:
    """Check if a media type is binary (not extractable as text)."""
    binary_types = (
        "image/",
        "video/",
        "audio/",
        "application/x-tar",
        "application/gzip",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-executable",
        "application/x-dosexec",
        "application/x-iso",
        "application/x-msdownload",
        "application/x-mach-binary",
        "font/",
        "application/x-font",
    )
    return any(media_type.startswith(t) for t in binary_types)

async def process_attachment(
    client: ConfluenceClient,
    page_id: str,
    att: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Download and extract a single attachment."""
    async with semaphore:
        att_filename = att.get("title", "")
        media_type = att.get("extensions", {}).get("mediaType", "")
        file_size = att.get("extensions", {}).get("fileSize", 0)

        try:
            download_path = f"/download/attachments/{page_id}/{att_filename}"
            att_bytes = await client.download_attachment(download_path)
            extracted = FileExtractor.extract(att_filename, att_bytes)

            return {
                "filename": att_filename,
                "content": extracted,
                "media_type": media_type,
                "size": file_size,
            }
        except Exception as e:
            return {
                "filename": att_filename,
                "content": f"[Error: {str(e)}]",
                "media_type": media_type,
                "size": file_size,
            }

async def process_page(
    client: ConfluenceClient,
    page_summary: dict[str, Any],
    space_key: str,
    include_attachments: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Fetch full page content and its attachments."""
    async with semaphore:
        page_id = page_summary.get("id", "")
        try:
            full_page = await client.get_page(page_id)
        except Exception:
            full_page = page_summary

        content = convert_page_to_markdown(full_page, space_key)
        path = generate_page_path(full_page, space_key)

        attachments_info = []
        if include_attachments:
            attachments = await client.get_all_attachments_paginated(page_id)
            if attachments:
                for att in attachments:
                    res = await process_attachment(client, page_id, att, semaphore)
                    if res is not None:
                        attachments_info.append(res)

        version = full_page.get("version", {})
        history = full_page.get("history", {})
        created_by = history.get("createdBy", {})
        last_modified_by = history.get("lastUpdated", {}).get("by", {})

        return {
            "page_id": page_id,
            "title": full_page.get("title", ""),
            "space_key": space_key,
            "path": path,
            "content": content,
            "created_date": history.get("createdDate", ""),
            "last_modified": version.get("when", ""),
            "created_by": created_by.get("displayName", created_by.get("username", "")),
            "last_modified_by": last_modified_by.get("displayName", last_modified_by.get("username", "")),
            "attachments": attachments_info,
        }
