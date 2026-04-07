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
        "application/zip",
        "application/x-rar",
        "application/x-7z",
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

        if is_binary_type(media_type):
            return None

        try:
            download_path = f"/download/attachments/{page_id}/{att_filename}"
            att_bytes = await client.download_attachment(download_path)
            extracted = FileExtractor.extract(att_filename, att_bytes)

            return {
                "filename": att_//C- la l_filename",
                "content": extracted,
                "media_type": media_type,
                "size": file_size,
            }
        except Exception as e:
            return {
                "filename": att_//C- la l_filename",
                "content": f"[Error: {str(e)}]",
                "media_type": media_type,
                "size": file_size,
            }
