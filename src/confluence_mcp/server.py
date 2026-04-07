"""FastMCP server for Confluence."""

import os
import asyncio
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from confluence_mcp.client import ConfluenceClient
from confluence_mcp.converter import (
    convert_page_to_markdown,
    generate_page_path,
)
from confluence_mcp.extractor import FileExtractor

load_dotenv()

MAX_CONCURRENT_REQUESTS = 10

mcp = FastMCP(
    "Confluence",
    instructions="MCP server for Confluence Server/Data Center. List spaces, get pages as markdown, and crawl entire spaces.",
)


def _get_confluence_client() -> ConfluenceClient:
    """Create a Confluence client from environment variables."""
    base_url = os.getenv("CONFLUENCE_URL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    verify_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"

    if not base_url or not api_token:
        raise ValueError(
            "Missing Confluence configuration. "
            "Set CONFLUENCE_URL and CONFLUENCE_API_TOKEN in your .env file."
        )

    return ConfluenceClient(
        base_url=base_url,
        api_token=api_token,
        verify_ssl=verify_ssl,
    )


def _is_binary_type(media_type: str) -> bool:
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


async def _process_attachment(
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

        if _is_binary_type(media_type):
            print(f"Skipping binary attachment: {att_filename} ({media_type})")
            return None

        try:
            download_path = f"/download/attachments/{page_id}/{att_filename}"
            att_bytes = await client.download_attachment(download_path)
            extracted = FileExtractor.extract(att_filename, att_bytes)

            print(f"Successfully extracted attachment: {att_filename}")
            return {
                "filename": att_filename,
                "content": extracted,
                "media_type": media_type,
                "size": file_size,
            }
        except Exception as e:
            print(f"Error processing attachment {att_filename}: {str(e)}")
            return {
                "filename": att_filename,
                "content": f"[Error: {str(e)}]",
                "media_type": media_type,
                "size": file_size,
            }


async def _process_page(
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
        attachments = full_page.get("children", {}).get("attachment", {}).get("results", [])

        if attachments and include_attachments:
            att_tasks = [
                _process_attachment(client, page_id, att, semaphore)
                for att in attachments
            ]
            att_results = await asyncio.gather(*att_tasks)
            attachments_info = [r for r in att_results if r is not None]

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


@mcp.tool()
async def list_spaces(limit: int = 25, start: int = 0) -> list[dict[str, Any]]:
    """List all Confluence spaces accessible to the authenticated user."""
    client = _get_confluence_client()
    try:
        result = await client.list_spaces(limit=limit, start=start)
        spaces = []
        for space in result.get("results", []):
            desc = ""
            description = space.get("description", {})
            if isinstance(description, dict):
                desc = description.get("plain", {}).get("value", "")
            spaces.append({
                "key": space.get("key", ""),
                "name": space.get("name", ""),
                "description": desc,
                "type": space.get("type", ""),
                "status": space.get("status", ""),
            })
        return spaces
    finally:
        await client.close()


@mcp.tool()
async def get_page_as_markdown(page_id: str, include_attachments: bool = True) -> dict[str, Any]:
    """Get a single Confluence page as Markdown by its page ID."""
    client = _get_confluence_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    try:
        page_summary = {"id": page_id}
        # We first need to get the page to find its space_key for conversion
        page = await client.get_page(page_id)
        space_key = page.get("space", {}).get("key", "")
        return await _process_page(client, page, space_key, include_attachments, semaphore)
    finally:
        await client.close()


@mcp.tool()
async def crawl_space(
    space_key: str,
    include_attachments: bool = True,
) -> list[dict[str, Any]]:
    """Crawl an entire Confluence space and convert all pages to Markdown."""
    client = _get_confluence_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    try:
        all_pages = await client.get_all_pages_paginated(space_key)
        if not all_pages:
            return []

        tasks = [
            _process_page(client, page, space_key, include_attachments, semaphore)
            for page in all_pages
        ]
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confluence MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
