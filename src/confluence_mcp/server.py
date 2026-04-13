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
from confluence_mcp.processor import process_page, process_attachment, is_binary_type

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


@mcp.tool()
async def convert_file_to_markdown(file_path: str) -> str:
    """Convert a local file (PDF, DOCX, XLSX, etc.) to Markdown.
    
    Args:
        file_path: The absolute path to the file to convert.
    """
    try:
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        return FileExtractor.extract(filename, file_bytes)
    except FileNotFoundError:
        return f"[Error: File not found at {file_path}]"
    except Exception as e:
        return f"[Error converting file: {type(e).__name__}: {e}]"


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
        page = await client.get_page(page_id)
        space_key = page.get("space", {}).get("key", "")
        return await process_page(client, page, space_key, include_attachments, semaphore)
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
        pages = await client.get_all_pages_paginated(space_key)
        if not pages:
            return []

        tasks = [
            process_page(client, p, space_key, include_attachments, semaphore)
            for p in pages
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
