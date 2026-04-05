"""FastMCP server for Confluence."""

import os
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

mcp = FastMCP(
    "Confluence",
    instructions="MCP server for Confluence Server/Data Center. List spaces, get pages as markdown, and crawl entire spaces.",
)


def _get_confluence_client() -> ConfluenceClient:
    """Create a Confluence client from environment variables."""
    base_url = os.getenv("CONFLUENCE_URL")
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    verify_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"

    if not base_url or not username or not api_token:
        raise ValueError(
            "Missing Confluence configuration. "
            "Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN in your .env file."
        )

    return ConfluenceClient(
        base_url=base_url,
        username=username,
        api_token=api_token,
        verify_ssl=verify_ssl,
    )


@mcp.tool()
async def list_spaces(limit: int = 25, start: int = 0) -> list[dict[str, Any]]:
    """List all Confluence spaces accessible to the authenticated user.

    Args:
        limit: Maximum number of spaces to return (default 25).
        start: Offset for pagination (default 0).

    Returns:
        List of space objects with key, name, description, and type.
    """
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
    """Get a single Confluence page as Markdown by its page ID.

    Args:
        page_id: The Confluence page ID (numeric).
        include_attachments: Whether to extract attachment content (default True).

    Returns:
        Dictionary with title, content (markdown), space_key, page_id, path,
        and attachments with extracted content.
    """
    client = _get_confluence_client()
    try:
        page = await client.get_page(page_id)

        space_key = ""
        if "space" in page:
            space_key = page["space"].get("key", "")

        content = convert_page_to_markdown(page, space_key)
        path = generate_page_path(page, space_key)

        attachments_info = []
        attachments = page.get("children", {}).get("attachment", {}).get("results", [])

        if attachments and include_attachments:
            for att in attachments:
                att_filename = att.get("title", "")
                media_type = att.get("extensions", {}).get("mediaType", "")
                file_size = att.get("extensions", {}).get("fileSize", 0)

                if _is_binary_type(media_type):
                    attachments_info.append({
                        "filename": att_filename,
                        "content": f"[Binary file skipped: {media_type}]",
                        "media_type": media_type,
                        "size": file_size,
                    })
                    continue

                try:
                    download_path = f"/download/attachments/{page_id}/{att_filename}"
                    att_bytes = await client.download_attachment(download_path)
                    extracted = FileExtractor.extract(att_filename, att_bytes)

                    attachments_info.append({
                        "filename": att_filename,
                        "content": extracted,
                        "media_type": media_type,
                        "size": file_size,
                    })
                except Exception as e:
                    attachments_info.append({
                        "filename": att_filename,
                        "content": f"[Error: {str(e)}]",
                        "media_type": media_type,
                        "size": file_size,
                    })
        else:
            for att in attachments:
                attachments_info.append({
                    "filename": att.get("title", ""),
                    "media_type": att.get("extensions", {}).get("mediaType", ""),
                    "size": att.get("extensions", {}).get("fileSize", 0),
                })

        return {
            "page_id": page.get("id", ""),
            "title": page.get("title", ""),
            "space_key": space_key,
            "path": path,
            "content": content,
            "attachments": attachments_info,
        }
    finally:
        await client.close()


@mcp.tool()
async def crawl_space(
    space_key: str,
    include_attachments: bool = True,
    skip_binary: bool = False,
) -> list[dict[str, Any]]:
    """Crawl an entire Confluence space and convert all pages to Markdown.

    Fetches all pages (including child pages) in the specified space,
    extracts content from attachments, and returns the complete list
    of page contents with attachment content.

    Args:
        space_key: The Confluence space key (e.g. 'ENG', 'DOC').
        include_attachments: Whether to extract attachment content (default True).
        skip_binary: If True, skip binary files (images, videos, archives).
            If False, binary files return a placeholder message.

    Returns:
        List of page objects, each containing:
        - page_id: Confluence page ID
        - title: Page title
        - space_key: Space key
        - path: Relative file path
        - content: Full Markdown content of the page
        - attachments: List of attachment objects with extracted content:
            - filename: Attachment filename
            - content: Extracted Markdown content
            - media_type: MIME type
            - size: File size in bytes
    """
    client = _get_confluence_client()
    results: list[dict[str, Any]] = []

    try:
        all_pages = await client.get_all_pages_paginated(space_key)

        if not all_pages:
            return []

        for page in all_pages:
            page_id = page.get("id", "")

            try:
                full_page = await client.get_page(page_id)
            except Exception:
                full_page = page

            content = convert_page_to_markdown(full_page, space_key)
            path = generate_page_path(full_page, space_key)

            attachments_info: list[dict[str, Any]] = []
            attachments = full_page.get("children", {}).get("attachment", {}).get("results", [])

            if attachments and include_attachments:
                for att in attachments:
                    att_filename = att.get("title", "")
                    media_type = att.get("extensions", {}).get("mediaType", "")
                    file_size = att.get("extensions", {}).get("fileSize", 0)

                    if skip_binary and _is_binary_type(media_type):
                        attachments_info.append({
                            "filename": att_filename,
                            "content": f"[Binary file skipped: {media_type}]",
                            "media_type": media_type,
                            "size": file_size,
                        })
                        continue

                    try:
                        download_path = f"/download/attachments/{page_id}/{att_filename}"
                        att_bytes = await client.download_attachment(download_path)
                        extracted = FileExtractor.extract(att_filename, att_bytes)

                        attachments_info.append({
                            "filename": att_filename,
                            "content": extracted,
                            "media_type": media_type,
                            "size": file_size,
                        })
                    except Exception as e:
                        attachments_info.append({
                            "filename": att_filename,
                            "content": f"[Error: {str(e)}]",
                            "media_type": media_type,
                            "size": file_size,
                        })

            results.append({
                "page_id": page_id,
                "title": full_page.get("title", ""),
                "space_key": space_key,
                "path": path,
                "content": content,
                "attachments": attachments_info,
            })

        return results

    finally:
        await client.close()


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
        "application/x-executable",
        "application/x-dosexec",
        "application/x-iso",
        "application/octet-stream",
        "font/",
    )
    return any(media_type.startswith(t) for t in binary_types)


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
