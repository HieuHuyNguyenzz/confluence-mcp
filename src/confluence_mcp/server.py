"""FastMCP server for Confluence."""

import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from confluence_mcp.chunker import RAGChunker
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
            Only text-based files are extracted (PDF, DOCX, XLSX, CSV, etc.).
            Binary files (images, videos, archives) are skipped.

    Returns:
        Dictionary with title, content (markdown), space_key, page_id, path,
        created_date, last_modified, created_by, last_modified_by,
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

        version = page.get("version", {})
        history = page.get("history", {})
        created_by = history.get("createdBy", {})
        last_modified_by = history.get("lastUpdated", {}).get("by", {})

        attachments_info = []
        attachments = page.get("children", {}).get("attachment", {}).get("results", [])

        if attachments and include_attachments:
            for att in attachments:
                att_filename = att.get("title", "")
                media_type = att.get("extensions", {}).get("mediaType", "")
                file_size = att.get("extensions", {}).get("fileSize", 0)

                if _is_binary_type(media_type):
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

        return {
            "page_id": page.get("id", ""),
            "title": page.get("title", ""),
            "space_key": space_key,
            "path": path,
            "content": content,
            "created_date": history.get("createdDate", ""),
            "last_modified": version.get("when", ""),
            "created_by": created_by.get("displayName", created_by.get("username", "")),
            "last_modified_by": last_modified_by.get("displayName", last_modified_by.get("username", "")),
            "attachments": attachments_info,
        }
    finally:
        await client.close()


@mcp.tool()
async def crawl_space(
    space_key: str,
    include_attachments: bool = True,
) -> list[dict[str, Any]]:
    """Crawl an entire Confluence space and convert all pages to Markdown.

    Fetches all pages (including child pages) in the specified space,
    extracts content from text-based attachments, and returns the complete list
    of page contents with attachment content.

    Binary files (images, videos, archives, executables) are automatically skipped.
    Only text-based files are extracted: PDF, DOCX, XLSX, PPTX, CSV, JSON, XML,
    HTML, TXT, and source code files.

    Args:
        space_key: The Confluence space key (e.g. 'ENG', 'DOC').
        include_attachments: Whether to extract attachment content (default True).

    Returns:
        List of page objects, each containing:
        - page_id: Confluence page ID
        - title: Page title
        - space_key: Space key
        - path: Relative file path
        - content: Full Markdown content of the page
        - created_date: ISO 8601 timestamp when page was created
        - last_modified: ISO 8601 timestamp when page was last modified
        - created_by: Display name of page creator
        - last_modified_by: Display name of last modifier
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

                    if _is_binary_type(media_type):
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

            version = full_page.get("version", {})
            history = full_page.get("history", {})
            created_by = history.get("createdBy", {})
            last_modified_by = history.get("lastUpdated", {}).get("by", {})

            results.append({
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
            })

        return results

    finally:
        await client.close()


def _get_chunker() -> RAGChunker:
    """Create a RAG chunker from environment variables."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "800"))

    if not api_key:
        raise ValueError(
            "Missing OpenAI API key. "
            "Set OPENAI_API_KEY in your .env file."
        )

    return RAGChunker(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
    )


@mcp.tool()
async def chunk_space_for_rag(
    space_key: str,
    include_attachments: bool = True,
) -> list[dict[str, Any]]:
    """Crawl a Confluence space and chunk all pages for RAG embedding.

    Uses an LLM (OpenAI-compatible API) to intelligently split page content
    into semantically coherent chunks suitable for vector embedding.

    Each page is crawled, converted to markdown, then split by the LLM
    into self-contained chunks with heading context preserved.
    Attachments are also chunked if include_attachments is True.

    Args:
        space_key: The Confluence space key (e.g. 'ENG', 'DOC').
        include_attachments: Whether to extract and chunk attachments (default True).

    Returns:
        List of page chunk results, each containing:
        - page_id: Confluence page ID
        - title: Page title
        - space_key: Space key
        - path: Relative file path
        - total_chunks: Total number of chunks for this page (including attachments)
        - chunks: List of chunk objects:
            - chunk_id: Unique ID (format: page_id-index or page_id-att-filename-index)
            - page_id: Parent page ID
            - title: Parent page title
            - space_key: Space key
            - path: Parent page path
            - content: Chunk markdown content
            - heading_path: Heading breadcrumb (H1 > H2 > H3)
            - created_date: Page creation date (ISO 8601)
            - last_modified: Page last modified date (ISO 8601)
            - source_file: Attachment filename (only for attachment chunks)
            - source_media_type: Attachment MIME type (only for attachment chunks)
    """
    client = _get_confluence_client()
    chunker = _get_chunker()
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

                    if _is_binary_type(media_type):
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
                    except Exception:
                        pass

            history = full_page.get("history", {})
            version = full_page.get("version", {})
            created_date = history.get("createdDate", "")
            last_modified = version.get("when", "")

            page_result = chunker.chunk_page(
                content=content,
                page_id=page_id,
                title=full_page.get("title", ""),
                space_key=space_key,
                path=path,
                created_date=created_date,
                last_modified=last_modified,
                attachments=attachments_info if include_attachments else None,
            )

            results.append(page_result)

        return results

    finally:
        await client.close()


def _is_binary_type(media_type: str) -> bool:
    """Check if a media type is binary (not extractable as text).
    
    Binary files are skipped - only text-based files are extracted.
    """
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
        "application/octet-stream",
        "font/",
        "application/x-font",
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
