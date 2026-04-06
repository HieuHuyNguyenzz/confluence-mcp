"""FastMCP server for Confluence."""

import os
import asyncio
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from confluence_mcp.chunker import RAGChunker
from confluence_mcp.client import ConfluenceClient
from confluence_mcp.converter import (
    convert_page_to_markdown,
    generate_page_path,
)
from confluence_mcp.embedder import EmbeddingClient
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
        "application/octet-stream",
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
            return None

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
        # Use the helper to ensure consistency and support for attachments
        page_summary = {"id": page_id}
        return await _process_page(client, page_summary, "", include_attachments, semaphore)
    except Exception as e:
        # In case of failure, we can't use _process_page easily because space_key might be missing
        # Let's just handle the error or refine the helper.
        # Actually, let's just call the helper. The space_key is filled inside _process_page if it's available.
        # Wait, _process_page uses space_key for converter and path.
        # For a single page, we should fetch the page first to get the space_key.
        
        # Redefining logic for single page to be safe
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


def _get_chunker() -> RAGChunker:
    """Create a RAG chunker from environment variables."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "800"))

    if not api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in your .env file.")

    return RAGChunker(base_url=base_url, api_key=api_key, model=model, max_tokens=max_tokens)


@mcp.tool()
async def chunk_space_for_rag(
    space_key: str,
    include_attachments: bool = True,
) -> list[dict[str, Any]]:
    """Crawl a Confluence space and chunk all pages for RAG embedding."""
    client = _get_confluence_client()
    chunker = _get_chunker()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    try:
        all_pages = await client.get_all_pages_paginated(space_key)
        if not all_pages:
            return []

        tasks = [
            _process_page(client, page, space_key, include_attachments, semaphore)
            for page in all_pages
        ]
        pages_data = await asyncio.gather(*tasks)

        results = []
        for page in pages_data:
            page_result = chunker.chunk_page(
                content=page["content"],
                page_id=page["page_id"],
                title=page["title"],
                space_key=page["space_key"],
                path=page["path"],
                created_date=page["created_date"],
                last_modified=page["last_modified"],
                attachments=page["attachments"] if include_attachments else None,
            )
            results.append(page_result)
        return results
    finally:
        await client.close()


def _get_embedder() -> EmbeddingClient:
    """Create an embedding client from environment variables."""
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080")
    endpoint = os.getenv("EMBEDDING_ENDPOINT", "/embed")
    api_key = os.getenv("EMBEDDING_API_KEY", "")
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    return EmbeddingClient(base_url=base_url, endpoint=endpoint, api_key=api_key, timeout=timeout), batch_size


@mcp.tool()
async def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Embed a list of chunks using an external embedding API."""
    embedder, default_batch = _get_embedder()
    batch = batch_size if batch_size > 0 else default_batch
    total = len(chunks)

    try:
        for i in range(0, total, batch):
            batch_chunks = chunks[i:i + batch]
            texts = [c.get("content", "") for c in batch_chunks]
            embeddings = await embedder.embed_batch(texts)
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk["embedding"] = embedding
        return chunks
    finally:
        await embedder.close()


@mcp.tool()
async def chunk_and_embed_space(
    space_key: str,
    include_attachments: bool = True,
    embedding_batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Crawl a Confluence space, chunk pages, and embed all chunks for RAG."""
    client = _get_confluence_client()
    chunker = _get_chunker()
    embedder, batch = _get_embedder()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    try:
        all_pages = await client.get_all_pages_paginated(space_key)
        if not all_pages:
            return []

        tasks = [
            _process_page(client, page, space_key, include_attachments, semaphore)
            for page in all_pages
        ]
        pages_data = await asyncio.gather(*tasks)

        all_chunks = []
        page_map = {}
        for page in pages_data:
            page_result = chunker.chunk_page(
                content=page["content"],
                page_id=page["page_id"],
                title=page["title"],
                space_key=page["space_key"],
                path=page["path"],
                created_date=page["created_date"],
                last_modified=page["last_modified"],
                attachments=page["attachments"] if include_attachments else None,
            )
            page_map[page["page_id"]] = page_result
            all_chunks.extend(page_result["chunks"])

        if all_chunks:
            texts = [c["content"] for c in all_chunks]
            embeddings = await embedder.embed_batch(texts)
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk["embedding"] = embedding

        return list(page_map.values())
    finally:
        await client.close()
        await embedder.close()


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
