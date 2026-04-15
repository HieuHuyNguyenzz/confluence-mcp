"""Thống kê các loại file đính kèm trong Confluence space."""

import os
import asyncio
from collections import Counter
from dotenv import load_dotenv
from confluence_mcp.client import ConfluenceClient

load_dotenv()

MAX_CONCURRENT = 10

async def stats_attachments(space_key: str) -> dict:
    base_url = os.getenv("CONFLUENCE_URL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    verify_ssl = os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true"

    if not base_url or not api_token:
        print("Error: Missing CONFLUENCE_URL or CONFLUENCE_API_TOKEN")
        return {}

    client = ConfluenceClient(base_url=base_url, api_token=api_token, verify_ssl=verify_ssl)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        print(f"Fetching pages in space: {space_key}...")
        pages = await client.get_all_pages_paginated(space_key)
        print(f"Found {len(pages)} pages\n")

        extensions = Counter()
        total_attachments = 0

        async def process_page(page: dict):
            nonlocal total_attachments
            async with semaphore:
                page_id = page.get("id")
                page_title = page.get("title", "Untitled")
                try:
                    attachments = await client.get_all_attachments_paginated(page_id)
                    for att in attachments:
                        filename = att.get("title", "")
                        if "." in filename:
                            ext = filename.rsplit(".", 1)[-1].lower()
                        else:
                            ext = "no_extension"
                        extensions[ext] += 1
                        total_attachments += 1
                except Exception as e:
                    print(f"Error getting attachments for page '{page_title}': {e}")

        tasks = [process_page(p) for p in pages]
        await asyncio.gather(*tasks)

        return {
            "space_key": space_key,
            "total_pages": len(pages),
            "total_attachments": total_attachments,
            "extensions": dict(extensions),
        }

    finally:
        await client.close()


def print_stats(stats: dict):
    print("=" * 50)
    print(f"Space: {stats['space_key']}")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Total attachments: {stats['total_attachments']}")
    print("-" * 50)
    print("File types:")
    print("-" * 50)
    
    exts = stats["extensions"]
    if not exts:
        print("No attachments found.")
        return

    sorted_exts = sorted(exts.items(), key=lambda x: x[1], reverse=True)
    max_ext_len = max(len(ext) for ext, _ in sorted_exts)
    max_count_len = max(len(str(count)) for _, count in sorted_exts)

    for ext, count in sorted_exts:
        ext_padded = ext.ljust(max_ext_len)
        count_str = str(count).ljust(max_count_len)
        pct = (count / stats["total_attachments"]) * 100
        bar = "█" * int(pct / 5)
        print(f"  .{ext_padded} | {count_str} ({pct:5.1f}%) {bar}")

    print("=" * 50)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Thống kê file đính kèm trong Confluence space")
    parser.add_argument("space_key", help="Confluence space key (e.g., TECH, DOC)")
    args = parser.parse_args()

    stats = await stats_attachments(args.space_key)
    if stats:
        print_stats(stats)


if __name__ == "__main__":
    asyncio.run(main())