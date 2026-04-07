"""Confluence Server/Data Center REST API client."""

import base64
from typing import Any

import httpx


class ConfluenceClient:
    """Async client for Confluence Server/Data Center REST API."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = self._build_auth(api_token)
        self.verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _build_auth(api_token: str) -> str:
        return f"Bearer {api_token}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": self.auth,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                verify=self.verify_ssl,
                timeout=httpx.Timeout(300.0, connect=20.0),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        client = await self._get_client()
        try:
            resp = await client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"Unexpected error during request to {path}: {type(e).__name__}: {str(e)}")
            raise

    async def list_spaces(
        self, limit: int = 25, start: int = 0
    ) -> dict[str, Any]:
        """List all spaces accessible to the user."""
        return await self._get(
            "/rest/api/space",
            params={"limit": limit, "start": start, "expand": "description,metadata"},
        )

    async def get_all_pages_in_space(
        self,
        space_key: str,
        limit: int = 100,
        start: int = 0,
    ) -> dict[str, Any]:
        """Get all pages in a space using CQL search (includes hierarchy)."""
        cql = f"space={space_key} AND type=page"
        return await self._get(
            "/rest/api/content/search",
            params={
                "cql": cql,
                "limit": limit,
                "start": start,
                "expand": "ancestors,children.page,version",
            },
        )

    async def get_page(
        self,
        page_id: str,
        expand: str = "body.view,children.attachment,ancestors,version,history",
    ) -> dict[str, Any]:
        """Get a single page by ID with expanded content."""
        return await self._get(
            f"/rest/api/content/{page_id}",
            params={"expand": expand},
        )

    async def get_all_pages_paginated(
        self, space_key: str, batch_size: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch ALL pages in a space with automatic pagination."""
        all_pages: list[dict[str, Any]] = []
        start = 0

        while True:
            result = await self.get_all_pages_in_space(
                space_key=space_key,
                limit=batch_size,
                start=start,
            )
            results = result.get("results", [])
            all_pages.extend(results)

            size = result.get("size", 0)
            total = result.get("totalSize", 0)

            if start + size >= total:
                break
            start += size

        return all_pages

    async def download_attachment(
        self, download_path: str
    ) -> bytes:
        """Download an attachment by its Confluence download path."""
        client = await self._get_client()
        resp = await client.get(download_path)
        resp.raise_for_status()
        return resp.content

    async def get_page_attachments(
        self, page_id: str, limit: int = 100, start: int = 0
    ) -> dict[str, Any]:
        """Get all attachments for a specific page."""
        return await self._get(
            f"/rest/api/content/{page_id}/child/attachment",
            params={"limit": limit, "start": start},
        )
