"""LLM-based chunking for RAG embedding."""

import json
from typing import Any

import openai


class RAGChunker:
    """Use LLM to intelligently chunk markdown content for RAG."""

    SYSTEM_PROMPT = """You are an expert at chunking technical documentation for Retrieval-Augmented Generation (RAG).

Your task is to split markdown content into semantically coherent chunks that are self-contained and meaningful for embedding."""

    USER_PROMPT = """Split the following markdown content into chunks for RAG.

Rules:
1. Each chunk must be self-contained - a reader should understand it without seeing other chunks
2. Keep related content together: do NOT split code blocks, tables, lists, or paragraphs
3. Include relevant heading context in each chunk so it's self-contained
4. Target chunk size: approximately {max_tokens} tokens
5. Allow ~15% overlap between adjacent chunks to maintain context
6. Minimum chunk size: 100 tokens - merge smaller fragments into their parent chunk
7. Preserve frontmatter (---) in the first chunk only
8. DO NOT create chunks that are empty, contain only whitespace, or have no meaningful text content
9. DO NOT create chunks that only contain markdown syntax (e.g. just "---", "...", "___", or just table borders)
10. If a section has no meaningful content, skip it entirely - do not output an empty chunk
11. CRITICAL: Only create a chunk if it contains information that could answer a real question. If a chunk is just headings, navigation, metadata, or structural placeholders with no substantive content, skip it. Each chunk must contain actual facts, explanations, descriptions, procedures, or data that someone might ask about.

Output ONLY a valid JSON array. No explanation, no markdown code fences.

Format:
[
  {{
    "content": "chunk markdown here...",
    "heading_path": "H1 > H2 > H3 breadcrumb"
  }}
]

Markdown content:
---
{content}
---"""

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 800):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens

    def chunk_content(self, content: str) -> list[dict[str, str]]:
        """Chunk markdown content using LLM.

        Returns list of {content, heading_path} dicts.
        """
        if not content or not content.strip():
            return []

        prompt = self.USER_PROMPT.format(
            max_tokens=self.max_tokens,
            content=content,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = response.choices[0].message.content or "[]"

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [{"content": content, "heading_path": ""}]

        if isinstance(data, dict):
            items = data.get("chunks", data.get("items", []))
        elif isinstance(data, list):
            items = data
        else:
            return [{"content": content, "heading_path": ""}]

        chunks = []
        for item in items:
            if isinstance(item, dict) and "content" in item:
                content = item["content"].strip()
                if not self._is_valid_chunk(content):
                    continue
                chunks.append({
                    "content": content,
                    "heading_path": item.get("heading_path", item.get("heading", "")),
                })

        return chunks if chunks else [{"content": content, "heading_path": ""}]

    @staticmethod
    def _is_valid_chunk(content: str) -> bool:
        """Check if a chunk has meaningful, answerable content.

        A valid chunk must contain information that could answer a real question.
        Structural content, navigation, metadata, and placeholders are rejected.
        """
        if not content:
            return False

        text = content.strip()
        if len(text) < 40:
            return False

        lines = text.splitlines()
        heading_only = all(
            line.strip().startswith("#") or line.strip() == "" or line.strip().startswith("---")
            for line in lines
        )
        if heading_only:
            return False

        text_no_markup = text
        for ch in ["#", "-", "*", "|", "`", ">", "_", "~"]:
            text_no_markup = text_no_markup.replace(ch, "")
        text_no_markup = text_no_markup.strip()

        if len(text_no_markup) < 20:
            return False

        words = text_no_markup.split()
        real_words = [w for w in words if w.isalpha() and len(w) > 2]

        if len(real_words) < 5:
            code_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
            if len(code_lines) >= 3:
                return True
            return False

        meaningless_phrases = [
            "table of contents",
            "on this page",
            "navigation",
            "breadcrumbs",
            "last updated",
            "created by",
            "modified by",
            "updated by",
            "no content",
            "not applicable",
        ]
        lower = text_no_markup.lower()
        has_meaningless = any(phrase in lower for phrase in meaningless_phrases)
        if has_meaningless:
            substantive_words = len([w for w in words if w.isalpha() and len(w) > 3])
            if substantive_words < 5:
                return False

        return True

    def chunk_page(
        self,
        content: str,
        page_id: str,
        title: str,
        space_key: str,
        path: str,
        created_date: str = "",
        last_modified: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Chunk a single page's markdown content with metadata.

        Returns page result with chunks list.
        """
        chunks_raw = self.chunk_content(content)

        chunks = []
        for i, chunk in enumerate(chunks_raw):
            chunk_obj: dict[str, Any] = {
                "chunk_id": f"{page_id}-{i}",
                "page_id": page_id,
                "title": title,
                "space_key": space_key,
                "path": path,
                "content": chunk["content"],
                "heading_path": chunk["heading_path"],
                "created_date": created_date,
                "last_modified": last_modified,
            }
            chunks.append(chunk_obj)

        if attachments:
            for att in attachments:
                att_content = att.get("content", "")
                if att_content and len(att_content.strip()) > 100:
                    att_chunks = self.chunk_content(att_content)
                    for j, att_chunk in enumerate(att_chunks):
                        if not self._is_valid_chunk(att_chunk["content"]):
                            continue
                        chunks.append({
                            "chunk_id": f"{page_id}-att-{att.get('filename', 'unknown')}-{j}",
                            "page_id": page_id,
                            "title": title,
                            "space_key": space_key,
                            "path": path,
                            "content": att_chunk["content"],
                            "heading_path": f"Attachment: {att.get('filename', '')} > {att_chunk['heading_path']}",
                            "created_date": created_date,
                            "last_modified": last_modified,
                            "source_file": att.get("filename", ""),
                            "source_media_type": att.get("media_type", ""),
                        })

        return {
            "page_id": page_id,
            "title": title,
            "space_key": space_key,
            "path": path,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
