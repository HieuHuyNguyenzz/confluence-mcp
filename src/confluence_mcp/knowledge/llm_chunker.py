"""LLM-based text chunking and summarization for RAG."""

import asyncio
import logging
import re
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

LLM_CHUNKING_PROMPT = """Bạn là một hệ thống chunking văn bản thông minh chuyên dụng cho RAG.

### Nhiệm vụ:
Nhận đầu vào là một văn bản và chia nó thành các đoạn (chunks) nhỏ sao cho mỗi chunk giữ được ý nghĩa trọn vẹn và có ngữ cảnh rõ ràng.

### Quy tắc chia chunk:
1. **Độ dài**: Mỗi chunk khoảng 100-300 từ.
2. **Tính toàn vẹn**: Không cắt giữa câu. Luôn giữ nguyên các câu đầy đủ.
3. **Cấu trúc**: Ưu tiên chia theo tiêu đề (#, ##, ###) hoặc đoạn văn rõ ràng.
4. **Định dạng**: Giữ nguyên các ký tự định dạng tiêu đề. 
5. **Nội dung**: Không thay đổi, thêm, hoặc bỏ nội dung gốc.
6. **Làm giàu**: 
   - Mỗi chunk phải có 5-10 từ khóa quan trọng.
   - Cuối mỗi chunk, thêm một câu tóm tắt ngữ cảnh ngắn (Bắt đầu bằng: "Đoạn này...").

### 🧩 Định dạng đầu ra (JSON):
Bạn PHẢI trả về kết quả dưới định dạng JSON thuần túy. Đảm bảo các chuỗi tiếng Việt được mã hóa đúng chuẩn UTF-8 và không bị ngắt quãng sai. 
Kết quả phải bắt đầu bằng `{` và kết thúc bằng `}`. Không thêm lời dẫn, không sử dụng markdown block (```json).

Ví dụ:
Input: \"# Hướng dẫn cài đặt\\nBước 1: Tải file. Bước 2: Chạy installer. Bước 3: Cấu hình IP.\"
Output:
{{
  \"chunks\": [
    {{
      \"content\": \"# Hướng dẫn cài đặt\\nBước 1: Tải file. Bước 2: Chạy installer. Bước 3: Cấu hình IP. (Đoạn này hướng dẫn các bước cài đặt cơ bản)\",
      \"keywords\": [\"cài đặt\", \"hướng dẫn\", \"installer\", \"IP\"]
    }}
  ]
}}

Văn bản cần chunk:
<text>
{text}
</text>"""

LLM_SUMMARY_PROMPT = """Bạn là một chuyên gia phân tích tài liệu cấp cao.

### Nhiệm vụ:
Hãy tạo một bản tóm tắt cô đọng và toàn diện cho tài liệu dưới đây để làm ngữ cảnh toàn cục (global context).

### Yêu cầu:
1. **Mục đích**: Tài liệu này viết về cái gì? Giải quyết vấn đề gì?
2. **Thực thể**: Liệt kê các khái niệm, thuật ngữ, sản phẩm then chốt.
3. **Cấu trúc**: Mô tả ngắn gọn luồng thông tin.
4. **Phong cách**: Súc tích, khách quan, tiếng Việt, 100-200 từ.
5. **Định dạng**: Không chào hỏi, không giải thích, chỉ trả về nội dung bản tóm tắt.

Văn bản cần tóm tắt:
<text>
{text}
</text>"""

class LLMChunker:
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str,
        openai_model: str,
        prompt: str = LLM_CHUNKING_PROMPT,
        max_chunk_size: int = 8000,
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
            temperature=0,
        )
        self.prompt = prompt
        self.max_chunk_size = max_chunk_size

    async def generate_document_summary(self, text: str) -> str:
        """Generate a global summary for the entire document to be used as context for chunks."""
        limit = 100000 
        if len(text) > limit:
            log.info(f"Document too large ({len(text)} chars), summarizing in parts...")
            mid = len(text) // 2
            part1 = await self.generate_document_summary(text[:mid])
            part2 = await self.generate_document_summary(text[mid:])
            text = f"Part 1 Summary:\n{part1}\n\nPart 2 Summary:\n{part2}"
        
        try:
            prompt = LLM_SUMMARY_PROMPT.replace("{text}", text)
            response = await self.llm.ainvoke(prompt)
            summary = response.content.strip()
            log.info("Successfully generated global document summary.")
            return summary
        except Exception as e:
            log.warning(f"Failed to generate global summary: {e}")
            return ""

    def _parse_json_response(self, content: str) -> Any:
        """Robustly extract JSON from LLM response, handling common issues with Vietnamese content."""
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*$', '', content)
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
        # Try to find the first { and last }
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                # Attempt to fix common trailing commas or broken JSON structures
                json_str = content[start:end+1]
                # Remove trailing commas before closing brackets/braces
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
        return None

    def _split_for_llm(self, text: str) -> List[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        
        splits = []
        for i in range(0, len(text), self.max_chunk_size):
            splits.append(text[i:i + self.max_chunk_size])
        return splits

    async def chunk_text(self, text: str, global_summary: str = "") -> List[Dict[str, Any]]:
        initial_splits = self._split_for_llm(text)
        
        async def process_segment(i, segment):
            prompt_with_text = self.prompt.replace("{text}", segment)
            # Semaphore is handled by the caller (sync_milvus.py)
            try:
                response = await self.llm.ainvoke(prompt_with_text)
                content = response.content
                data = self._parse_json_response(content)
                if data is not None:
                    if isinstance(data, dict) and "chunks" in data:
                        chunks = data["chunks"]
                        if isinstance(chunks, list):
                            processed = []
                            for c in chunks:
                                if isinstance(c, dict):
                                    original_content = c.get("content", "")
                                    c["content"] = f"--- GLOBAL CONTEXT ---\n{global_summary}\n\n--- CONTENT ---\n{original_content}"
                                    processed.append(c)
                                else:
                                    processed.append({"content": segment, "keywords": []})
                            return processed
                        elif isinstance(data, list):
                            processed = []
                            for c in data:
                                if isinstance(c, dict):
                                    original_content = c.get("content", "")
                                    c["content"] = f"--- GLOBAL CONTEXT ---\n{global_summary}\n\n--- CONTENT ---\n{original_content}"
                                    processed.append(c)
                                else:
                                    processed.append({"content": segment, "keywords": []})
                            return processed
                    log.warning(f"LLM returned non-JSON content for segment {i}. Content: {content[:200]}...")
                    return [{"content": segment, "keywords": []}]
            except Exception as e:
                log.warning(f"LLM chunking failed for segment {i}: {e}")
                return [{"content": segment, "keywords": []}]
        
        tasks = [process_segment(i, segment) for i, segment in enumerate(initial_splits)]
        results = await asyncio.gather(*tasks)
        
        all_chunks = []
        for res in results:
            all_chunks.extend(res)
        return all_chunks
