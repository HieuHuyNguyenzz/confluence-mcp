# Confluence MCP Server

MCP server kết nối LLMs với Confluence Server/Data Center. Cho phép liệt kê spaces, crawl toàn bộ nội dung và convert sang Markdown.

## Cài đặt

```bash
pip install -e .
```

Hoặc dùng uv:

```bash
uv pip install -e .
```

## Cấu hình

Copy file `.env.example` thành `.env` và điền thông tin:

```env
CONFLUENCE_URL=https://confluence.yourcompany.com
CONFLUENCE_USERNAME=your_username
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_VERIFY_SSL=true
```

### Lấy API Token

1. Vào Confluence → Profile → Personal Settings → API Tokens
2. Click "Create API Token"
3. Copy token vào `CONFLUENCE_API_TOKEN`

## Chạy server

### Stdio mode (cho Claude Desktop, Cursor...)

```bash
python -m confluence_mcp.server
```

### HTTP mode

```bash
python -m confluence_mcp.server --transport streamable-http
```

## Cấu hình trong Claude Desktop

Thêm vào `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "confluence": {
      "command": "python",
      "args": ["-m", "confluence_mcp.server"],
      "env": {
        "CONFLUENCE_URL": "https://confluence.yourcompany.com",
        "CONFLUENCE_USERNAME": "your_username",
        "CONFLUENCE_API_TOKEN": "your_api_token"
      }
    }
  }
}
```

## Tools

### `list_spaces`

Liệt kê tất cả Confluence spaces có thể truy cập.

**Parameters:**
- `limit` (int, default: 25): Số lượng spaces tối đa
- `start` (int, default: 0): Offset pagination

**Returns:**
```json
[
  {
    "key": "ENG",
    "name": "Engineering",
    "description": "Engineering documentation",
    "type": "global",
    "status": "current"
  }
]
```

### `get_page_as_markdown`

Lấy nội dung một page cụ thể dưới dạng Markdown.

**Parameters:**
- `page_id` (str): Confluence page ID (numeric)

**Returns:**
```json
{
  "page_id": "123456",
  "title": "Architecture Overview",
  "space_key": "ENG",
  "path": "ENG/Architecture Overview.md",
  "content": "---\ntitle: Architecture Overview\n...\n",
  "attachments": [
    {"filename": "diagram.png", "media_type": "image/png", "size": 12345}
  ]
}
```

### `crawl_space`

Crawl toàn bộ space, bao gồm tất cả pages (cả child pages), download attachments và convert sang Markdown.

**Parameters:**
- `space_key` (str): Confluence space key (vd: `ENG`, `DOC`)
- `include_attachments` (bool, default: true): Có download attachments không
- `output_dir` (str | None): Thư mục lưu files. Nếu `None`, chỉ trả về trong response

**Returns:**
```json
[
  {
    "page_id": "123456",
    "title": "Page Title",
    "space_key": "ENG",
    "path": "ENG/Parent/Page Title.md",
    "content": "---\ntitle: Page Title\n...\n",
    "attachments": [
      {
        "filename": "file.pdf",
        "local_path": "output/ENG/Parent/attachments/file.pdf",
        "media_type": "application/pdf",
        "size": 54321
      }
    ]
  }
]
```

## Cấu trúc project

```
confluence-mcp/
├── pyproject.toml
├── .env.example
├── .gitignore
└── src/
    └── confluence_mcp/
        ├── __init__.py
        ├── server.py           # FastMCP server + 3 tools
        ├── client.py           # Confluence REST API wrapper
        └── converter.py        # HTML → Markdown converter
```

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| MCP Framework | `fastmcp` 3.2.0 |
| HTTP Client | `httpx` |
| HTML → Markdown | `markdownify` + `bs4` |
| Config | `python-dotenv` |
| Python | 3.10+ |
