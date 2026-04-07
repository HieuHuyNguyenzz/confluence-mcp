# Bitu Confluence MCP Server

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
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_VERIFY_SSL=true

# Cấu hình Dify Sync
DIFY_BASE_URL=https://api.dify.ai/v1
DIFY_API_KEY=your_dify_api_key
DIFY_DATASET_ID=your_dataset_id
SYNC_SPACE_KEY=your_space_key
```

### Lấy API Token

1. Vào Confluence → Profile → Personal Settings → API Tokens
2. Click "Create API Token"
3. Copy token vào `CONFLUENCE_API_TOKEN`

## Chạy server

### Local (Stdio mode)

Cho Claude Desktop, Cursor...

```bash
python -m confluence_mcp.server
```

### Local (HTTP mode)

```bash
python -m confluence_//C- la l_server --transport streamable-http
```

### Docker (khuyên dùng cho server)

**Cách 1: Dùng script triển khai nhanh (Khuyên dùng)**

Script `deploy.sh` tự động hóa quá trình build image với hỗ trợ Proxy nội bộ và chạy container.

```bash
# 1. Copy và điền file .env
cp .env.example .env

# 2. Chạy script deploy
chmod +x deploy.sh
./deploy.sh

# Server chạy tại http://localhost:8000
```

**Cách 2: docker compose**

```bash
# 1. Copy và điền file .env
cp .env.example .env

# 2. Build và chạy
docker compose up -d --build

# Server chạy tại http://localhost:8000
```

**Cách 3: Chạy trên server remote**

```bash
# Clone repo trên server
git clone https://github.com/HieuHuyNguyenzz/bitu-confluence-mcp.git
cd bitu-confluence-mcp

# Tạo file .env
cp .env.example .env
nano .env  # Điền credentials

# Build và chạy nhanh
chmod +x deploy.sh
./deploy.sh

# Kiểm tra logs
docker logs -f bitu-confluence-mcp

# Server chạy tại http://<server-ip>:8000/mcp
```

**Cấu hình reverse proxy (nginx)**

```nginx
server {
    listen 80;
    server_name mcp.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Đồng bộ dữ liệu sang Dify (RAG Pipeline)

Nếu bạn muốn nạp toàn bộ dữ liệu từ Confluence vào Dify Knowledge để làm RAG:

```bash
python -m confluence_mcp.sync_dify
```
Script này sẽ tự động crawl Space được cấu hình trong `SYNC_SPACE_KEY` và đẩy toàn bộ nội dung (bao gồm file đính kèm) vào Dify Dataset.

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
        "CONFLUENCE_API_TOKEN": "your_api_token"
      }
    }
  }
}
```

## Tools

**Lưu ý:** Tất cả các tool crawl đều sử dụng xử lý song song (parallel fetching) để tối ưu tốc độ cho các Space có dữ liệu lớn.

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
- `include_attachments` (bool, default: true): Có trích xuất file đính kèm không

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

Crawl toàn bộ space, bao gồm tất cả pages (cả child pages), download attachments và convert sang Markdown. Toàn bộ quá trình xử lý diễn ra in-memory để đảm bảo tốc độ và bảo mật.

**Parameters:**
- `space_key` (str): Confluence space key (vd: `ENG`, `DOC`)
- `include_attachments` (bool, default: true): Có download attachments không

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
        "media_type": "application/pdf",
        "size": 54321,
        "content": "...extracted text..."
      }
    ]
  }
]
```

## Cấu trúc project

```
bitu-confluence-mcp/
├── pyproject.toml
├── .env.example
├── .gitignore
├── deploy.sh           # Script triển khai Docker nhanh
└── src/
    └── confluence_mcp/
        ├── __init__.py
        ├── server.py           # FastMCP server + 3 tools
        ├── client.py           # Confluence REST API wrapper
        ├── converter.py        # HTML → Markdown converter
        ├── extractor.py        # File attachment extractor
        └── sync_dify.py        # Sync pipeline to Dify Knowledge
```

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| MCP Framework | `fastmcp` 3.2.0 |
| HTTP Client | `httpx` |
| HTML → Markdown | `markdownify` + `bs4` |
| Config | `python-dotenv` |
| Python | 3.10+ |
