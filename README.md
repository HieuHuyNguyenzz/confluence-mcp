# Bitu Confluence MCP Server

MCP server kết nối LLMs với Confluence Server/Data Center. Cho phép liệt kê spaces, lấy page dưới dạng Markdown và tìm kiếm trong Knowledge Base.

## Cài đặt

```bash
pip install -e .
```

## Cấu hình

Copy file `.env.example` thành `.env` và điền thông tin.

## Chạy server

### Local (Stdio mode)
Cho Claude Desktop, Cursor...
```bash
python -m confluence_server.server
```

### Local (HTTP mode)
```bash
python -m confluence_server.server --transport streamable-http
```

### Docker
```bash
chmod +x deploy.sh
./deploy.sh
```

## Cấu hình trong Claude Desktop
Thêm vào `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "confluence": {
      "command": "python",
      "args": ["-m", "confluence_server.server"],
      "env": {
        "CONFLUENCE_URL": "https://confluence.yourcompany.com",
        "CONFLUENCE_API_TOKEN": "your_api_token"
      }
    }
  }
}
```

## Tools
- `list_spaces`: Liệt kê các spaces.
- `get_page_as_markdown`: Lấy nội dung page.
- `crawl_space`: Crawl toàn bộ space.
- `convert_file_to_markdown`: Convert file local sang markdown.
