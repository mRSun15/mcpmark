# Filesystem MCP HTTP Mode Setup

Run the Filesystem MCP service via HTTP/REST instead of STDIO.

**Note:** This uses a simple REST API (not MCP-over-HTTP protocol). The server provides:
- `GET /tools` - List available tools
- `POST /mcp/tools/{tool_name}` - Execute a tool

## Quick Start

### 1. Start REST Server

```bash
./start-filesystem-rest-server.sh
```

### 2. Configure MCPMark

Add to `.mcp_env`:
```env
FILESYSTEM_USE_HTTP_MODE=true
FILESYSTEM_REST_URL=http://127.0.0.1:8001
```

### 3. Run Tasks

```bash
python -m pipeline \
  --exp-name gpt-5-http-test \
  --mcp filesystem \
  --tasks desktop/music_report \
  --models gpt-5-mini \
  --k 1
```

**How to verify HTTP mode is active:**
- ✅ **HTTP mode**: Log shows `"Connecting to filesystem MCP REST server at: http://127.0.0.1:8001/"`
- ❌ **STDIO mode**: Log shows `"Secure MCP Filesystem Server running on stdio"`

### 4. Stop Server

```bash
docker stop filesystem-mcp-rest-server
```

## API Endpoints

- `GET /health` - Health check
- `GET /tools` - List available tools  
- `POST /mcp/tools/{tool_name}` - Execute a tool

### Examples

**List all available tools:**
```bash
curl http://localhost:8001/tools
```

**List directories under /app/results:**
```bash
curl -X POST http://localhost:8001/mcp/tools/list_directory \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/results"}'
```

**Read a file:**
```bash
curl -X POST http://localhost:8001/mcp/tools/read_file \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/results/gpt-5-http-test/execution.log"}'
```

**Get file info:**
```bash
curl -X POST http://localhost:8001/mcp/tools/get_file_info \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/results/gpt-5-http-test"}'
```

## Management

```bash
# View logs
docker logs filesystem-mcp-rest-server

# Restart
docker restart filesystem-mcp-rest-server

# Remove
docker rm -f filesystem-mcp-rest-server
```

