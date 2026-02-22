# MCP stdio ↔ Streamable HTTP bridge

A single Rust executable that Claude Desktop can spawn as an MCP server. It reads newline-delimited JSON-RPC from stdin, forwards each message to a remote MCP server over Streamable HTTP, and writes responses (and streamed events) as newline-delimited JSON to stdout. No Node or Python required.

## Build

Install [Rust](https://rustup.rs) (via rustup). This crate uses `rust-toolchain.toml` to require **Rust 1.85.0** or newer (needed for current dependencies). From this directory, run `cargo build`; rustup will use or install the right toolchain automatically.

```bash
cargo build --release
```

The binary is `target/release/mcp-stdio-http-bridge` (or `mcp_stdio_http_bridge` on some platforms).

### Universal macOS binary (x86_64 + aarch64)

```bash
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create \
  target/x86_64-apple-darwin/release/mcp-stdio-http-bridge \
  target/aarch64-apple-darwin/release/mcp-stdio-http-bridge \
  -output mcp-stdio-http-bridge-macos-universal
```

Install the targets if needed:

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
```

## Run manually

```bash
export URI=<your-deployment-url>/mcp
export BEARER_TOKEN=your-token
export RUST_LOG=info
./target/release/mcp-stdio-http-bridge
```

Then type or pipe newline-delimited JSON-RPC lines to stdin; responses appear on stdout. Logs go to stderr.

## Claude Desktop config

Point Claude Desktop’s MCP `command` at the bridge binary so it runs as a stdio MCP server:

```json
{
  "mcpServers": {
    "<your-mcp-name>": {
      "command": "/absolute/path/to/mcp-stdio-http-bridge",
      "env": {
        "URI": "<your-deployment-url>/mcp",
        "BEARER_TOKEN": "your-token",
        "MCP_NAME": "<your-mcp-name>",
        "RUST_LOG": "info"
      }
    }
  }
}
```

Optional env vars:

- `URI` – remote MCP Streamable HTTP endpoint (default: `<your-deployment-url>/mcp`)
- `BEARER_TOKEN` – if set, sent as `Authorization: Bearer <token>`
- `MCP_NAME` – used in logs only
- `MCP_TIMEOUT_MS` – request timeout in ms (default: 60000)
- `MCP_MAX_QUEUE` – max queued stdin messages (default: 10000)
- `RUST_LOG` – log level (e.g. `info`, `debug`, `mcp_stdio_http_bridge=debug`)

## Tests

```bash
cargo test
```

Unit tests cover SSE parsing and behaviour (see `src/remote_transport.rs`). No integration tests (avoids extra dependencies that require newer Rust).

## Design notes

### Stdio ↔ Streamable HTTP mapping

- **Stdio side:** One JSON-RPC message per line (newline-delimited). No framing beyond that; request IDs and payloads are preserved and not reinterpreted.
- **Remote side:** Each line is sent as a single HTTP POST to the configured URI:
  - Body: raw JSON-RPC (UTF-8), `Content-Type: application/json`
  - `Accept: application/json, text/event-stream` so the server may respond with either a single JSON body or an SSE stream.
- **Responses:**
  - **200 + application/json:** Body is forwarded as one line to stdout.
  - **200 + text/event-stream:** Response is parsed as SSE; each event’s `data` (or concatenated `data` lines per event) is emitted as one newline-delimited JSON line to stdout. Event boundaries are blank lines.
  - **202 Accepted:** Empty body; nothing is written to stdout.
  - **4xx/5xx:** Body (if any) is still forwarded as one line so the client sees the server’s error.

### Streaming

Streaming is handled entirely in `remote_transport`: we read the response body and, when `Content-Type` is `text/event-stream`, parse SSE and collect one JSON message per event (or per concatenated `data` block). Those are returned as a `Vec<String>` and the bridge writes each element as a separate line to stdout. No reordering; message order is preserved.

### Retry / backoff

- Only **network-level** failures are retried (connection, timeout, request errors). Invalid UTF-8 or I/O errors are not retried.
- Backoff is exponential: start 500 ms, double each time, capped at 30 s. The same message is retried until success or non-retryable error. Message order is preserved; we do not pull the next message until the current one is sent.
- While retrying, new stdin messages are queued in a bounded channel (`MCP_MAX_QUEUE`, default 10k). When the channel is full, the stdin reader blocks (backpressure).

### Assumptions about Streamable HTTP

- Single endpoint: POST one JSON-RPC message per request. No GET/session in this bridge (stateless).
- Response is either (a) one JSON body, or (b) SSE with one JSON-RPC message per event (`data:` line or concatenated `data:` lines per event). The format is implemented in `src/remote_transport.rs`; change `parse_sse_to_json_lines` and the `Content-Type` handling there if your server differs (e.g. different SSE format or newline-delimited JSON instead of SSE).
