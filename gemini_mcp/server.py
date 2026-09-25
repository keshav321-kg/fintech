"""Minimal MCP server that exposes Google Gemini as a tool.

Standard library only. Speaks MCP JSON-RPC over stdio (for Claude Code /
Claude Desktop) or over HTTP POST (for a claude.ai custom connector).

    GEMINI_API_KEY=... python -m gemini_mcp.server            # stdio
    GEMINI_API_KEY=... python -m gemini_mcp.server --http 8080  # HTTP at /mcp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

TOOLS = [
    {
        "name": "ask_gemini",
        "description": "Send a prompt to Google Gemini and return its text reply.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt to send."},
                "system": {"type": "string", "description": "Optional system instruction."},
                "model": {"type": "string", "description": f"Model id (default {DEFAULT_MODEL})."},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
            },
            "required": ["prompt"],
        },
    }
]


def _post_json(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def ask_gemini(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    post: Callable[[str, dict, dict], dict] = _post_json,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if temperature is not None:
        body["generationConfig"] = {"temperature": temperature}
    url = f"{API_BASE}/models/{model or DEFAULT_MODEL}:generateContent"
    data = post(url, body, {"Content-Type": "application/json", "x-goog-api-key": api_key})
    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise RuntimeError(f"Gemini returned no answer: {reason}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict, post: Callable[[str, dict, dict], dict] = _post_json) -> dict | None:
    """Handle one JSON-RPC message; returns None for notifications."""
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return None
    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "gemini-mcp", "version": "0.1.0"},
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params", {})
        if params.get("name") != "ask_gemini":
            return _error(msg_id, -32602, f"Unknown tool: {params.get('name')}")
        args = params.get("arguments", {})
        try:
            text = ask_gemini(
                args["prompt"], args.get("system"), args.get("model"),
                args.get("temperature"), post=post,
            )
            return _result(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            text = f"Gemini API error {e.code}: {detail}"
        except (KeyError, RuntimeError, urllib.error.URLError) as e:
            text = f"Error: {e}"
        return _result(msg_id, {"content": [{"type": "text", "text": text}], "isError": True})
    return _error(msg_id, -32601, f"Method not found: {method}")


def serve_stdio() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line))
        except json.JSONDecodeError:
            reply = _error(None, -32700, "Parse error")
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


class _HTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self.send_error(404)
            return
        token = os.environ.get("MCP_AUTH_TOKEN")
        if token and self.headers.get("Authorization") != f"Bearer {token}":
            self.send_error(401)
            return
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            reply = handle(msg)
        except json.JSONDecodeError:
            reply = _error(None, -32700, "Parse error")
        if reply is None:
            self.send_response(202)
            self.end_headers()
            return
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.send_error(405)  # no server-initiated SSE stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--http", type=int, metavar="PORT", help="serve over HTTP instead of stdio")
    args = parser.parse_args()
    if args.http:
        ThreadingHTTPServer(("0.0.0.0", args.http), _HTTPHandler).serve_forever()
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
