import pytest

from gemini_mcp import server


def fake_post(url, body, headers):
    fake_post.calls.append((url, body, headers))
    return {"candidates": [{"content": {"parts": [{"text": "Hello"}, {"text": " there"}]}}]}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake_post.calls = []


def call(args):
    return server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "ask_gemini", "arguments": args}},
        post=fake_post,
    )["result"]


def test_initialize_and_list():
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["capabilities"] == {"tools": {}}
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert [t["name"] for t in tools["result"]["tools"]] == ["ask_gemini"]


def test_notification_has_no_reply():
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ask_gemini_builds_request():
    result = call({"prompt": "Hi", "system": "Be brief", "model": "gemini-x", "temperature": 0.2})
    assert result == {"content": [{"type": "text", "text": "Hello there"}], "isError": False}
    url, body, headers = fake_post.calls[0]
    assert url.endswith("/models/gemini-x:generateContent")
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief"}]}
    assert body["generationConfig"] == {"temperature": 0.2}
    assert headers["x-goog-api-key"] == "test-key"


def test_missing_key_is_tool_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    result = call({"prompt": "Hi"})
    assert result["isError"] and "GEMINI_API_KEY" in result["content"][0]["text"]


def test_unknown_method():
    reply = server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
    assert reply["error"]["code"] == -32601
