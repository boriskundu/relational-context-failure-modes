"""
Truncation-detection tests for each provider client. Constructs each client without calling
__init__ (which requires a real API key) and injects a fake underlying SDK client whose response
shape mimics just enough of the real one to exercise the truncation check.

Regression coverage for: a provider reporting its output was cut off by the token limit must
always raise, even if the (possibly incomplete) text happens to still look parseable -- silently
accepting truncated output was the root cause of two earlier bugs (Claude's ThinkingBlock crash,
Groq/Qwen's empty-content-under-reasoning failure).
"""

import pytest

from agentic_docs.llm_clients import (
    AnthropicClient,
    GoogleClient,
    GroqClient,
    OpenAIClient,
    TruncatedOutputError,
)


class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _AnthropicResp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


def _bare(cls):
    """Construct a client instance without running __init__ (avoids needing a real API key)."""
    return object.__new__(cls)


def test_anthropic_raises_on_max_tokens_stop_reason(monkeypatch):
    client = _bare(AnthropicClient)
    client.model_id, client.temperature, client.thinking, client.effort = (
        "m",
        None,
        {"type": "adaptive"},
        None,
    )
    fake_resp = _AnthropicResp("max_tokens", [_Block("thinking")])

    class FakeMessages:
        def create(self, **kwargs):
            return fake_resp

    class FakeSDKClient:
        messages = FakeMessages()

    client._client = FakeSDKClient()
    with pytest.raises(TruncatedOutputError):
        client.generate_text("prompt")


def test_anthropic_does_not_raise_on_normal_stop(monkeypatch):
    client = _bare(AnthropicClient)
    client.model_id, client.temperature, client.thinking, client.effort = "m", None, None, None
    fake_resp = _AnthropicResp("end_turn", [_Block("text", "hello")])

    class FakeMessages:
        def create(self, **kwargs):
            return fake_resp

    class FakeSDKClient:
        messages = FakeMessages()

    client._client = FakeSDKClient()
    assert client.generate_text("prompt") == "hello"


class _Choice:
    def __init__(self, finish_reason, content):
        self.finish_reason = finish_reason
        self.message = type("M", (), {"content": content})()


class _ChatResp:
    def __init__(self, finish_reason, content):
        self.choices = [_Choice(finish_reason, content)]


def _fake_openai_style_client(resp):
    class FakeCompletions:
        def create(self, **kwargs):
            return resp

    class FakeChat:
        completions = FakeCompletions()

    class FakeSDKClient:
        chat = FakeChat()

    return FakeSDKClient()


def test_openai_raises_on_length_finish_reason():
    client = _bare(OpenAIClient)
    client.model_id, client.temperature, client.reasoning_effort = "m", None, None
    client._client = _fake_openai_style_client(_ChatResp("length", "partial..."))
    with pytest.raises(TruncatedOutputError):
        client.generate_text("prompt")


def test_openai_does_not_raise_on_stop():
    client = _bare(OpenAIClient)
    client.model_id, client.temperature, client.reasoning_effort = "m", None, None
    client._client = _fake_openai_style_client(_ChatResp("stop", "hello"))
    assert client.generate_text("prompt") == "hello"


def test_groq_raises_on_length_finish_reason():
    client = _bare(GroqClient)
    client.model_id, client.temperature, client.reasoning_effort = "m", None, "default"
    client._client = _fake_openai_style_client(_ChatResp("length", "partial..."))
    with pytest.raises(TruncatedOutputError):
        client.generate_text("prompt")


class _GoogleResp:
    def __init__(self, finish_reason, text):
        self.candidates = [type("C", (), {"finish_reason": finish_reason})()]
        self.text = text


def test_google_raises_on_max_tokens_finish_reason():
    from google.genai import types

    client = _bare(GoogleClient)
    client.model_id, client.temperature, client.thinking_level = "m", None, "low"
    client._genai = None
    fake_resp = _GoogleResp(types.FinishReason.MAX_TOKENS, "")

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_resp

    class FakeSDKClient:
        models = FakeModels()

    client._client = FakeSDKClient()
    with pytest.raises(TruncatedOutputError):
        client.generate_text("prompt")


def test_google_does_not_raise_on_stop_finish_reason():
    from google.genai import types

    client = _bare(GoogleClient)
    client.model_id, client.temperature, client.thinking_level = "m", None, "low"
    client._genai = None
    fake_resp = _GoogleResp(types.FinishReason.STOP, "hello")

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_resp

    class FakeSDKClient:
        models = FakeModels()

    client._client = FakeSDKClient()
    assert client.generate_text("prompt") == "hello"
