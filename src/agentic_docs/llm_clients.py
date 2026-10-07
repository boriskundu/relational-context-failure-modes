"""
Unified LLM client interface (ported from healthcare-eval's llm_clients.py, generalized to honor
per-model temperature/thinking config instead of one uniform setting — see config.py's module
docstring for why that's necessary here).

Adding a new provider:
  1. Add a class inheriting ``LLMClient`` implementing ``generate_text``.
  2. Add an entry to ``MODELS`` in config.py.
  3. Handle the new provider string in ``_PROVIDER_CLASSES``.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from agentic_docs.config import (
    ANTHROPIC_MAX_OUTPUT_TOKENS,
    GROQ_MAX_OUTPUT_TOKENS,
    MODELS,
    PROVIDER_ENV_VARS,
    RATE_LIMIT_MIN_WAIT,
    REQUEST_TIMEOUT,
    RUN_BASE_DELAY,
    RUN_MAX_RETRIES,
)


class TruncatedOutputError(RuntimeError):
    """Raised when a provider reports its output was cut off by the token limit, even if the
    partial text happens to still parse as valid JSON (e.g. truncated exactly at a field boundary)
    -- callers must never silently accept a truncated answer as complete. Caught by agent_graph.py
    the same as any other parse failure."""


class EmptyResponseError(RuntimeError):
    """Raised when a provider returns no usable content at all (e.g. a safety-filter block with
    zero candidates) -- deterministic at temperature 0, so not worth retrying. Caught by
    agent_graph.py the same as any other parse failure."""


# ── Retry decorator ──────────────────────────────────────────────────────────


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


def _with_retry(max_retries: int = RUN_MAX_RETRIES, base_delay: float = RUN_BASE_DELAY):
    """Retry transient failures with exponential backoff; wait longer on rate-limit errors.

    Never retries ``TruncatedOutputError``: the same prompt at the same max_tokens will truncate
    again (retrying just burns ~45s+ of backoff and repeats the cost for no chance of a different
    outcome) -- let it propagate immediately as a parse error instead.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except (TruncatedOutputError, EmptyResponseError):
                    raise
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = base_delay * (2**attempt)
                        if _is_rate_limit(exc):
                            wait = max(wait, RATE_LIMIT_MIN_WAIT)
                        print(
                            f"  Retry {attempt + 1}/{max_retries} in {wait:.0f}s "
                            f"-- {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        time.sleep(wait)
                    else:
                        raise

        return wrapper

    return decorator


# ── Base class ───────────────────────────────────────────────────────────────


class LLMClient(ABC):
    """A minimal text-in/text-out client. All prompts are built by callers."""

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        """Return the model's text completion for ``prompt``."""


# ── Anthropic (Claude) ───────────────────────────────────────────────────────


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        temperature: float | None = None,
        thinking: dict | None = None,
        effort: str | None = None,
    ):
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id
        self.temperature = temperature  # None => omit kwarg (Sonnet 5 rejects explicit temperature)
        self.thinking = thinking
        # Separate, orthogonal knob from `thinking` -- adaptive thinking is Sonnet 5's only on-mode
        # (budget_tokens returns 400), so `effort` is what actually bounds thinking depth/latency.
        # Omitting it defaults to "high"; every cell ran there until this was added, at 100-370s/cell.
        self.effort = effort

    @_with_retry()
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            # Mandatory field on this API (unlike the other three providers) -- use the model's own
            # output ceiling rather than an arbitrary cap, so adaptive thinking is never starved of
            # room to also emit an answer.
            "max_tokens": max_tokens or ANTHROPIC_MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": REQUEST_TIMEOUT,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self._client.messages.create(**kwargs)
        if resp.stop_reason == "max_tokens":
            raise TruncatedOutputError(
                f"Anthropic output truncated: stop_reason=max_tokens " f"(model={self.model_id})"
            )
        if not resp.content:
            return ""  # content-policy refusal; caller records parse_error=True
        # Adaptive thinking may return multiple content blocks (thinking + text); take the text one.
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        return ""  # no text block at all (e.g. thinking-only) -> parse_error


# ── OpenAI (GPT-5) ───────────────────────────────────────────────────────────


class OpenAIClient(LLMClient):
    def __init__(
        self, model_id: str, temperature: float | None = None, reasoning_effort: str | None = None
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id
        self.temperature = (
            temperature  # None => omit kwarg (base GPT-5 rejects explicit temperature)
        )
        self.reasoning_effort = reasoning_effort

    @_with_retry()
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": REQUEST_TIMEOUT,
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens  # GPT-5 rejects max_tokens (confirmed)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self._client.chat.completions.create(**kwargs)
        if resp.choices[0].finish_reason == "length":
            raise TruncatedOutputError(
                f"OpenAI output truncated: finish_reason=length " f"(model={self.model_id})"
            )
        return (resp.choices[0].message.content or "").strip()


# ── Groq (OpenAI-compatible; hosts the open-weight model) ────────────────────


class GroqClient(LLMClient):
    """Fills the open-weight slot (Key decision #6). DeepSeek-V4-Pro was tried first but forces
    extended thinking on for every call with no bound (234-521s/cell live) -- intractable at grid
    scale. Qwen3.6-27B on Groq gives genuine thinking (``reasoning_effort="default"``, this model's
    only non-"none" option) while staying fast. Deliberately NOT qwen3.8-27b: confirmed live via
    Groq's response headers that 3.8 carries its own much lower rate limit (8,000 TPM vs. 3.6's
    250,000 TPM) -- an account-and-model-specific quirk, not a general Groq constraint. Emits its
    reasoning inline as a leading ``<think>...</think>`` block rather than a separate API field or a
    markdown fence -- agent_graph.py's ``parse_json_answers`` strips this before parsing."""

    def __init__(
        self, model_id: str, temperature: float | None = None, reasoning_effort: str | None = None
    ):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"
        )
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    @_with_retry()
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": REQUEST_TIMEOUT,
            # Groq's own implicit default (2048) starves the answer once reasoning is on (confirmed
            # live: finish_reason=length, reasoning alone consuming the full default budget) -- give
            # it the platform's real per-call ceiling for this model instead, same fix as Anthropic.
            "max_tokens": max_tokens or GROQ_MAX_OUTPUT_TOKENS,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self._client.chat.completions.create(**kwargs)
        if resp.choices[0].finish_reason == "length":
            raise TruncatedOutputError(
                f"Groq output truncated: finish_reason=length "
                f"(model={self.model_id}) -- even at the platform's max "
                f"per-call token ceiling; likely an outlier-sized document"
            )
        return (resp.choices[0].message.content or "").strip()


# ── Google (Gemini) ──────────────────────────────────────────────────────────


class GoogleClient(LLMClient):
    """Uses the maintained ``google-genai`` SDK (``google.generativeai`` was sunset upstream and
    its GenerationConfig has no ``thinking_level`` field at all — confirmed by inspecting the
    installed package directly, not assumed)."""

    def __init__(
        self, model_id: str, temperature: float | None = None, thinking_level: str | None = None
    ):
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model_id = model_id
        self.temperature = temperature
        self.thinking_level = thinking_level

    @_with_retry()
    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        from google.genai import types

        cfg_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_tokens
        if self.temperature is not None:
            cfg_kwargs["temperature"] = self.temperature
        if self.thinking_level is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)  # type: ignore[arg-type]
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp = self._client.models.generate_content(
            model=self.model_id, contents=prompt, config=cfg
        )
        if not resp.candidates:
            # e.g. a safety-filter block with no content at all -- fails the same way on every
            # retry, so raise a clear, non-retryable error rather than let `resp.text` throw its
            # own opaque SDK exception and burn through _with_retry's backoff for nothing.
            raise EmptyResponseError(
                f"Gemini returned no candidates (model={self.model_id}), " f"likely a safety block"
            )
        if resp.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
            raise TruncatedOutputError(
                f"Gemini output truncated: finish_reason=MAX_TOKENS " f"(model={self.model_id})"
            )
        return (resp.text or "").strip()


# ── Factory ──────────────────────────────────────────────────────────────────

_PROVIDER_CLASSES = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "groq": GroqClient,
    "google": GoogleClient,
}


def is_model_available(model_name: str) -> bool:
    provider = MODELS[model_name]["provider"]
    env_var = PROVIDER_ENV_VARS.get(provider)
    return bool(env_var and os.environ.get(env_var))


def get_available_models() -> list[str]:
    """Return configured models whose API key is present; warn about the rest."""
    available, skipped = [], []
    for name in MODELS:
        if is_model_available(name):
            available.append(name)
        else:
            env_var = PROVIDER_ENV_VARS.get(MODELS[name]["provider"], "UNKNOWN")
            skipped.append((name, env_var))
    if skipped:
        print("[warn] Skipping models with missing API keys:")
        for name, var in skipped:
            print(f"   {name:20s} -> set {var} in .env to enable")
    if available:
        print(f"[ok] Available models: {available}")
    return available


def get_client(model_name: str) -> LLMClient:
    """Return an ``LLMClient`` for ``model_name`` (a key in config.MODELS), honoring that model's
    own temperature/thinking config exactly as declared — never a uniform cross-model default."""
    cfg = dict(MODELS[model_name])
    provider = cfg.pop("provider")
    model_id = cfg.pop("model_id")
    cls = _PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'")
    return cls(model_id=model_id, **cfg)
