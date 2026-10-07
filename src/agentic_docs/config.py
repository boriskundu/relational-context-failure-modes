"""
Central configuration: model definitions (with per-model temperature/thinking config), paths,
and shared run parameters.

Per-model config is NOT uniform across models, unlike a typical judge-grid setup — this is a
deliberate, verified decision (see plan/ARCHITECTURE.md): Claude Sonnet 5 and base GPT-5 reject
any explicit ``temperature`` value outright (API error), and Gemini 3.x cannot fully disable
thinking. Each model therefore uses its own best-available settings, held CONSTANT across all five
representation conditions for that model — the comparison this project makes is within-model (same
model, same settings, only the input representation varies), never a cross-model "whose config is
more aggressive" comparison. See Key decision #6 in the plan for the full reasoning.

Open-weight slot: DeepSeek-V4-Pro was tried first and rejected -- forces extended thinking on for
every call with no bound, confirmed live at 234-521s/cell, intractable at grid scale. Replaced with
Qwen3.8-27B via Groq: verified live to expose a genuine, distinct reasoning trace at
reasoning_effort="high" while staying sub-second per call (Groq's inference infrastructure is fast
regardless of reasoning), and to reliably follow the extraction prompt's strict-JSON-only
instruction.

Exact model_id strings and provider-specific reasoning/thinking kwargs below were verified live
against each provider's actual API (not just docs) before use.
"""
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

FUNSD_RAW_DIR = DATA_DIR / "funsd_raw" / "dataset"
# One combined dataset (199 docs = FUNSD's 149 train + 50 test) -- the project no longer holds out
# a frozen test split; see the project memory for why (user's explicit call, 2026-08-31).
DOCUMENTS_PATH = DATA_DIR / "documents_all.json"

# ── Shared run parameters ────────────────────────────────────────────────────
# No artificial output-token cap: OpenAI/Google omit max_tokens entirely and use each provider's
# own default. Anthropic's Messages API requires an explicit value, so it gets the model's real
# output ceiling (confirmed via `client.models.retrieve`) rather than an arbitrary number — a real,
# low arbitrary cap previously caused adaptive thinking to consume the entire budget with no room
# left for the answer (confirmed live on a large document: 4999/5000 tokens spent on thinking, zero
# text emitted, stop_reason=max_tokens). Groq's own implicit default (2048) hits the identical
# failure mode once reasoning is enabled -- confirmed live -- so it also gets an explicit ceiling,
# capped at Groq's own hard per-call max for this model (confirmed live: requests above 16384 are
# rejected outright).
ANTHROPIC_MAX_OUTPUT_TOKENS = 128_000
GROQ_MAX_OUTPUT_TOKENS = 16_384
RANDOM_SEED = 42           # heuristic tie-breaking, derangement, bootstrap resampling

REQUEST_TIMEOUT = 600     # generous: unbounded adaptive thinking on large documents can run minutes
RUN_MAX_RETRIES = 5
RUN_BASE_DELAY = 3.0
RATE_LIMIT_MIN_WAIT = 20.0
CHECKPOINT_EVERY = 5      # partial file (and --status visibility) refreshes at least this often
HEARTBEAT_EVERY = 1       # print a progress line to stdout after every N completed cells

# Heuristic validation: internal holdout size carved out of the combined 199-document dataset
# (superseded Key decision #2's train/test carve-out, now that there's no separate frozen test set).
HEURISTIC_HOLDOUT_SIZE = 35

# ── Models — 4 distinct labs, per-model temperature/thinking config ─────────
# `temperature: None` means the provider REJECTS an explicit value — omit the kwarg entirely rather
# than pass a default, since some of these APIs 400 on any explicit temperature at all.
MODELS: dict[str, dict[str, Any]] = {
    "claude-sonnet-5": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-5",  # verify exact model_id against Anthropic's models list
        "temperature": None,  # rejects explicit temperature outright (verified)
        "thinking": {"type": "adaptive"},  # default-on adaptive thinking; only real option here
        # Effort is a separate knob from `thinking` -- bounds thinking depth/latency within adaptive
        # mode. Omitting it defaults to "high" (100-370s/cell observed live); "low" keeps thinking
        # genuinely on while staying fast, matching the task's actual complexity (structured
        # extraction/verification, not deep multi-step reasoning).
        "effort": "low",
    },
    "gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",  # verify exact model_id against OpenAI's models list
        "temperature": None,  # base GPT-5 rejects explicit temperature outright (verified)
        "reasoning_effort": "low",  # minimal/low/medium/high; no true zero-reasoning on base GPT-5.
        # "low" (not "minimal") to keep thinking genuinely on, matching the other 3 models' own
        # lowest-while-still-thinking setting -- consistent low-effort standard across all 4 models.
    },
    "qwen3.6-27b": {
        "provider": "groq",
        "model_id": "qwen/qwen3.6-27b",  # NOT qwen3.8-27b -- verified live via response headers:
        # 3.8's account-level rate limit is 8,000 TPM (its own model-specific cap, unrelated to
        # the account overall), while 3.6 gets 250,000 TPM -- over 31x higher. 3.6 is also the
        # version already validated in the sibling healthcare-eval project.
        "temperature": 0.0,  # verified accepted (unlike Sonnet 5/base GPT-5)
        "reasoning_effort": "default",  # this model only accepts "none" or "default" -- "default"
        # keeps thinking genuinely enabled (needed for a fair per-model comparison against the
        # other 3 models' own thinking/reasoning settings), unlike healthcare-eval's judges (which
        # deliberately used "none" everywhere to pin an identical zero-thinking baseline).
    },
    "gemini-3.7-flash": {
        "provider": "google",
        "model_id": "gemini-3.7-flash",  # verify exact model_id against Gemini API docs
        "temperature": 0.0,  # independently settable alongside thinking on this model
        "thinking_level": "low",  # thinking cannot be fully disabled on Gemini 3.x
    },
}

# ── Provider → required environment variable ─────────────────────────────────
PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
}
