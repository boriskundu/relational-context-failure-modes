import json

import pytest

from agentic_docs.agent_graph import parse_json_answers, run_cell


class FakeLLMClient:
    """Returns queued responses in order, one per generate_text call — lets a test script the
    extract call's response and the verify call's response independently."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _json_list(pairs: dict[str, str]) -> str:
    return json.dumps([{"question": q, "answer": a} for q, a in pairs.items()])


def test_parse_json_answers_plain():
    raw = '[{"question": "Name", "answer": "John"}]'
    assert parse_json_answers(raw) == {"Name": "John"}


def test_parse_json_answers_strips_markdown_fence():
    raw = '```json\n[{"question": "Name", "answer": "John"}]\n```'
    assert parse_json_answers(raw) == {"Name": "John"}


def test_parse_json_answers_rejects_non_list():
    with pytest.raises(TypeError):
        parse_json_answers('{"question": "Name", "answer": "John"}')


def test_parse_json_answers_strips_leading_think_block():
    # Regression test: some open reasoning models (confirmed live: qwen3.6-27b on Groq) emit their
    # reasoning trace inline as a leading <think>...</think> block, ahead of the actual JSON answer,
    # rather than in a separate API field or a markdown fence.
    raw = '<think>\nLet me work through this form field by field...\n</think>\n[{"question": "Name", "answer": "John"}]'
    assert parse_json_answers(raw) == {"Name": "John"}


def test_extract_then_verify_wiring_calls_model_twice():
    client = FakeLLMClient([
        _json_list({"Name": "wrong guess"}),
        _json_list({"Name": "corrected"}),
    ])
    row = run_cell(client, "flat", "some document")
    assert len(client.calls) == 2
    assert row["initial_answers"] == {"Name": "wrong guess"}
    assert row["final_answers"] == {"Name": "corrected"}


def test_verify_node_correction_overrides_extract_output():
    client = FakeLLMClient([
        _json_list({"Date": "1990"}),
        _json_list({"Date": "1999"}),
    ])
    row = run_cell(client, "raw", "doc")
    assert row["verify_changed"] == ["Date"]
    assert row["final_answers"]["Date"] == "1999"


def test_unchanged_answers_are_not_flagged_as_verify_changed():
    client = FakeLLMClient([
        _json_list({"Date": "1999", "Name": "Bob"}),
        _json_list({"Date": "1999", "Name": "Bob"}),
    ])
    row = run_cell(client, "raw", "doc")
    assert row["verify_changed"] == []


def test_extract_parse_error_short_circuits_without_crashing():
    client = FakeLLMClient(["not valid json at all"])
    row = run_cell(client, "raw", "doc")
    assert row["extract_parse_error"] is True
    assert row["parse_error"] is True  # back-compat alias == extract_parse_error
    assert row["verify_parse_error"] is False  # verify was never attempted
    assert row["initial_answers"] == {}
    assert row["final_answers"] == {}
    # verify node must not have been called since extract already failed
    assert len(client.calls) == 1


def test_verify_parse_error_falls_back_to_initial_answers_and_is_still_scorable():
    client = FakeLLMClient([
        _json_list({"Name": "Alice"}),
        "garbage, not json",
    ])
    row = run_cell(client, "raw", "doc")
    # Extract succeeded -- this cell must NOT be treated as "nothing usable" (that's what
    # previously caused analyze.py to silently drop perfectly good extract-stage data whenever
    # only the verify call failed to parse).
    assert row["extract_parse_error"] is False
    assert row["parse_error"] is False
    assert row["verify_parse_error"] is True
    assert row["initial_answers"] == {"Name": "Alice"}
    assert row["final_answers"] == {"Name": "Alice"}
