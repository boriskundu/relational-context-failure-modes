"""
The self-verifying extraction agent: a 2-node LangGraph pipeline (extract -> forced verify/revise)
per (document, condition, model) cell. Uses this project's own ``LLMClient`` abstraction directly
inside the graph nodes (not a LangChain chat-model wrapper) — LangGraph's StateGraph is used purely
for flow control, consistent with llm_clients.py's deliberately lightweight, non-LangChain design.

Both the extract node's output and the verify node's output are retained in the result — RQ5
(correction rate, regression rate, net correction) scores both, not just the final answer.
"""
import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_docs.llm_clients import LLMClient
from agentic_docs.prompt_builder import build_prompt, build_verify_prompt

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_json_answers(raw: str) -> dict[str, str]:
    """Parse the model's JSON array of {"question","answer"} objects into a {question: answer}
    dict. Raises on malformed output — callers record this as a parse error, not a crash.

    Strips a leading ``<think>...</think>`` block first (the convention several open reasoning
    models use to emit their reasoning trace inline, ahead of the actual answer, rather than in a
    separate API field or a markdown fence) before the existing fence-stripping logic runs.
    """
    text = _THINK_RE.sub("", raw).strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    data = json.loads(text)
    if not isinstance(data, list):
        raise TypeError("Expected a JSON array of {question, answer} objects")
    result: dict[str, str] = {}
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        result[question] = answer
    return result


class _AgentState(TypedDict):
    condition: str
    document_repr: Any
    initial_answers: dict[str, str]
    final_answers: dict[str, str]
    extract_parse_error: bool
    verify_parse_error: bool


def build_agent(client: LLMClient):
    """Compile the extract->verify graph against a given client. Inject a fake client in tests.

    extract-stage and verify-stage failures are tracked as SEPARATE flags (not one shared
    ``parse_error``) because they mean different things downstream: if extract fails there is
    nothing usable at all (initial_answers={}). If verify fails, extract's own output is still
    valid and is kept as final_answers (a legitimate "verification made no change" outcome) --
    conflating the two into one flag previously caused analyze.py to drop the whole cell, including
    perfectly good extract-stage data, whenever only the verify call hiccuped.
    """

    def extract_node(state: _AgentState) -> dict:
        prompt = build_prompt(state["condition"], state["document_repr"])
        try:
            raw = client.generate_text(prompt)
            answers = parse_json_answers(raw)
        except Exception:  # noqa: BLE001 — malformed/refused output recorded, never crashes the run
            return {"initial_answers": {}, "extract_parse_error": True}
        return {"initial_answers": answers, "extract_parse_error": False}

    def verify_node(state: _AgentState) -> dict:
        if state.get("extract_parse_error"):
            return {"final_answers": {}, "verify_parse_error": False}  # verify never attempted
        previous_list = [{"question": q, "answer": a} for q, a in state["initial_answers"].items()]
        prompt = build_verify_prompt(state["condition"], state["document_repr"], previous_list)
        try:
            raw = client.generate_text(prompt)
            answers = parse_json_answers(raw)
        except Exception:  # noqa: BLE001
            return {"final_answers": dict(state["initial_answers"]), "verify_parse_error": True}
        return {"final_answers": answers, "verify_parse_error": False}

    graph = StateGraph(_AgentState)
    graph.add_node("extract", extract_node)
    graph.add_node("verify", verify_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def run_cell(client: LLMClient, condition: str, document_repr: Any) -> dict:
    """Run one (document, condition) cell against ``client`` and return a scoring-ready row."""
    app = build_agent(client)
    result = app.invoke({
        "condition": condition,
        "document_repr": document_repr,
        "initial_answers": {},
        "final_answers": {},
        "extract_parse_error": False,
        "verify_parse_error": False,
    })
    initial = result["initial_answers"]
    final = result["final_answers"]
    verify_changed = sorted(q for q in final if final.get(q) != initial.get(q))
    extract_parse_error = result.get("extract_parse_error", False)
    return {
        "initial_answers": initial,
        "final_answers": final,
        "verify_changed": verify_changed,
        "extract_parse_error": extract_parse_error,
        "verify_parse_error": result.get("verify_parse_error", False),
        # Back-compat alias: "nothing usable at all" -- this is what scoring should skip on.
        "parse_error": extract_parse_error,
    }
