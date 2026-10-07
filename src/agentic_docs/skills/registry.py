"""
Maps each representation condition to its skill/prompt template. The three graph conditions
(predicted/oracle/shuffled) deliberately share ONE skill file — they present the same *shape* of
input (entities + links), differing only in which edges are attached, so the model must never be
able to tell which condition it's in from the prompt wording, only from the data.
"""

import importlib.resources

SKILL_FOR_CONDITION = {
    "raw": "extract_raw",
    "flat": "extract_flat",
    "predicted_graph": "extract_graph",
    "oracle_graph": "extract_graph",
    "shuffled_graph": "extract_graph",
}


def get_skill(name: str) -> str:
    path = importlib.resources.files("agentic_docs.skills") / name / "SKILL.md"
    return path.read_text(encoding="utf-8")


def get_skill_for_condition(condition: str) -> str:
    return get_skill(SKILL_FOR_CONDITION[condition])
