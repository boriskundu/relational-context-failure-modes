"""
Builds the extract prompt (from a SKILL.md template, plain string substitution — no templating
engine, deliberately, so a diff between two rendered prompts is trivially readable) and the
verify/revise prompt (re-uses each condition's own input-specific description, so the verify step
still describes the document the same way the extract step did).
"""

import json
from typing import Any

from agentic_docs.skills.registry import get_skill_for_condition

_BEGIN_MARKER = "<!-- INPUT-SPECIFIC:BEGIN -->"
_END_MARKER = "<!-- INPUT-SPECIFIC:END -->"

_VERIFY_TEMPLATE = """You are re-checking your own previous answer to a document form-filling \
task, to catch and correct any mistakes before finalizing.

{input_specific}

Your previous answer was:
{previous_answers}

Re-read the document below and your previous answer above. For each question, check whether the \
answer is actually supported by the document — correct any answer that is wrong, ungrounded, or \
inconsistent, and leave correct answers unchanged.

Instructions:
- Base corrections ONLY on information present in the document below. Do not infer, guess, or fabricate.
- If an answer should be blank because the document doesn't contain it, output an empty string "".
- Respond with ONLY a JSON array, no other text, in this exact format:
[{{"question": "<question text as it appears>", "answer": "<answer text, or empty string>"}}, ...]

Document:
{document}
"""


def _serialize_document(document_repr: Any) -> str:
    if isinstance(document_repr, str):
        return document_repr
    return json.dumps(document_repr, indent=2)


def _extract_input_specific(skill_text: str) -> str:
    start = skill_text.index(_BEGIN_MARKER) + len(_BEGIN_MARKER)
    end = skill_text.index(_END_MARKER)
    return skill_text[start:end].strip()


def build_prompt(condition: str, document_repr: Any) -> str:
    skill = get_skill_for_condition(condition)
    return skill.replace("<<DOCUMENT>>", _serialize_document(document_repr))


def build_verify_prompt(condition: str, document_repr: Any, previous_answers: list[dict]) -> str:
    skill = get_skill_for_condition(condition)
    return _VERIFY_TEMPLATE.format(
        input_specific=_extract_input_specific(skill),
        previous_answers=json.dumps(previous_answers, indent=2),
        document=_serialize_document(document_repr),
    )
