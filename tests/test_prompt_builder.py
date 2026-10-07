"""
The "same prompt template, only representation varies" guarantee — the core scientific claim
this project depends on. If this test ever fails, the experiment's central comparison is no
longer valid.
"""
from agentic_docs.prompt_builder import _extract_input_specific, build_prompt, build_verify_prompt
from agentic_docs.skills.registry import get_skill_for_condition

BEGIN = "<!-- INPUT-SPECIFIC:BEGIN -->"
END = "<!-- INPUT-SPECIFIC:END -->"


def _outside_block(skill_text: str) -> tuple[str, str]:
    before = skill_text[: skill_text.index(BEGIN)]
    after = skill_text[skill_text.index(END) + len(END):]
    return before, after


def test_raw_flat_and_graph_skills_are_fully_condition_blind():
    """Invariant flipped deliberately (was: input-specific blocks must differ per condition).
    The old per-condition wording -- describing each representation's structure, and for graph
    conditions specifically warning the model that links "may contain errors" -- gave the model a
    wording-based signal about which condition it was in, on top of the representation content
    itself. That confounded RQ2/RQ4/RQ5: flat/raw never carried an analogous distrust instruction.
    Instruction text is now identical across all five conditions (raw/flat/predicted/oracle/
    shuffled) -- the ONLY thing that may differ between conditions is the substituted
    `{document_representation}` content itself."""
    raw_skill = get_skill_for_condition("raw")
    flat_skill = get_skill_for_condition("flat")
    graph_skill = get_skill_for_condition("predicted_graph")

    raw_before, raw_after = _outside_block(raw_skill)
    flat_before, flat_after = _outside_block(flat_skill)
    graph_before, graph_after = _outside_block(graph_skill)

    assert raw_before == flat_before == graph_before
    assert raw_after == flat_after == graph_after
    assert raw_skill == flat_skill == graph_skill


def test_predicted_oracle_and_shuffled_share_the_exact_same_skill_template():
    """The model must never be able to tell which of these three conditions it's in from the
    prompt wording alone — only the attached data (which edges) may differ."""
    predicted = get_skill_for_condition("predicted_graph")
    oracle = get_skill_for_condition("oracle_graph")
    shuffled = get_skill_for_condition("shuffled_graph")
    assert predicted == oracle == shuffled


def test_all_five_conditions_produce_byte_identical_prompts_outside_the_document():
    """Permanent regression guard for the condition-blindness fix (see
    test_raw_flat_and_graph_skills_are_fully_condition_blind): render the extract and verify
    prompts for every one of the 5 real conditions against the same placeholder document content,
    then assert the surrounding instruction text is byte-identical once that placeholder is
    removed. The document representation is the only thing this study manipulates -- if this test
    ever fails, a condition-specific wording difference has crept back in."""
    conditions = ["raw", "flat", "predicted_graph", "oracle_graph", "shuffled_graph"]
    placeholder = "PLACEHOLDER_DOCUMENT_CONTENT_FOR_AUDIT"

    extract_prompts = {build_prompt(c, placeholder).replace(placeholder, "") for c in conditions}
    assert len(extract_prompts) == 1, "extract prompts differ outside the document content"

    previous = [{"question": "Q", "answer": "A"}]
    verify_prompts = {
        build_verify_prompt(c, placeholder, previous).replace(placeholder, "") for c in conditions
    }
    assert len(verify_prompts) == 1, "verify prompts differ outside the document content"


def test_build_prompt_substitutes_document_placeholder():
    prompt = build_prompt("raw", "SOME UNIQUE DOCUMENT TEXT 12345")
    assert "SOME UNIQUE DOCUMENT TEXT 12345" in prompt
    assert "<<DOCUMENT>>" not in prompt


def test_build_prompt_serializes_dict_documents_as_json():
    doc = {"entities": [{"id": 1, "text": "hi"}], "links": []}
    prompt = build_prompt("flat", doc)
    assert '"text": "hi"' in prompt


def test_build_verify_prompt_includes_previous_answers_and_document():
    previous = [{"question": "Name", "answer": "wrong guess"}]
    prompt = build_verify_prompt("flat", "DOC CONTENT", previous)
    assert "wrong guess" in prompt
    assert "DOC CONTENT" in prompt
    assert "re-checking" in prompt.lower()


# Run 3: words that would leak reliability/epistemic-weight signals about the shown links,
# reintroducing the confound the condition-blind rewrite removed (see the module docstring above).
# A clean grep is an implementation safeguard, not proof the prompt is neutral -- it only rules out
# this specific word list, not every way wording could carry a signal.
_BANNED_WORDS = [
    "correct", "incorrect", "reliable", "unreliable", "candidate", "proposed",
    "trust", "distrust", "primary guide", "plausible", "blindly", "override",
    "fall back", "fallback",
]


def test_input_specific_blocks_contain_no_banned_reliability_words():
    for condition in ["raw", "flat", "predicted_graph", "oracle_graph", "shuffled_graph"]:
        block = _extract_input_specific(get_skill_for_condition(condition)).lower()
        for word in _BANNED_WORDS:
            assert word not in block, f"banned word {word!r} found in {condition} input-specific block"


def test_input_specific_blocks_do_not_gate_instructions_on_link_presence():
    """The prompt must describe what a link is and that some questions lack one, without adding a
    procedural rule like 'if no link, then...' that would tell the model how to weigh link
    presence/absence -- that's exactly the kind of hidden fallback policy the distrust-instruction
    confound introduced originally."""
    for condition in ["raw", "flat", "predicted_graph", "oracle_graph", "shuffled_graph"]:
        block = _extract_input_specific(get_skill_for_condition(condition)).lower()
        assert "if no link" not in block
        assert "if there is no link" not in block
        assert "if not linked" not in block
        assert "in that case" not in block
