"""
Tests against a real FUNSD annotation file (tests/fixtures/sample_funsd_annotation.json,
copied verbatim from the training set) chosen specifically because it contains every edge case
this parser needs to handle correctly:
  - unlinked questions (ids 1, 5) -> excluded from scorable pairs
  - a header linking to multiple question entities (id 19 -> 6, 7) -> not a scorable pair, since
    6 and 7 are themselves labeled "question", not "answer"
  - a question with one linked answer (id 3 -> 12)
  - a question with TWO linked answer entities (id 20 -> 21, 22) -> concatenated in reading order
"""
from agentic_docs.funsd.parse import parse_document

EXPECTED_SCORABLE_QUESTION_IDS = {2, 3, 10, 13, 15, 17, 20}


def test_parses_all_entities(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="0000971160")
    assert doc.doc_id == "0000971160"
    assert len(doc.entities) == 24
    assert {e.id for e in doc.entities} == set(range(24))


def test_links_are_deduplicated_and_undirected(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    # Each linked pair appears on both endpoints in the raw file; parsed links must be deduplicated.
    assert (2, 16) in doc.links
    assert (3, 12) in doc.links
    assert (4, 13) in doc.links  # stored as (13,4) in the raw file; normalized to (min,max)
    assert len(doc.links) == 10  # 10 unique undirected edges in this fixture


def test_scorable_pairs_exclude_unlinked_questions(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    scorable_ids = {p.question_id for p in doc.scorable_pairs}
    assert scorable_ids == EXPECTED_SCORABLE_QUESTION_IDS
    # id 1 (":" , empty linking) and id 5 ("", empty linking) must not appear
    assert 1 not in scorable_ids
    assert 5 not in scorable_ids


def test_header_to_multiple_questions_is_not_a_scorable_pair(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    scorable_ids = {p.question_id for p in doc.scorable_pairs}
    # ids 6 and 7 are labeled "question" but only linked to header id 19 (not to any "answer")
    assert 6 not in scorable_ids
    assert 7 not in scorable_ids
    # the header<->question edges still exist in the undirected link list (structural context)
    assert (6, 19) in doc.links
    assert (7, 19) in doc.links


def test_single_answer_pair_resolves_correct_text(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    pair = next(p for p in doc.scorable_pairs if p.question_id == 3)
    assert pair.question_text == "Date:"
    assert pair.answer_entity_ids == (12,)
    assert pair.gold_answer_text == "9/ 3/ 92"


def test_multi_answer_pair_concatenates_in_reading_order(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    pair = next(p for p in doc.scorable_pairs if p.question_id == 20)
    assert pair.question_text == "Manager Comments:"
    # entity 21's box has a smaller y1 (644) than entity 22's (662) -> 21 must come first
    assert pair.answer_entity_ids == (21, 22)
    assert pair.gold_answer_text == (
        "Manager, please contact suggester and forward comments to the Quality Council."
    )


def test_entity_by_id_lookup(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    entity = doc.entity_by_id(9)
    assert entity.label == "header"
    assert "QUALITY" in entity.text or "SUGGESTION" in entity.text
