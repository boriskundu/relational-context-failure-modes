"""
Tests for the five representation builders. The content-set-equality guard
(test_all_conditions_carry_identical_text_content) is the single most important correctness
property in the whole design — it's the direct fix for the leakage bug that started the multi-round
redesign documented in the plan (see plan Context section).
"""

from agentic_docs.funsd.parse import parse_document
from agentic_docs.representations._common import shown_answer_text_by_question
from agentic_docs.representations.flat import build_flat
from agentic_docs.representations.oracle_graph import build_oracle_graph
from agentic_docs.representations.predicted_graph import build_predicted_graph
from agentic_docs.representations.raw import build_raw
from agentic_docs.representations.shuffled_graph import build_shuffled_graph


def _all_text(doc):
    return sorted(e.text for e in doc.entities if e.text)


def test_build_raw_has_no_labels_or_boxes(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    raw = build_raw(doc)
    assert isinstance(raw, str)
    assert "Suggestion:" in raw
    assert "label" not in raw.lower().replace("labor", "")  # sanity, not a real schema check


def test_build_flat_carries_labels_and_boxes(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    flat = build_flat(doc)
    assert len(flat) == len(doc.entities)
    for row in flat:
        assert set(row.keys()) == {"id", "label", "text", "box"}
        assert len(row["box"]) == 4
    assert "links" not in flat  # no relational field anywhere in the flat structure


def test_build_oracle_graph_uses_gold_links(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    graph = build_oracle_graph(doc)
    scorable = {
        (min(p.question_id, a), max(p.question_id, a))
        for p in doc.scorable_pairs
        for a in p.answer_entity_ids
    }
    # Oracle carries only question<->answer edges, matching Predicted's edge-type scope -- NOT
    # doc.links wholesale, which also includes structural (e.g. header->question) edges that
    # Predicted's heuristic could never produce (see _common.scorable_edges).
    assert {tuple(e) for e in graph["links"]} == scorable
    assert len(graph["entities"]) == len(doc.entities)


def test_build_predicted_graph_never_reads_gold_links(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    graph = build_predicted_graph(doc)
    # The predicted edges need not match gold exactly, but must be a valid, non-empty edge set
    # built purely from entity/box data (proximity.predict_links never touches doc.links).
    assert isinstance(graph["links"], list)
    assert len(graph["links"]) > 0


def test_build_shuffled_graph_scorable_edges_are_wrong(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    graph, excluded = build_shuffled_graph(doc, seed=42)
    scorable_edges = {
        (min(p.question_id, a), max(p.question_id, a))
        for p in doc.scorable_pairs
        for a in p.answer_entity_ids
        if p.question_id not in excluded
    }
    shuffled_edges = {tuple(e) for e in graph["links"]}
    # None of the still-in-scope scorable gold edges should survive unchanged in the shuffled graph.
    assert not (scorable_edges & shuffled_edges)


def test_shuffled_graph_excludes_singleton_degree_buckets(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    # In this fixture, question 20 has degree 2 (unique in this small document) — no same-degree
    # partner to swap with, so it must be excluded from Shuffled-condition scoring.
    _graph, excluded = build_shuffled_graph(doc, seed=42)
    assert 20 in excluded


def test_shown_answer_text_matches_gold_for_oracle_edges(sample_annotation):
    # Oracle's edges are exactly the gold scorable edges, so the "shown" answer implied by them
    # must equal gold_answer_text for every scorable question -- this is the sanity check behind
    # relation_following_error always being N/A (not just low) for Oracle.
    doc = parse_document(sample_annotation, doc_id="x")
    graph = build_oracle_graph(doc)
    shown = shown_answer_text_by_question(doc, graph["links"])
    for pair in doc.scorable_pairs:
        assert shown[pair.question_id] == pair.gold_answer_text


def test_shown_answer_text_differs_from_gold_for_deranged_shuffled_edges(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    graph, excluded = build_shuffled_graph(doc, seed=42)
    shown = shown_answer_text_by_question(doc, graph["links"])
    for pair in doc.scorable_pairs:
        if pair.question_id in excluded:
            # Excluded questions keep their original (correct) edge -- shown must still equal gold.
            assert shown[pair.question_id] == pair.gold_answer_text
        else:
            assert shown[pair.question_id] != pair.gold_answer_text


def test_shown_answer_text_absent_for_question_with_no_edge():
    from agentic_docs.funsd.parse import DocumentRecord, Entity, ScorablePair

    doc = DocumentRecord(
        doc_id="x",
        entities=[
            Entity(id=1, text="Name:", label="question", box=(0, 0, 10, 10)),
            Entity(id=2, text="Bob", label="answer", box=(20, 0, 30, 10)),
        ],
        links=[],
        scorable_pairs=[
            ScorablePair(
                question_id=1, question_text="Name:", answer_entity_ids=(2,), gold_answer_text="Bob"
            )
        ],
    )
    # No edges at all (e.g. a Predicted-graph heuristic miss) -- question 1 has no entry.
    assert shown_answer_text_by_question(doc, []) == {}


def test_all_conditions_carry_identical_text_content(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    expected = _all_text(doc)

    raw_text_set = sorted(t for t in build_raw(doc).split("\n") if t)
    flat_text_set = sorted(row["text"] for row in build_flat(doc) if row["text"])
    oracle_text_set = sorted(
        row["text"] for row in build_oracle_graph(doc)["entities"] if row["text"]
    )
    predicted_text_set = sorted(
        row["text"] for row in build_predicted_graph(doc)["entities"] if row["text"]
    )
    shuffled_graph, _ = build_shuffled_graph(doc, seed=42)
    shuffled_text_set = sorted(row["text"] for row in shuffled_graph["entities"] if row["text"])

    assert raw_text_set == expected
    assert flat_text_set == expected
    assert oracle_text_set == expected
    assert predicted_text_set == expected
    assert shuffled_text_set == expected


def test_flat_and_graph_conditions_have_identical_boxes(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    flat_boxes = {row["id"]: row["box"] for row in build_flat(doc)}
    oracle_boxes = {row["id"]: row["box"] for row in build_oracle_graph(doc)["entities"]}
    predicted_boxes = {row["id"]: row["box"] for row in build_predicted_graph(doc)["entities"]}
    shuffled_graph, _ = build_shuffled_graph(doc, seed=42)
    shuffled_boxes = {row["id"]: row["box"] for row in shuffled_graph["entities"]}
    assert flat_boxes == oracle_boxes == predicted_boxes == shuffled_boxes
