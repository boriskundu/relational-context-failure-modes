from agentic_docs.funsd.parse import DocumentRecord, Entity, parse_document
from agentic_docs.heuristic.proximity import predict_links
from agentic_docs.heuristic.validate import select_holdout, validate_heuristic


def test_predict_links_returns_deduplicated_undirected_edges(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")
    edges = predict_links(doc)
    assert len(edges) == len(set(edges))
    for a, b in edges:
        assert a < b


def test_predict_links_never_touches_gold_linking_field(sample_annotation):
    """Sanity check on the leakage guard: corrupting doc.links entirely must not change the
    heuristic's output, since predict_links only ever reads entity type/text/box data."""
    doc = parse_document(sample_annotation, doc_id="x")
    edges_before = predict_links(doc)
    doc.links = []  # sabotage the gold links
    doc.scorable_pairs = []
    edges_after = predict_links(doc)
    assert edges_before == edges_after


def test_select_holdout_is_reproducible():
    fake_docs = list(range(149))  # stand-in for DocumentRecords; only identity/order matters here
    h1 = select_holdout(fake_docs, holdout_size=35, seed=42)
    h2 = select_holdout(fake_docs, holdout_size=35, seed=42)
    assert h1 == h2
    assert len(h1) == 35


def test_predict_links_breaks_ties_by_earlier_reading_order():
    # Two questions equidistant (by squared center distance) from the answer -- Q_early and
    # Q_late have centers (100,100) and (100,110) against the answer's center (105,105), both at
    # squared distance 50. Q_early's box has a smaller (y1, x1) reading-order key, so it must win
    # the tie, not whichever happens to be considered first/last by iteration order.
    answer = Entity(id=2, text="A", label="answer", box=(100, 100, 110, 110))
    q_early = Entity(id=0, text="Q1", label="question", box=(95, 95, 105, 105))
    q_late = Entity(id=1, text="Q2", label="question", box=(95, 105, 105, 115))
    doc = DocumentRecord(doc_id="x", entities=[q_early, q_late, answer], links=[], scorable_pairs=[])

    edges = predict_links(doc)
    assert edges == [(0, 2)]  # answer 2 must link to q_early (id 0), not q_late (id 1)


def test_validate_heuristic_perfect_prediction_scores_1(sample_annotation):
    doc = parse_document(sample_annotation, doc_id="x")

    class _PerfectDoc:
        """Wraps a real doc but reports predict_links == gold, to test the scoring math in
        isolation from the heuristic's actual (imperfect) predictions."""

    import agentic_docs.heuristic.validate as validate_module

    gold = validate_module._gold_edges(doc)
    original_predict = validate_module.predict_links
    try:
        validate_module.predict_links = lambda d: sorted(gold)
        result = validate_heuristic([doc])
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
    finally:
        validate_module.predict_links = original_predict
