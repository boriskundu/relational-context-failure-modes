"""
Layout-proximity heuristic for the Predicted Graph condition (Key decision #2).

For each answer entity, predicts its governing question as the nearest question entity that
precedes it in reading order (above, or left-on-the-same-row) — the standard pattern lightweight
deployed form parsers use when a full ML linking model isn't in budget. Deterministic, zero API
calls, uses ONLY entity type/text/box data — never reads FUNSD's gold `linking` field, which is the
leakage guard for this module specifically (verified by a code-review grep, not just by
convention — see the plan's Verification section).

Ties (two candidate questions at equal distance) are broken by picking the one earlier in reading
order, so the result is fully deterministic for a fixed document.
"""

from agentic_docs.funsd.parse import DocumentRecord, Entity, reading_order_key


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance_sq(a: Entity, b: Entity) -> float:
    ax, ay = _center(a.box)
    bx, by = _center(b.box)
    return (ax - bx) ** 2 + (ay - by) ** 2


def predict_links(doc: DocumentRecord) -> list[tuple[int, int]]:
    """Returns deduplicated, undirected (question_id, answer_id) edges, one per answer entity that
    has at least one preceding question candidate (answers with no preceding question — e.g. the
    very first entity on the page — get no predicted edge)."""
    questions = [e for e in doc.entities if e.label == "question"]
    answers = [e for e in doc.entities if e.label == "answer"]

    edges: set[tuple[int, int]] = set()
    for answer in answers:
        candidates = [q for q in questions if reading_order_key(q) < reading_order_key(answer)]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda q: (_distance_sq(q, answer), reading_order_key(q)),
        )
        edges.add((min(nearest.id, answer.id), max(nearest.id, answer.id)))
    return sorted(edges)
