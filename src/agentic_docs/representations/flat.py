"""
Flat condition: entities with type labels AND bounding boxes, but no links between them.

Carries the same bounding boxes as every graph condition — a real gap caught in review: the
proximity heuristic uses layout geometry to build edges, so if Flat lacked boxes, a Predicted-Graph
win could reflect "layout information helped" rather than "relations helped." See Key decision #1.
"""
from agentic_docs.funsd.parse import DocumentRecord, reading_order_key


def build_flat(doc: DocumentRecord) -> list[dict]:
    ordered = sorted(doc.entities, key=reading_order_key)
    return [
        {"id": e.id, "label": e.label, "text": e.text, "box": list(e.box)}
        for e in ordered
    ]
