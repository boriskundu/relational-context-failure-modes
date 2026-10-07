"""
Raw condition (RQ1 floor): reading-order text only, no type labels, no bounding boxes, no links.

Named precisely in the paper as "structure-stripped, reading-order text" — it's still sorted by
box position (a soft positional signal survives), just not the literal unstructured document text
an OCR pass over a printed page would produce. See Key decision #1.
"""
from agentic_docs.funsd.parse import DocumentRecord, reading_order_key


def build_raw(doc: DocumentRecord) -> str:
    ordered = sorted(doc.entities, key=reading_order_key)
    return "\n".join(e.text for e in ordered if e.text)
