"""
Predicted Graph condition: entities + edges from the layout-proximity heuristic
(heuristic/proximity.py) — never FUNSD's gold linking field. See Key decision #2.
"""
from agentic_docs.funsd.parse import DocumentRecord
from agentic_docs.heuristic.proximity import predict_links
from agentic_docs.representations._common import build_graph_shape


def build_predicted_graph(doc: DocumentRecord) -> dict:
    return build_graph_shape(doc, predict_links(doc))
