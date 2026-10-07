"""
Oracle Graph condition: entities + FUNSD's gold question<->answer edges (scorable edges only —
excludes structural/header edges, matching Predicted Graph's edge-type scope; see
_common.scorable_edges for why).

Reframed explicitly per Key decision #1/RQ3: this tests FAITHFUL RELAY under complete information
(does the agent accurately relay information it's already been fully given), not "relational
reasoning helps" — an agent handed the exact answer mapping isn't reasoning about anything. State
this in the paper's Method and Limitations, not just here.
"""
from agentic_docs.funsd.parse import DocumentRecord
from agentic_docs.representations._common import build_graph_shape, scorable_edges


def build_oracle_graph(doc: DocumentRecord) -> dict:
    return build_graph_shape(doc, sorted(scorable_edges(doc)))
