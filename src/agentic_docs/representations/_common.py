"""Shared shape-builder for the three graph conditions (oracle/predicted/shuffled) — identical
entity list as Flat, differing only in which edge list is attached."""
from collections import defaultdict

from agentic_docs.funsd.parse import DocumentRecord, reading_order_key
from agentic_docs.representations.flat import build_flat


def build_graph_shape(doc: DocumentRecord, edges: list[tuple[int, int]]) -> dict:
    return {
        "entities": build_flat(doc),
        "links": [list(pair) for pair in edges],
    }


def scorable_edges(doc: DocumentRecord) -> set[tuple[int, int]]:
    """The gold question<->answer edges only (excludes structural edges, e.g. a header chaining to
    several sub-questions). Shared by oracle_graph.py and shuffled_graph.py so all three graph
    conditions carry the SAME edge-type scope as predicted_graph.py's heuristic (which can only ever
    produce question<->answer edges) -- an earlier version let Oracle/Shuffled additionally carry
    structural edges Predicted structurally cannot, which would confound a Predicted-vs-Oracle
    comparison with edge-type coverage, not just edge correctness (the same class of confound the
    box-parity fix eliminated for Flat vs. the graph conditions)."""
    return {
        (min(p.question_id, a), max(p.question_id, a))
        for p in doc.scorable_pairs
        for a in p.answer_entity_ids
    }


def shown_answer_text_by_question(doc: DocumentRecord, edges: list) -> dict[int, str]:
    """For each question, the answer text implied by ``edges`` alone -- i.e. what a model that
    faithfully relayed exactly the graph it was shown would answer. Used by analyze.py's
    relation-following-error metric: comparing this against gold_answer_text tells us whether a
    given question's edge in this representation was itself correct (Oracle: always is; Predicted/
    Shuffled: may not be), independent of whether the MODEL's actual answer matched.

    Only meaningful for the three graph conditions -- Raw/Flat carry no edges, so callers should
    treat a question absent from the result (no edge at all, e.g. a heuristic miss) as "not
    applicable", the same as a question whose edge happens to be correct."""
    entities_by_id = {e.id: e for e in doc.entities}
    question_ids = {p.question_id for p in doc.scorable_pairs}
    answer_ids_by_question: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        if a in question_ids and b not in question_ids:
            answer_ids_by_question[a].append(b)
        elif b in question_ids and a not in question_ids:
            answer_ids_by_question[b].append(a)
        # both-question or neither-question edges don't occur for scorable Q&A edges; skip.

    result = {}
    for qid, answer_ids in answer_ids_by_question.items():
        answers = sorted((entities_by_id[i] for i in answer_ids if i in entities_by_id),
                          key=reading_order_key)
        result[qid] = " ".join(e.text for e in answers if e.text)
    return result
