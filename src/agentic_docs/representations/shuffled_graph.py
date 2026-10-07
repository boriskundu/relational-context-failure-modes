"""
Shuffled Graph condition: entities + a degree-preserving derangement of the gold question->answer
edges — a negative control proving the agent benefits from CORRECT relational info, not merely
relational-shaped input. See Key decisions #1/#3.

Only the SCORABLE question->answer edges are ever included (deranged); structural edges (e.g. a
header chaining to several sub-questions) are dropped entirely, not just left un-deranged — matching
Predicted Graph's and Oracle Graph's edge-type scope (see _common.scorable_edges). An earlier version
passed structural edges through unchanged, which meant Shuffled (and Oracle) carried edge types
Predicted's heuristic could never produce -- an edge-type-coverage confound distinct from the
intended experimental variable (edge correctness).

Derangement is degree-preserving and per-document (per Key decision #3): a question's whole
answer-group is reassigned to another question with the SAME number of linked answers, within the
same document — reassigning across documents would be meaningless (the agent only ever sees one
document), and reassigning across different degrees would silently change the structural shape of
the negative control. A question whose degree is unique within its own document (no same-degree
partner to swap with) cannot be deranged and is excluded from Shuffled-condition SCORING for that
question (not from the representation — its original, correct edge is kept, since inventing an
incorrect one isn't possible, but scoring it under "Shuffled" would misleadingly count a still-correct
edge as if it were part of the negative control).
"""
import random
from collections import defaultdict

from agentic_docs.funsd.parse import DocumentRecord
from agentic_docs.representations._common import build_graph_shape


def _derangement(n: int, rng: random.Random) -> list[int]:
    """A random permutation of range(n) with no fixed points. Reject-and-resample; fine for the
    small n this project ever calls it with (rarely more than a handful of same-degree questions
    per document)."""
    if n < 2:
        raise ValueError("Cannot derange fewer than 2 items")
    indices = list(range(n))
    while True:
        candidate = indices[:]
        rng.shuffle(candidate)
        if all(candidate[i] != i for i in range(n)):
            return candidate


def build_shuffled_graph(doc: DocumentRecord, seed: int) -> tuple[dict, set[int]]:
    """Returns (representation, excluded_question_ids) — the latter must be excluded from
    Shuffled-condition scoring (their edge could not be deranged and remains correct)."""
    rng = random.Random(seed)

    by_degree: dict[int, list] = defaultdict(list)
    for pair in doc.scorable_pairs:
        by_degree[len(pair.answer_entity_ids)].append(pair)

    deranged_edges: list[tuple[int, int]] = []
    excluded_question_ids: set[int] = set()

    for pairs in by_degree.values():
        if len(pairs) < 2:
            # No same-degree partner in this document — keep the original (correct) edges, and
            # flag these questions as unusable for the Shuffled-condition negative control.
            for pair in pairs:
                excluded_question_ids.add(pair.question_id)
                for a in pair.answer_entity_ids:
                    deranged_edges.append((min(pair.question_id, a), max(pair.question_id, a)))
            continue
        permutation = _derangement(len(pairs), rng)
        for i, pair in enumerate(pairs):
            wrong_answers = pairs[permutation[i]].answer_entity_ids
            for a in wrong_answers:
                deranged_edges.append((min(pair.question_id, a), max(pair.question_id, a)))

    all_edges = sorted(set(deranged_edges))
    return build_graph_shape(doc, all_edges), excluded_question_ids
