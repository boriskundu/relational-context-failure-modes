"""
Validate the proximity heuristic's predicted edges against gold, on an internal holdout carved out
of the combined document dataset (no separate frozen test set in this project — see
config.DOCUMENTS_PATH). Reports micro-averaged precision/recall/F1 across the holdout, since entity
ids are only unique within a document (per-document overlaps are summed before dividing).
"""

import random
from dataclasses import dataclass

from agentic_docs.config import HEURISTIC_HOLDOUT_SIZE, RANDOM_SEED
from agentic_docs.funsd.parse import DocumentRecord
from agentic_docs.heuristic.proximity import predict_links


@dataclass(frozen=True)
class HeuristicValidationResult:
    n_documents: int
    n_gold_edges: int
    n_predicted_edges: int
    n_true_positive: int
    precision: float
    recall: float
    f1: float


def select_holdout(
    documents: list[DocumentRecord],
    holdout_size: int = HEURISTIC_HOLDOUT_SIZE,
    seed: int = RANDOM_SEED,
) -> list[DocumentRecord]:
    """A fixed, reproducible subset of the combined document dataset, used only to validate the
    heuristic (never fed to the model-comparison grid)."""
    rng = random.Random(seed)
    indices = list(range(len(documents)))
    rng.shuffle(indices)
    return [documents[i] for i in indices[:holdout_size]]


def _gold_edges(doc: DocumentRecord) -> set[tuple[int, int]]:
    return {
        (min(p.question_id, a), max(p.question_id, a))
        for p in doc.scorable_pairs
        for a in p.answer_entity_ids
    }


def validate_heuristic(holdout_docs: list[DocumentRecord]) -> HeuristicValidationResult:
    total_gold = 0
    total_predicted = 0
    total_tp = 0
    for doc in holdout_docs:
        gold = _gold_edges(doc)
        predicted = set(predict_links(doc))
        total_gold += len(gold)
        total_predicted += len(predicted)
        total_tp += len(gold & predicted)

    precision = total_tp / total_predicted if total_predicted else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return HeuristicValidationResult(
        n_documents=len(holdout_docs),
        n_gold_edges=total_gold,
        n_predicted_edges=total_predicted,
        n_true_positive=total_tp,
        precision=precision,
        recall=recall,
        f1=f1,
    )
