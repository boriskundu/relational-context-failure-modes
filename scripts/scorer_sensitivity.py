"""Scorer sensitivity checks for the RQ2/RQ4 headline contrasts (see Limitations):
does the primary fuzzy-match ruler's exact threshold choice, or its
known order-sensitivity on multi-value answers, change the paper's reported conclusions?

Two independent checks, both against Round 3 (results/extractions/run3.json) only, since that's
where the headline RQ2/RQ4 numbers come from:

1. **Threshold sweep**: rerun score_run at thresholds {80, 85, 90, 95} (module default is 85) and
   recompute RQ2 (predicted_graph vs flat) / RQ4 (shuffled_graph vs flat) bootstrap deltas per model.
   Reports whether sign and rough magnitude survive.
2. **Order-insensitive multi-value rescoring**: for scorable pairs with more than one gold answer
   entity (`len(answer_entity_ids) > 1`), rescore correctness via a token-multiset overlap
   comparator instead of straight `fuzz.ratio` on the concatenated (order-dependent) string --
   already-disclosed scorer limitation (see Limitations). Single-value pairs are untouched.
   Reports the resulting RQ2/RQ4 deltas alongside the primary numbers, as a SEPARATE sensitivity
   check -- this alternate scorer is not swapped in as the primary scorer anywhere else.

Ad hoc script, not a permanent CLI capability, zero new API calls. Run with:
    python scripts/scorer_sensitivity.py
"""
import json
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from agentic_docs.analyze import score_run
from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.metrics import (
    FUZZY_MATCH_THRESHOLD,
    FieldScore,
    accuracy_statistic,
    align_answers_to_pairs,
    bootstrap_document_level,
    normalize,
)

RUN3_PATH = "results/extractions/run3.json"
OUT_PATH = "results/analysis/scorer_sensitivity.json"

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]
THRESHOLDS = [80.0, 85.0, 90.0, 95.0]


def threshold_sweep(rows, documents) -> dict:
    out = {}
    for threshold in THRESHOLDS:
        _, final_scores = score_run(rows, documents, threshold=threshold)
        out[str(threshold)] = {"rq2_predicted_vs_flat": {}, "rq4_shuffled_vs_flat": {}}
        for cond_key, cond in (("rq2_predicted_vs_flat", "predicted_graph"),
                                ("rq4_shuffled_vs_flat", "shuffled_graph")):
            for model in MODELS:
                scores_a = [s for s in final_scores if s.condition == cond and s.model == model]
                scores_b = [s for s in final_scores if s.condition == "flat" and s.model == model]
                if scores_a and scores_b:
                    r = bootstrap_document_level(scores_a, scores_b, accuracy_statistic,
                                                  correction_n=len(MODELS))
                    out[str(threshold)][cond_key][model] = {
                        "point_delta": r["point_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    }
    return out


def _token_multiset_match(pred: str, gold: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Order-insensitive comparator: treats both strings as whitespace-token multisets and scores
    by multiset overlap (0-100 scale, same threshold convention as fuzz.ratio) -- fixes the
    already-diagnosed limitation where a multi-value answer with 1-2 values transposed scores as a
    total mismatch under the order-dependent fuzz.ratio on the concatenated string."""
    pred_tokens, gold_tokens = normalize(pred).split(), normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return True
    if not pred_tokens or not gold_tokens:
        return False
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    score = 100.0 * overlap / max(len(pred_tokens), len(gold_tokens))
    return score >= threshold


def order_insensitive_rescore(rows, documents) -> dict[str, dict[str, list[FieldScore]]]:
    """Per-(doc, question) FieldScore stand-ins, condition x model, RESCORING only multi-value pairs
    (len(answer_entity_ids) > 1) via `_token_multiset_match`; single-value pairs keep the standard
    fuzz.ratio verdict already computed by score_run's default-threshold pass. Returns per-question
    records (not pre-averaged per-document accuracy) so the result can be fed through
    `bootstrap_document_level` and stay pooled/micro-averaged, the same convention the primary
    RQ2/RQ4 headline uses -- an earlier version of this function pre-averaged per document and used
    a macro-average bootstrap, caught before it reached the paper as an apples-to-oranges comparison
    against the primary (pooled) numbers it's meant to sensitivity-check."""
    docs_by_id = {d.doc_id: d for d in documents}
    _, final_scores = score_run(rows, documents)  # default threshold, for single-value pairs
    default_correct = {(s.doc_id, s.condition, s.model, s.question_id): s.fuzzy_match
                        for s in final_scores}

    out: dict[str, dict[str, list[FieldScore]]] = defaultdict(lambda: defaultdict(list))

    for row in rows.values():
        doc = docs_by_id.get(row["document_id"])
        if doc is None or row.get("parse_error"):
            continue
        condition, model = row["condition"], row["model"]
        excluded = set(row.get("excluded_question_ids", []))
        scorable = [p for p in doc.scorable_pairs
                    if p.question_id not in excluded or condition != "shuffled_graph"]
        question_pairs = [(p.question_id, p.question_text) for p in scorable]
        aligned = align_answers_to_pairs(question_pairs, row.get("final_answers", {}))

        for pair in scorable:
            key = (doc.doc_id, condition, model, pair.question_id)
            if key not in default_correct:
                continue
            if len(pair.answer_entity_ids) > 1:
                correct = _token_multiset_match(aligned[pair.question_id], pair.gold_answer_text)
            else:
                correct = default_correct[key]
            out[condition][model].append(FieldScore(
                doc_id=doc.doc_id, condition=condition, model=model, stage="final",
                question_id=pair.question_id, predicted_answer="", gold_answer="",
                exact_match=False, fuzzy_score=100.0 if correct else 0.0, fuzzy_match=correct,
                omission=False, hallucination=False, misassociation=False,
                relation_following_error=None,
            ))
    return out


def main():
    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(open(RUN3_PATH, encoding="utf-8").read())

    sweep = threshold_sweep(rows, documents)

    rescored = order_insensitive_rescore(rows, documents)
    order_insensitive_headline = {"rq2_predicted_vs_flat": {}, "rq4_shuffled_vs_flat": {}}
    for cond_key, cond in (("rq2_predicted_vs_flat", "predicted_graph"),
                            ("rq4_shuffled_vs_flat", "shuffled_graph")):
        for model in MODELS:
            scores_a = rescored[cond][model]
            scores_b = rescored["flat"][model]
            order_insensitive_headline[cond_key][model] = bootstrap_document_level(
                scores_a, scores_b, accuracy_statistic, correction_n=len(MODELS))

    out = {"threshold_sweep": sweep, "order_insensitive_rescore_headline": order_insensitive_headline}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
