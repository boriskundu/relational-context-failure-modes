"""
Joins runner output rows against the document gold data and computes the full metrics taxonomy.
Scores against the single combined 199-document dataset (no separate frozen test split -- see
config.DOCUMENTS_PATH).
"""

import argparse
import json
from pathlib import Path

from agentic_docs.config import ANALYSIS_DIR, DOCUMENTS_PATH, EXTRACTIONS_DIR
from agentic_docs.funsd.parse import DocumentRecord, load_documents
from agentic_docs.metrics import (
    FUZZY_MATCH_THRESHOLD,
    FieldScore,
    accuracy_statistic,
    aggregate,
    align_answers_to_pairs,
    bootstrap_document_level,
    correction_regression,
    score_field,
)
from agentic_docs.representations._common import shown_answer_text_by_question
from agentic_docs.runner import build_representation

GRAPH_CONDITIONS = {"predicted_graph", "oracle_graph", "shuffled_graph"}


def _other_source_texts(doc: DocumentRecord, exclude_entity_ids: tuple[int, ...]) -> list[str]:
    """Every other entity's text in the document, excluding this pair's own gold answer entities BY
    ID. Excluding by the concatenated gold_answer_text string instead (an earlier bug) misses each
    individual constituent entity of a multi-answer pair -- e.g. gold "Bob Smith" concatenated from
    entities "Bob" + "Smith" would leak "Bob" and "Smith" into this list, causing a genuine partial
    match of the correct field (an agent answering just "Bob") to be misclassified as misassociation
    (a real value from a DIFFERENT field) instead of an incomplete match of the same field."""
    return [e.text for e in doc.entities if e.text and e.id not in exclude_entity_ids]


def score_run(
    rows: dict,
    documents: list[DocumentRecord],
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> tuple[list[FieldScore], list[FieldScore]]:
    """Join runner rows against document gold data. Returns (initial_scores, final_scores) — one
    FieldScore per (row, scorable_pair), skipping rows with a parse error and, for shuffled_graph
    specifically, questions whose edge couldn't be deranged (Key decision #3).

    ``threshold`` overrides the fuzzy-match ruler passed to every ``score_field`` call -- used only
    by the scorer-sensitivity sweep (scripts/scorer_sensitivity.py); every other caller relies on the
    default (the paper's primary scoring rule)."""
    docs_by_id = {d.doc_id: d for d in documents}
    initial_scores: list[FieldScore] = []
    final_scores: list[FieldScore] = []
    # (doc_id, condition) -> {question_id: shown_answer_text} -- independent of model/stage, so
    # rebuilt once per document/condition rather than once per row (representations are pure
    # functions of (doc, condition, the fixed RANDOM_SEED), safe to reconstruct here).
    shown_answers_cache: dict[tuple[str, str], dict[int, str]] = {}

    for row in rows.values():
        doc = docs_by_id.get(row["document_id"])
        if doc is None or row.get("parse_error"):
            continue
        condition = row["condition"]
        excluded = set(row.get("excluded_question_ids", []))
        scorable_pairs = [
            p
            for p in doc.scorable_pairs
            if p.question_id not in excluded or condition != "shuffled_graph"
        ]
        question_pairs = [(p.question_id, p.question_text) for p in scorable_pairs]

        shown_by_question: dict[int, str] = {}
        if condition in GRAPH_CONDITIONS:
            cache_key = (doc.doc_id, condition)
            if cache_key not in shown_answers_cache:
                representation, recomputed_excluded = build_representation(condition, doc)
                # Sanity check, not just an assumption: this reconstruction is only valid if it's
                # bit-for-bit identical to what runner.py built at run time (same doc, same fixed
                # RANDOM_SEED). Cheap to verify directly against the row's own stored value rather
                # than silently trusting determinism across a run/analyze time gap.
                if recomputed_excluded != excluded and condition == "shuffled_graph":
                    raise RuntimeError(
                        f"Reconstructed excluded_question_ids {sorted(recomputed_excluded)} for "
                        f"{doc.doc_id}/{condition} does not match the row's stored "
                        f"{sorted(excluded)} -- representation reconstruction at analyze time is "
                        f"no longer reproducing the run-time representation (e.g. RANDOM_SEED or "
                        f"the derangement algorithm changed); relation_following_error scoring for "
                        f"this condition is unsafe until this is resolved."
                    )
                shown_answers_cache[cache_key] = shown_answer_text_by_question(
                    doc,
                    representation["links"],  # type: ignore[index]
                )
            shown_by_question = shown_answers_cache[cache_key]

        for stage, answers, bucket in (
            ("initial", row.get("initial_answers", {}), initial_scores),
            ("final", row.get("final_answers", {}), final_scores),
        ):
            aligned = align_answers_to_pairs(question_pairs, answers)
            for pair in scorable_pairs:
                others = _other_source_texts(doc, pair.answer_entity_ids)
                predicted = aligned[pair.question_id]
                bucket.append(
                    score_field(
                        doc.doc_id,
                        condition,
                        row["model"],
                        stage,
                        pair.question_id,
                        predicted,
                        pair.gold_answer_text,
                        others,
                        shown_answer_text=shown_by_question.get(pair.question_id),
                        threshold=threshold,
                    )
                )
    return initial_scores, final_scores


def build_summary(initial_scores: list[FieldScore], final_scores: list[FieldScore]) -> dict:
    conditions = sorted({s.condition for s in final_scores})
    models = sorted({s.model for s in final_scores})

    by_condition_model = {}
    rq5_correction = {}
    for cond in conditions:
        for model in models:
            key = f"{cond}|{model}"
            cond_model_final = [s for s in final_scores if s.condition == cond and s.model == model]
            cond_model_initial = [
                s for s in initial_scores if s.condition == cond and s.model == model
            ]
            by_condition_model[key] = aggregate(cond_model_final)
            rq5_correction[key] = correction_regression(cond_model_initial, cond_model_final)

    def _headline(cond_a: str, cond_b: str) -> dict:
        """Document-level paired bootstrap of cond_a vs cond_b, once per model. Bonferroni-corrected
        for testing the same hypothesis across all `len(models)` models (Group A item 11)."""
        result = {}
        for model in models:
            scores_a = [s for s in final_scores if s.condition == cond_a and s.model == model]
            scores_b = [s for s in final_scores if s.condition == cond_b and s.model == model]
            if scores_a and scores_b:
                result[model] = bootstrap_document_level(
                    scores_a, scores_b, accuracy_statistic, correction_n=len(models)
                )
        return result

    rq2_headline = _headline("predicted_graph", "flat")
    rq3_headline = _headline("oracle_graph", "predicted_graph")
    rq4_headline = _headline("shuffled_graph", "flat")

    return {
        "n_initial_scores": len(initial_scores),
        "n_final_scores": len(final_scores),
        "by_condition_model": by_condition_model,
        "rq5_correction": rq5_correction,
        "rq2_headline_predicted_vs_flat": rq2_headline,
        "rq3_headline_oracle_vs_predicted": rq3_headline,
        "rq4_headline_shuffled_vs_flat": rq4_headline,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docs analyze")
    parser.add_argument(
        "--run", default=None, help="Path to the run output JSON (default inferred)"
    )
    args = parser.parse_args(argv)

    run_path = Path(args.run) if args.run else EXTRACTIONS_DIR / "run.json"

    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(run_path.read_text(encoding="utf-8"))
    initial_scores, final_scores = score_run(rows, documents)
    summary = build_summary(initial_scores, final_scores)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for key, agg in summary["by_condition_model"].items():
        print(f"  {key}: accuracy={agg['accuracy']}, n={agg['n']}")
    for rq_name, key, cond_pair in (
        ("RQ2", "rq2_headline_predicted_vs_flat", "predicted_graph vs flat"),
        ("RQ3", "rq3_headline_oracle_vs_predicted", "oracle_graph vs predicted_graph"),
        ("RQ4", "rq4_headline_shuffled_vs_flat", "shuffled_graph vs flat"),
    ):
        for model, headline in summary.get(key, {}).items():
            dropped = headline.get("n_dropped_one_sided", 0)
            if dropped:
                print(
                    f"  [warn] {rq_name}/{model}: dropped {dropped} document(s) present on only "
                    f"one side ({cond_pair}) -- likely a parse error under one condition"
                )


if __name__ == "__main__":
    main()
