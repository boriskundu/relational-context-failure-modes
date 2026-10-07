"""No-edge blank-penalty and edge-wrong misassociation deltas, Round 2 vs Round 3 (2026-09-03 review
round, Section 4.4). Both reuse the ID-based bucket diagnostics already written to disk
(round2_bucket_diagnostic.json / run3_bucket_diagnostic.json -- run round2_bucket_diagnostic.py and
run3_bucket_diagnostic.py first) rather than re-deriving bucket membership a second time.

For each round, per model, restricted to (doc, question) pairs in the relevant bucket:
  - no_edge blank penalty      = blank_rate(heuristic, no_edge bucket) - blank_rate(flat, same qs)
  - edge_wrong misassoc delta  = misassoc_rate(heuristic, edge_wrong bucket) - misassoc_rate(flat, same qs)

Both rates are POOLED across all matching (doc, question) pairs in the resampled documents each
bootstrap draw -- the same convention every other headline number in this project uses (matching
`accuracy_statistic`), via FieldScore stand-ins fed through `bootstrap_document_level`, NOT a
mean-of-per-document-rates macro-average. An earlier version of this script used the latter (via a
dict-keyed bootstrap helper) and was caught and rewritten before it reached the paper -- see
cross_round_analysis.py's docstring for the same fix applied there.

Round 3 minus Round 2 (did the penalty shrink under Round 3's more neutral prompt?) is itself a
difference of two deltas, so it uses `bootstrap_interaction_contrast` directly rather than bootstrapping
each round's delta separately and subtracting point estimates by hand.

Round 1 is excluded -- its per-condition prompts weren't condition-invariant, same reasoning already
used to exclude Round 1 from RQ2/RQ4 elsewhere in the paper.

Ad hoc script, not a permanent CLI capability. Run with:
    .venv/Scripts/python scripts/round_sensitivity_analysis.py
"""
import json

from agentic_docs.metrics import (
    FieldScore,
    accuracy_statistic,
    bootstrap_document_level,
    bootstrap_interaction_contrast,
)

ROUND2_BUCKET_PATH = "results/analysis/round2_bucket_diagnostic.json"
ROUND3_BUCKET_PATH = "results/analysis/run3_bucket_diagnostic.json"
OUT_PATH = "results/analysis/round_sensitivity_analysis.json"

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]


def _score_from_row(doc_id, condition, model, question_id, is_target_class: bool) -> FieldScore:
    """Minimal FieldScore stand-in -- accuracy_statistic only reads .fuzzy_match. Same pattern as
    final_paper_numbers.py's own _score_from_row, repurposed here to carry "is this question's
    classification == the target class (blank / misassociation)" rather than "is this correct"."""
    return FieldScore(
        doc_id=doc_id, condition=condition, model=model, stage="final", question_id=question_id,
        predicted_answer="", gold_answer="", exact_match=False,
        fuzzy_score=100.0 if is_target_class else 0.0, fuzzy_match=is_target_class,
        omission=False, hallucination=False, misassociation=False, relation_following_error=None,
    )


def _bucket_scores(records: list[dict], model: str, bucket: str, target_class: str
                    ) -> tuple[list[FieldScore], list[FieldScore]]:
    """(heuristic_final_class == target_class, flat_class == target_class) FieldScore-stand-in
    lists, restricted to `bucket`-bucket (doc, question) pairs."""
    bucket_recs = [r for r in records if r["bucket"] == bucket]
    heuristic_scores = [
        _score_from_row(r["doc_id"], "predicted_graph", model, r["question_id"],
                         r["heuristic_final_class"] == target_class)
        for r in bucket_recs
    ]
    flat_scores = [
        _score_from_row(r["doc_id"], "flat", model, r["question_id"], r["flat_class"] == target_class)
        for r in bucket_recs
    ]
    return heuristic_scores, flat_scores


def main():
    bucket_data = {
        "round2": json.load(open(ROUND2_BUCKET_PATH, encoding="utf-8")),
        "round3": json.load(open(ROUND3_BUCKET_PATH, encoding="utf-8")),
    }

    out = {"no_edge_blank_penalty": {}, "edge_wrong_misassoc_delta": {}}
    scores_by_round_model = {"no_edge_blank_penalty": {"round2": {}, "round3": {}},
                              "edge_wrong_misassoc_delta": {"round2": {}, "round3": {}}}

    for out_key, bucket, target_class in (
        ("no_edge_blank_penalty", "no_edge", "blank"),
        ("edge_wrong_misassoc_delta", "edge_wrong", "misassociation"),
    ):
        for round_name in ["round2", "round3"]:
            out[out_key][round_name] = {}
            for model in MODELS:
                records = bucket_data[round_name][model]
                heuristic_scores, flat_scores = _bucket_scores(records, model, bucket, target_class)
                scores_by_round_model[out_key][round_name][model] = (heuristic_scores, flat_scores)
                out[out_key][round_name][model] = bootstrap_document_level(
                    heuristic_scores, flat_scores, accuracy_statistic, correction_n=len(MODELS))

        out[out_key]["round3_minus_round2"] = {}
        for model in MODELS:
            r3_heuristic, r3_flat = scores_by_round_model[out_key]["round3"][model]
            r2_heuristic, r2_flat = scores_by_round_model[out_key]["round2"][model]
            out[out_key]["round3_minus_round2"][model] = bootstrap_interaction_contrast(
                r3_heuristic, r3_flat, r2_heuristic, r2_flat, accuracy_statistic,
                correction_n=len(MODELS))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
