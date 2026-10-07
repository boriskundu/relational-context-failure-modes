"""Diagnostic for Run3: edge_correct/edge_wrong/no_edge bucket stratification of Heuristic (predicted_graph) vs Flat, error taxonomy +
initial->final by bucket, and a worst-case sensitivity check on the 4 persistent failed cells.

Not a permanent CLI capability -- ad hoc script reusing existing scoring/representation functions,
same pattern as the Smoke Test A/B ad hoc runs. Run with:
    python scripts/run3_bucket_diagnostic.py
"""
import json
from collections import defaultdict

from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.metrics import FUZZY_MATCH_THRESHOLD, normalize
from agentic_docs.representations._common import shown_answer_text_by_question
from agentic_docs.runner import build_representation, cell_key
from rapidfuzz import fuzz

RUN3_PATH = "results/extractions/run3.json"
MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]


def bucket_for(shown_text: str | None, gold_text: str) -> str:
    if shown_text is None:
        return "no_edge"
    shown_norm, gold_norm = normalize(shown_text), normalize(gold_text)
    score = fuzz.ratio(shown_norm, gold_norm) if (shown_norm or gold_norm) else 100.0
    return "edge_correct" if score >= FUZZY_MATCH_THRESHOLD else "edge_wrong"


def is_correct(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    score = fuzz.ratio(p, g) if (p or g) else 100.0
    return score >= FUZZY_MATCH_THRESHOLD


def is_blank(pred: str) -> bool:
    return not normalize(pred)


def align(question_pairs, answers: dict) -> dict[int, str]:
    """Mirror of metrics.align_answers_to_pairs (greedy one-to-one fuzzy match by score),
    duplicated here read-only to avoid importing analyze.py's private helpers."""
    from agentic_docs.metrics import align_answers_to_pairs
    return align_answers_to_pairs(question_pairs, answers)


def compute_bucket_records(run_path: str) -> dict[str, list[dict]]:
    """Same bucket stratification logic as Run3's original diagnostic, factored out so it
    can be re-run against ANY round's raw extraction JSON (representations/bucket membership are
    pure functions of the document, not the round -- only the model rows themselves differ)."""
    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(open(run_path, encoding="utf-8").read())

    # bucket_records[model] -> list of dicts with doc_id, question_id, bucket, flat_correct,
    # flat_blank, heuristic_correct_initial, heuristic_correct_final, heuristic_blank_final,
    # heuristic_misassoc_final, heuristic_halluc_final
    per_model_rows = defaultdict(list)

    for model in MODELS:
        flat_key_prefix = None
        for doc in documents:
            flat_row = rows.get(cell_key(doc.doc_id, "flat", model))
            pred_row = rows.get(cell_key(doc.doc_id, "predicted_graph", model))
            if flat_row is None or pred_row is None:
                continue
            if flat_row.get("parse_error") or pred_row.get("parse_error"):
                continue  # excluded cells handled separately in the sensitivity check

            representation, _ = build_representation("predicted_graph", doc)
            shown_by_q = shown_answer_text_by_question(doc, representation["links"])

            excluded_pred = set(pred_row.get("excluded_question_ids", []))
            scorable = [p for p in doc.scorable_pairs if p.question_id not in excluded_pred]
            question_pairs = [(p.question_id, p.question_text) for p in scorable]

            flat_final_aligned = align(question_pairs, flat_row.get("final_answers", {}))
            pred_initial_aligned = align(question_pairs, pred_row.get("initial_answers", {}))
            pred_final_aligned = align(question_pairs, pred_row.get("final_answers", {}))

            for pair in scorable:
                qid = pair.question_id
                gold = pair.gold_answer_text
                bucket = bucket_for(shown_by_q.get(qid), gold)

                flat_pred = flat_final_aligned.get(qid, "")
                pred_init = pred_initial_aligned.get(qid, "")
                pred_final = pred_final_aligned.get(qid, "")

                other_texts = [e.text for e in doc.entities if e.text and e.id not in pair.answer_entity_ids]

                def classify(pred_text):
                    p = normalize(pred_text)
                    if is_correct(pred_text, gold):
                        return "correct"
                    if not p:
                        return "blank"
                    matches_other = any(
                        fuzz.ratio(p, normalize(o)) >= FUZZY_MATCH_THRESHOLD
                        for o in other_texts if normalize(o))
                    return "misassociation" if matches_other else "hallucination"

                per_model_rows[model].append({
                    "doc_id": doc.doc_id,
                    "question_id": qid,
                    "bucket": bucket,
                    "flat_class": classify(flat_pred),
                    "heuristic_initial_class": classify(pred_init),
                    "heuristic_final_class": classify(pred_final),
                })

    return per_model_rows


def main():
    per_model_rows = compute_bucket_records(RUN3_PATH)

    # ---- Report 1: bucket-stratified accuracy, Heuristic vs Flat ----
    print("=" * 100)
    print("BUCKET STRATIFICATION: Heuristic (predicted_graph) vs Flat, matched (doc,question), Run3")
    print("=" * 100)
    for model in MODELS:
        recs = per_model_rows[model]
        print(f"\n--- {model} (n={len(recs)}) ---")
        for bucket in ["edge_correct", "edge_wrong", "no_edge"]:
            bucket_recs = [r for r in recs if r["bucket"] == bucket]
            n = len(bucket_recs)
            if n == 0:
                print(f"  {bucket}: n=0")
                continue
            flat_acc = sum(r["flat_class"] == "correct" for r in bucket_recs) / n
            heur_final_acc = sum(r["heuristic_final_class"] == "correct" for r in bucket_recs) / n
            heur_init_acc = sum(r["heuristic_initial_class"] == "correct" for r in bucket_recs) / n
            print(f"  {bucket}: n={n} ({n/len(recs)*100:.1f}% of scorable) | "
                  f"flat_acc={flat_acc*100:.1f}% | heuristic_initial_acc={heur_init_acc*100:.1f}% | "
                  f"heuristic_final_acc={heur_final_acc*100:.1f}% | delta(heur_final-flat)={( heur_final_acc-flat_acc)*100:+.1f}pp")

    # ---- Report 2: error taxonomy by bucket (final stage), heuristic condition ----
    print()
    print("=" * 100)
    print("ERROR TAXONOMY BY BUCKET (Heuristic, final stage), Run3")
    print("=" * 100)
    for model in MODELS:
        recs = per_model_rows[model]
        print(f"\n--- {model} ---")
        for bucket in ["edge_correct", "edge_wrong", "no_edge"]:
            bucket_recs = [r for r in recs if r["bucket"] == bucket]
            n = len(bucket_recs)
            if n == 0:
                continue
            counts = defaultdict(int)
            for r in bucket_recs:
                counts[r["heuristic_final_class"]] += 1
            parts = ", ".join(f"{k}={v/n*100:.1f}%" for k, v in sorted(counts.items()))
            print(f"  {bucket} (n={n}): {parts}")

    # ---- Report 3: initial -> final transition by bucket ----
    print()
    print("=" * 100)
    print("INITIAL -> FINAL CORRECTION/REGRESSION BY BUCKET (Heuristic), Run3")
    print("=" * 100)
    for model in MODELS:
        recs = per_model_rows[model]
        print(f"\n--- {model} ---")
        for bucket in ["edge_correct", "edge_wrong", "no_edge"]:
            bucket_recs = [r for r in recs if r["bucket"] == bucket]
            if not bucket_recs:
                continue
            init_wrong = [r for r in bucket_recs if r["heuristic_initial_class"] != "correct"]
            init_correct = [r for r in bucket_recs if r["heuristic_initial_class"] == "correct"]
            corrected = sum(1 for r in init_wrong if r["heuristic_final_class"] == "correct")
            regressed = sum(1 for r in init_correct if r["heuristic_final_class"] != "correct")
            corr_rate = corrected / len(init_wrong) if init_wrong else None
            reg_rate = regressed / len(init_correct) if init_correct else None
            corr_str = f"{corr_rate*100:.1f}%" if corr_rate is not None else "n/a"
            reg_str = f"{reg_rate*100:.1f}%" if reg_rate is not None else "n/a"
            print(f"  {bucket}: n_initially_wrong={len(init_wrong)} correction_rate={corr_str}  "
                  f"n_initially_correct={len(init_correct)} regression_rate={reg_str}")

    with open("results/analysis/run3_bucket_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump({m: per_model_rows[m] for m in MODELS}, f, indent=2)
    print("\nWrote results/analysis/run3_bucket_diagnostic.json (raw per-question records)")


if __name__ == "__main__":
    main()
