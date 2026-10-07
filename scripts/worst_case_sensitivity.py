"""Worst-case sensitivity check for Limitations item 1: how much would each of Table 3's
headline point deltas shift if every persistent-parse-failure cell, currently excluded via the
paired-intersection convention, were instead scored as entirely wrong (fuzzy_match=False for
every one of that document's scorable questions) rather than dropped?

Four persistent-parse-failure cells exist in results/extractions/run3.json (found by grepping
parse_error=True): Claude/shuffled_graph on one document, and Qwen/predicted_graph,
Qwen/shuffled_graph, Qwen/oracle_graph all on a second, different document. Ad hoc script, zero
new API calls. Run with:
    python scripts/worst_case_sensitivity.py
"""
import json

from agentic_docs.analyze import score_run
from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.metrics import FieldScore, accuracy_statistic

RUN3_PATH = "results/extractions/run3.json"
OUT_PATH = "results/analysis/worst_case_sensitivity.json"

# (failed_condition, other_condition, model, label) for every Table 3 headline a persistent
# parse failure can affect. Oracle-Heuristic is excluded: it is an Appendix table, not Table 3,
# and Limitations item 1 scopes this check to Table~\ref{tab:bootstrap} only.
CONTRASTS = [
    ("predicted_graph", "flat", "qwen3.6-27b", "Heuristic-Flat"),
    ("oracle_graph", "flat", "qwen3.6-27b", "Oracle-Flat"),
    ("shuffled_graph", "flat", "qwen3.6-27b", "Deranged-Flat"),
    ("shuffled_graph", "flat", "claude-sonnet-5", "Deranged-Flat"),
]


def _by_doc(scores):
    out: dict[str, list] = {}
    for s in scores:
        out.setdefault(s.doc_id, []).append(s)
    return out


def main():
    documents = load_documents(DOCUMENTS_PATH)
    docs_by_id = {d.doc_id: d for d in documents}
    rows = json.loads(open(RUN3_PATH, encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    failed_cells = [(v["document_id"], v["condition"], v["model"])
                    for v in rows.values() if v.get("parse_error")]

    def get(cond, model):
        return [s for s in final_scores if s.condition == cond and s.model == model]

    out = {}
    for failed_cond, other_cond, model, label in CONTRASTS:
        by_doc_failed = _by_doc(get(failed_cond, model))
        by_doc_other = _by_doc(get(other_cond, model))
        clean_docs = sorted(set(by_doc_failed) & set(by_doc_other))
        clean_a = [s for d in clean_docs for s in by_doc_failed[d]]
        clean_b = [s for d in clean_docs for s in by_doc_other[d]]
        clean_delta = accuracy_statistic(clean_a) - accuracy_statistic(clean_b)

        worst_a, worst_b, added_docs = list(clean_a), list(clean_b), []
        for doc_id, cond, mdl in failed_cells:
            if cond != failed_cond or mdl != model or doc_id in clean_docs:
                continue
            other_doc_scores = by_doc_other.get(doc_id)
            if not other_doc_scores:
                continue  # doc failed on both sides of this contrast; cancels out, not addable
            synthetic = [FieldScore(doc_id=doc_id, condition=failed_cond, model=model,
                                     stage="final", question_id=f"synthetic_{i}",
                                     predicted_answer="", gold_answer="", exact_match=False,
                                     fuzzy_score=0.0, fuzzy_match=False, omission=True,
                                     hallucination=False, misassociation=False,
                                     relation_following_error=None)
                         for i in range(len(other_doc_scores))]
            worst_a.extend(synthetic)
            worst_b.extend(other_doc_scores)
            added_docs.append(doc_id)

        worst_delta = accuracy_statistic(worst_a) - accuracy_statistic(worst_b)
        out[f"{label}_{model}"] = {
            "clean_delta_pp": round(clean_delta * 100, 2),
            "worst_case_delta_pp": round(worst_delta * 100, 2),
            "shift_pp": round((worst_delta - clean_delta) * 100, 2),
            "n_clean_documents": len(clean_docs),
            "added_documents": added_docs,
            "added_document_scorable_questions": [len(docs_by_id[d].scorable_pairs) for d in added_docs],
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))
    max_shift = max(abs(v["shift_pp"]) for v in out.values())
    print(f"\nMax |shift| across all Table 3 headlines: {max_shift}pp")


if __name__ == "__main__":
    main()
