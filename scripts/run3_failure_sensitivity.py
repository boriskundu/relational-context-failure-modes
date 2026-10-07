"""Worst-case sensitivity check for Run3's 4 persistent parse-error cells (see Limitations):
does treating each failed (doc, condition, model) cell as 100% wrong instead of
excluding the document from that comparison change the RQ2/RQ4 headline deltas meaningfully?

Ad hoc script, not a permanent CLI capability. Run with:
    python scripts/run3_failure_sensitivity.py
"""
import json

from agentic_docs.analyze import score_run
from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.metrics import FieldScore, accuracy_statistic, bootstrap_document_level

RUN3_PATH = "results/extractions/run3.json"

# (doc_id, condition, model) for the 4 persistent failures, confirmed in logs/run3*.log
FAILED_CELLS = [
    ("82253362_3364", "shuffled_graph", "claude-sonnet-5"),
    ("0000999294", "predicted_graph", "qwen3.6-27b"),
    ("0000999294", "shuffled_graph", "qwen3.6-27b"),
    ("0000999294", "oracle_graph", "qwen3.6-27b"),
]


def worst_case_scores_for(doc, condition, model) -> list[FieldScore]:
    return [
        FieldScore(
            doc_id=doc.doc_id, condition=condition, model=model, stage="final",
            question_id=p.question_id, predicted_answer="", gold_answer=p.gold_answer_text,
            exact_match=False, fuzzy_score=0.0, fuzzy_match=False,
            omission=True, hallucination=False, misassociation=False,
            relation_following_error=None,
        )
        for p in doc.scorable_pairs
    ]


def main():
    documents = load_documents(DOCUMENTS_PATH)
    docs_by_id = {d.doc_id: d for d in documents}
    rows = json.loads(open(RUN3_PATH, encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    augmented = list(final_scores)
    for doc_id, condition, model in FAILED_CELLS:
        augmented.extend(worst_case_scores_for(docs_by_id[doc_id], condition, model))

    def headline(scores, cond_a, cond_b, model):
        a = [s for s in scores if s.condition == cond_a and s.model == model]
        b = [s for s in scores if s.condition == cond_b and s.model == model]
        return bootstrap_document_level(a, b, accuracy_statistic, correction_n=4)

    print("=" * 100)
    print("RQ2 (predicted_graph vs flat) -- observed-case vs worst-case, affected models only")
    print("=" * 100)
    for model in ["qwen3.6-27b"]:
        obs = headline(final_scores, "predicted_graph", "flat", model)
        wc = headline(augmented, "predicted_graph", "flat", model)
        print(f"{model}:")
        print(f"  observed:   delta={obs['point_delta']*100:+.1f}pp  "
              f"bonf_ci=({obs['ci_low_bonferroni']*100:+.1f}, {obs['ci_high_bonferroni']*100:+.1f})  "
              f"n_docs={obs['n_documents']} n_dropped={obs['n_dropped_one_sided']}")
        print(f"  worst-case: delta={wc['point_delta']*100:+.1f}pp  "
              f"bonf_ci=({wc['ci_low_bonferroni']*100:+.1f}, {wc['ci_high_bonferroni']*100:+.1f})  "
              f"n_docs={wc['n_documents']} n_dropped={wc['n_dropped_one_sided']}")

    print()
    print("=" * 100)
    print("RQ4 (shuffled_graph vs flat) -- observed-case vs worst-case, affected models only")
    print("=" * 100)
    for model in ["claude-sonnet-5", "qwen3.6-27b"]:
        obs = headline(final_scores, "shuffled_graph", "flat", model)
        wc = headline(augmented, "shuffled_graph", "flat", model)
        print(f"{model}:")
        print(f"  observed:   delta={obs['point_delta']*100:+.1f}pp  "
              f"bonf_ci=({obs['ci_low_bonferroni']*100:+.1f}, {obs['ci_high_bonferroni']*100:+.1f})  "
              f"n_docs={obs['n_documents']} n_dropped={obs['n_dropped_one_sided']}")
        print(f"  worst-case: delta={wc['point_delta']*100:+.1f}pp  "
              f"bonf_ci=({wc['ci_low_bonferroni']*100:+.1f}, {wc['ci_high_bonferroni']*100:+.1f})  "
              f"n_docs={wc['n_documents']} n_dropped={wc['n_dropped_one_sided']}")


if __name__ == "__main__":
    main()
