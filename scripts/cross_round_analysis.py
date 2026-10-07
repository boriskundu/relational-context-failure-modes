"""Bootstrap CIs on the Round3-minus-Round1 wrong-edge-consistency (relation_following_error_rate)
deltas, per model, for the deranged (shuffled_graph) and heuristic (predicted_graph) conditions.

Section 4.4 currently reports this trajectory as bare percentage ranges pulled from each summary's
stored `by_condition_model` aggregate -- an eyeballed ordered pattern, not an interval estimate. This
script bootstraps the SAME pooled statistic (relation_following_error_rate_statistic, matching
`aggregate`'s own convention exactly -- pooled over all applicable questions across the resampled
documents, NOT the mean of each document's own rate) via `bootstrap_document_level`, the same
resampling machinery every other headline number in this project already uses. Zero new API calls.

An earlier version of this script used a mean-of-per-document-rates (macro-average) via a
dict-keyed bootstrap helper -- caught before being used in the paper as inconsistent with every
other reported number's pooled/micro convention (a document with many applicable questions and one
with few were being weighted equally instead of by their actual question count), and rewritten to
match.

Ad hoc script, not a permanent CLI capability. Run with:
    .venv/Scripts/python scripts/cross_round_analysis.py
"""
import json

from agentic_docs.analyze import score_run
from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.metrics import bootstrap_document_level, relation_following_error_rate_statistic

RUN1_PATH = "results/extractions/run.json"
RUN3_PATH = "results/extractions/run3.json"
OUT_PATH = "results/analysis/cross_round_analysis.json"

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]
CONDITIONS = ["shuffled_graph", "predicted_graph"]


def main():
    documents = load_documents(DOCUMENTS_PATH)
    round1_rows = json.loads(open(RUN1_PATH, encoding="utf-8").read())
    round3_rows = json.loads(open(RUN3_PATH, encoding="utf-8").read())
    _, round1_final = score_run(round1_rows, documents)
    _, round3_final = score_run(round3_rows, documents)

    out = {}
    for condition in CONDITIONS:
        out[condition] = {}
        for model in MODELS:
            scores_round1 = [s for s in round1_final if s.condition == condition and s.model == model]
            scores_round3 = [s for s in round3_final if s.condition == condition and s.model == model]
            out[condition][model] = bootstrap_document_level(
                scores_round3, scores_round1, relation_following_error_rate_statistic,
                correction_n=len(MODELS))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
