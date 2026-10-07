"""Per-document heuristic edge precision and coverage, correlated against the per-document
Heuristic-Flat accuracy delta (2026-09-03 review round). Descriptive/observational only -- this
does NOT manipulate precision or coverage independently (both reviews' explicit caution against
overclaiming this as a causal "quality curve"), it asks whether documents where the heuristic
happens to be more precise, or happens to cover more of the question set, also happen to show a
larger or smaller accuracy delta.

`predict_links`/`_gold_edges` are already per-document functions (heuristic/proximity.py,
heuristic/validate.py) -- this script is the first thing to loop them across the full corpus and
persist a per-document table, joined against Round 3's already-scored per-document accuracy delta.
Zero new API calls. Correlation is a hand-rolled Spearman rho (see spearman_rho below) rather than
scipy.stats -- this environment blocks one of scipy.stats' compiled DLLs via a Windows Application
Control policy.

Ad hoc script, not a permanent CLI capability. Run with:
    .venv/Scripts/python scripts/precision_coverage_regression.py
"""
import json

from agentic_docs.analyze import score_run
from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.heuristic.proximity import predict_links
from agentic_docs.heuristic.validate import _gold_edges

RUN3_PATH = "results/extractions/run3.json"
OUT_PATH = "results/analysis/precision_coverage_regression.json"

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]


def _ranks(values: list[float]) -> list[float]:
    """1-based ranks, ties broken by average rank (standard Spearman convention)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Pure-Python Spearman rank correlation -- this environment's scipy.stats import is blocked
    by a Windows Application Control policy on one of its compiled DLLs (unrelated to this
    project's own code), so this avoids the dependency entirely rather than working around it.
    No p-value is computed (would need a t-distribution CDF); this analysis is reported as
    descriptive/observational only, per both reviews' explicit caution against a causal framing,
    so rho and n are what's actually used in the paper."""
    n = len(xs)
    if n < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def per_doc_covariates(documents) -> dict[str, dict]:
    """Per-document heuristic precision (of the edges it predicted, how many are gold) and
    coverage (what fraction of scorable questions have at least one predicted edge)."""
    out = {}
    for doc in documents:
        gold = _gold_edges(doc)
        predicted = set(predict_links(doc))
        n_questions = len(doc.scorable_pairs)
        question_ids = {p.question_id for p in doc.scorable_pairs}
        covered = {q for edge in predicted for q in edge if q in question_ids}

        precision = (len(gold & predicted) / len(predicted)) if predicted else None
        coverage = (len(covered) / n_questions) if n_questions else None
        out[doc.doc_id] = {"precision": precision, "coverage": coverage,
                            "n_gold_edges": len(gold), "n_predicted_edges": len(predicted)}
    return out


def _per_doc_accuracy(final_scores, condition: str, model: str) -> dict[str, float]:
    by_doc: dict[str, list[bool]] = {}
    for s in final_scores:
        if s.condition == condition and s.model == model:
            by_doc.setdefault(s.doc_id, []).append(s.fuzzy_match)
    return {doc_id: sum(vals) / len(vals) for doc_id, vals in by_doc.items()}


def main():
    documents = load_documents(DOCUMENTS_PATH)
    covariates = per_doc_covariates(documents)

    rows = json.loads(open(RUN3_PATH, encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    out = {"per_document_covariates": covariates, "correlations": {}}
    for model in MODELS:
        flat_acc = _per_doc_accuracy(final_scores, "flat", model)
        heuristic_acc = _per_doc_accuracy(final_scores, "predicted_graph", model)
        common = set(flat_acc) & set(heuristic_acc)
        delta = {d: heuristic_acc[d] - flat_acc[d] for d in common}

        precision_pairs = [(covariates[d]["precision"], delta[d]) for d in common
                            if covariates[d]["precision"] is not None]
        coverage_pairs = [(covariates[d]["coverage"], delta[d]) for d in common
                           if covariates[d]["coverage"] is not None]

        prec_rho = spearman_rho([p for p, _ in precision_pairs], [d for _, d in precision_pairs])
        cov_rho = spearman_rho([c for c, _ in coverage_pairs], [d for _, d in coverage_pairs])

        out["correlations"][model] = {
            "n_documents": len(common),
            "precision_vs_delta": {"spearman_rho": prec_rho, "n": len(precision_pairs)},
            "coverage_vs_delta": {"spearman_rho": cov_rho, "n": len(coverage_pairs)},
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out["correlations"], indent=2))


if __name__ == "__main__":
    main()
