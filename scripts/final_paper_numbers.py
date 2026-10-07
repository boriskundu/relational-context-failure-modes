"""One-off consolidation script for the paper rewrite (2026-09-03), per the two external reviewers'
"Phase 1 -- Analysis cleanup" lists. Computes every NEW number the rewrite needs that isn't already
sitting in a summary*.json, from already-collected data only (zero new API calls):

1. RQ1 (Raw vs Flat) headline bootstrap, Run2 only, at n_boot=20000 with Bonferroni n=4 -- no stored
   field exists for this pair (build_summary only computes predicted/oracle/shuffled-vs-flat).
2. RQ3b (Oracle vs Flat) headline bootstrap, Run3 only -- same reason, a new pairwise contrast.
3. Document-bootstrap CIs on (Heuristic_final - Flat) within each of the three ID-based buckets
   (edge_correct / edge_wrong / no_edge), per model, from the already-saved
   results/analysis/run3_bucket_diagnostic.json per-question records.
4. Heuristic's (predicted_graph) own wrong-edge-consistency (relation_following_error_rate)
   trajectory across Run1/Run2/Run3 -- pulled from each summary's by_condition_model, not
   recomputed (aggregate() already stores it).
5. Heuristic F1 on the 35-doc validation holdout AND its 164-doc complement, to show it isn't
   overfit to the small holdout it was originally checked against.
6. GPT-5-vs-other-model interaction contrasts (2026-09-03 review round): a formal
   difference-in-differences test of whether GPT-5's own Heuristic-Flat/Deranged-Flat delta is
   itself different from each other model's same delta, not just individually significant against
   zero -- the actual statistical test the "GPT-5 is uniquely robust" claim needs.

Ad hoc script, not a permanent CLI capability, same pattern as run3_bucket_diagnostic.py. Run with:
    .venv/Scripts/python scripts/final_paper_numbers.py
"""
import json
import random
from collections import defaultdict

from agentic_docs.config import DOCUMENTS_PATH
from agentic_docs.funsd.parse import load_documents
from agentic_docs.heuristic.validate import _gold_edges, select_holdout
from agentic_docs.heuristic.proximity import predict_links
from agentic_docs.metrics import FieldScore, accuracy_statistic, bootstrap_document_level

RUN1_PATH = "results/extractions/run.json"
RUN2_PATH = "results/extractions/run_condition_blind.json"
SUMMARY1_PATH = "results/analysis/summary.json"
SUMMARY2_PATH = "results/analysis/summary_condition_blind.json"
SUMMARY3_PATH = "results/analysis/summary_run3.json"
BUCKET_PATH = "results/analysis/run3_bucket_diagnostic.json"
OUT_PATH = "results/analysis/final_paper_numbers.json"

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]


def _score_from_row(doc_id, condition, model, stage, question_id, fuzzy_match: bool) -> FieldScore:
    """Minimal FieldScore stand-in -- accuracy_statistic only reads .fuzzy_match, so the other
    fields are filled with harmless placeholders. Used to feed bootstrap_document_level from
    pre-aggregated per-question correctness (run3_bucket_diagnostic.json) rather than re-deriving
    correctness from raw answer strings a second time with a second, potentially-diverging ruler."""
    return FieldScore(
        doc_id=doc_id, condition=condition, model=model, stage=stage, question_id=question_id,
        predicted_answer="", gold_answer="", exact_match=False, fuzzy_score=100.0 if fuzzy_match else 0.0,
        fuzzy_match=fuzzy_match, omission=False, hallucination=False, misassociation=False,
        relation_following_error=None,
    )


def rq1_raw_vs_flat():
    """Raw vs Flat, Run2 (condition-blind) only -- the first fully condition-invariant version of
    this comparison. Reuses analyze.score_run against run_condition_blind.json directly rather than
    the stored summary (which has no raw-vs-flat headline field)."""
    from agentic_docs.analyze import score_run
    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(open(RUN2_PATH, encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    result = {}
    for model in MODELS:
        scores_raw = [s for s in final_scores if s.condition == "raw" and s.model == model]
        scores_flat = [s for s in final_scores if s.condition == "flat" and s.model == model]
        result[model] = bootstrap_document_level(scores_raw, scores_flat, accuracy_statistic, correction_n=4)
    return result


def rq3b_oracle_vs_flat():
    """Oracle vs Flat, Run3 only -- a new standalone contrast (Feedback item: "gold relations help
    every model" independent of whether Heuristic collapses)."""
    from agentic_docs.analyze import score_run
    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(open(RUN1_PATH.replace("run.json", "run3.json"), encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    result = {}
    for model in MODELS:
        scores_oracle = [s for s in final_scores if s.condition == "oracle_graph" and s.model == model]
        scores_flat = [s for s in final_scores if s.condition == "flat" and s.model == model]
        result[model] = bootstrap_document_level(scores_oracle, scores_flat, accuracy_statistic, correction_n=4)
    return result


def bucket_cis():
    """Document-bootstrap CI on (Heuristic_final - Flat) within each ID-based bucket, per model,
    from the already-saved per-question bucket records. Same resampling unit (document) and same
    seed/n_boot as every other headline in this project, just applied to a filtered subset."""
    raw = json.load(open(BUCKET_PATH, encoding="utf-8"))
    result = {}
    for model in MODELS:
        recs = raw[model]
        result[model] = {}
        for bucket in ["edge_correct", "edge_wrong", "no_edge"]:
            bucket_recs = [r for r in recs if r["bucket"] == bucket]
            scores_heur = [
                _score_from_row(r["doc_id"], "predicted_graph", model, "final", r["question_id"],
                                 r["heuristic_final_class"] == "correct")
                for r in bucket_recs
            ]
            scores_flat = [
                _score_from_row(r["doc_id"], "flat", model, "final", r["question_id"],
                                 r["flat_class"] == "correct")
                for r in bucket_recs
            ]
            result[model][bucket] = bootstrap_document_level(
                scores_heur, scores_flat, accuracy_statistic, correction_n=4 * 3)
    return result


def heuristic_wrong_edge_trajectory():
    """Heuristic's (predicted_graph) own relation_following_error_rate across all 3 runs, alongside
    Deranged's (already reported) -- pulled straight from stored by_condition_model, not recomputed."""
    out = {"predicted_graph": {}, "shuffled_graph": {}}
    for run_name, path in [("run1", SUMMARY1_PATH), ("run2", SUMMARY2_PATH), ("run3", SUMMARY3_PATH)]:
        summary = json.load(open(path, encoding="utf-8"))
        for cond in ["predicted_graph", "shuffled_graph"]:
            for model in MODELS:
                key = f"{cond}|{model}"
                agg = summary["by_condition_model"].get(key, {})
                out[cond].setdefault(model, {})[run_name] = {
                    "rate": agg.get("relation_following_error_rate"),
                    "n_applicable": agg.get("n_relation_following_applicable"),
                }
    return out


def gpt5_interaction_contrasts():
    """Formal difference-in-differences test of the "GPT-5 is uniquely robust" claim: is GPT-5's own
    (Heuristic-Flat) or (Deranged-Flat) delta itself different from each other model's same delta,
    not just individually significant-or-not against zero? For each of the 3 model pairs (GPT-5 vs
    Claude/Qwen/Gemini) x 2 contrasts (Heuristic-Flat, Deranged-Flat) = 6 tests, bootstrap via
    bootstrap_interaction_contrast, correction_n=6 across the whole family. Uses the FieldScore
    lists directly (pooled/micro convention, resampling one shared set of documents per draw across
    all four lists), NOT a mean-of-per-document-deltas macro-average -- an earlier version of this
    function used bootstrap_paired_values on pre-averaged per-document deltas, caught before it
    reached the paper as inconsistent with every other reported number's pooled convention."""
    from agentic_docs.analyze import score_run
    from agentic_docs.metrics import accuracy_statistic, bootstrap_interaction_contrast
    documents = load_documents(DOCUMENTS_PATH)
    rows = json.loads(open(RUN1_PATH.replace("run.json", "run3.json"), encoding="utf-8").read())
    _, final_scores = score_run(rows, documents)

    by_condition_model = {}
    for model in MODELS:
        for cond in ["flat", "predicted_graph", "shuffled_graph"]:
            by_condition_model[(cond, model)] = [
                s for s in final_scores if s.condition == cond and s.model == model]

    other_models = [m for m in MODELS if m != "gpt-5"]
    result = {"heuristic_minus_flat": {}, "deranged_minus_flat": {}}
    for contrast_name, cond in (("heuristic_minus_flat", "predicted_graph"),
                                 ("deranged_minus_flat", "shuffled_graph")):
        for other in other_models:
            result[contrast_name][f"gpt-5_minus_{other}"] = bootstrap_interaction_contrast(
                by_condition_model[(cond, "gpt-5")], by_condition_model[("flat", "gpt-5")],
                by_condition_model[(cond, other)], by_condition_model[("flat", other)],
                accuracy_statistic, correction_n=6)
    return result


def coverage_correctness_table(bucket_summary_result, f1_result):
    """Assemble the coverage/correctness reframe table (2026-09-03 review round): Flat, Heuristic,
    Oracle, and Deranged span two conflated factors -- how much of the question set gets an edge at
    all (coverage), and how often that edge is right (correctness) -- entirely from numbers already
    computed above/elsewhere in the paper, never a new experiment. Deranged's own near-zero
    correctness (7 of 2,609 edges technically identity-correct by chance, 84 more a text-coincidence
    with a repeated value, both already disclosed in the paper's Sec 4.2/Limitations) is a fixed
    property of the derangement audit, not recomputed here -- see main.tex line ~211-213/534."""
    correct_pcts = [bucket_summary_result[m]["edge_correct"]["pct_of_scorable"] for m in MODELS]
    wrong_pcts = [bucket_summary_result[m]["edge_wrong"]["pct_of_scorable"] for m in MODELS]
    coverage_pcts = [c + w for c, w in zip(correct_pcts, wrong_pcts)]
    f1s = [f1_result["holdout_35"]["f1"], f1_result["complement_164"]["f1"]]
    return {
        "flat": {"coverage": 0.0, "correctness": None},
        "heuristic": {
            "coverage_range": [min(coverage_pcts), max(coverage_pcts)],
            "correctness_f1_range": [min(f1s), max(f1s)],
        },
        "oracle": {"coverage": 1.0, "correctness": 1.0},
        "deranged": {
            "coverage": 1.0,
            "correctness_note": "7 of 2,609 edges technically identity-correct by chance "
                                 "(~0.27%), 84 more a text-coincidence with a repeated value "
                                 "(~3.2%); already disclosed in main.tex, not recomputed here",
        },
    }


def heuristic_f1_holdout_vs_complement():
    documents = load_documents(DOCUMENTS_PATH)
    holdout = select_holdout(documents)
    holdout_ids = {d.doc_id for d in holdout}
    complement = [d for d in documents if d.doc_id not in holdout_ids]

    def _validate(docs):
        total_gold = total_pred = total_tp = 0
        for doc in docs:
            gold = _gold_edges(doc)
            predicted = set(predict_links(doc))
            total_gold += len(gold)
            total_pred += len(predicted)
            total_tp += len(gold & predicted)
        precision = total_tp / total_pred if total_pred else 0.0
        recall = total_tp / total_gold if total_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"n_documents": len(docs), "n_gold_edges": total_gold, "n_predicted_edges": total_pred,
                "n_true_positive": total_tp, "precision": precision, "recall": recall, "f1": f1}

    return {"holdout_35": _validate(holdout), "complement_164": _validate(complement)}


def bucket_summary(bucket_ci_result):
    """Merge population share / absolute accuracy (from run3_bucket_diagnostic.json) with the new
    bootstrap CIs above into one table-ready structure -- everything Table/Figure 2 needs in one
    place, traceable to the pipeline, never hand-recomputed from a rounded table."""
    raw = json.load(open(BUCKET_PATH, encoding="utf-8"))
    out = {}
    for model in MODELS:
        recs = raw[model]
        n_total = len(recs)
        out[model] = {}
        for bucket in ["edge_correct", "edge_wrong", "no_edge"]:
            bucket_recs = [r for r in recs if r["bucket"] == bucket]
            n = len(bucket_recs)
            flat_acc = sum(r["flat_class"] == "correct" for r in bucket_recs) / n
            heur_init_acc = sum(r["heuristic_initial_class"] == "correct" for r in bucket_recs) / n
            heur_final_acc = sum(r["heuristic_final_class"] == "correct" for r in bucket_recs) / n
            counts = defaultdict(int)
            for r in bucket_recs:
                counts[r["heuristic_final_class"]] += 1
            ci = bucket_ci_result[model][bucket]
            out[model][bucket] = {
                "n": n, "pct_of_scorable": n / n_total,
                "flat_acc": flat_acc, "heuristic_initial_acc": heur_init_acc,
                "heuristic_final_acc": heur_final_acc,
                "error_composition_final": {k: v / n for k, v in counts.items()},
                "delta_point": ci["point_delta"], "delta_ci": [ci["ci_low"], ci["ci_high"]],
                "delta_ci_bonferroni": [ci["ci_low_bonferroni"], ci["ci_high_bonferroni"]],
            }
    return out


def main():
    bucket_ci_result = bucket_cis()
    bucket_summary_result = bucket_summary(bucket_ci_result)
    f1_result = heuristic_f1_holdout_vs_complement()
    out = {
        "rq1_raw_vs_flat_run2": rq1_raw_vs_flat(),
        "rq3b_oracle_vs_flat_run3": rq3b_oracle_vs_flat(),
        "bucket_cis_run3": bucket_ci_result,
        "bucket_summary_run3": bucket_summary_result,
        "wrong_edge_consistency_trajectory": heuristic_wrong_edge_trajectory(),
        "heuristic_f1_holdout_vs_complement": f1_result,
        "gpt5_interaction_contrasts": gpt5_interaction_contrasts(),
        "coverage_correctness_table": coverage_correctness_table(bucket_summary_result, f1_result),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
