"""
Scoring taxonomy and document-level paired bootstrap (Key decision #9).

Core distinction this module exists to enforce: fuzzy/exact matching is the MEASUREMENT RULER,
applied identically no matter what's being measured. What varies is the INTERVENTION being tested
(which condition, which model, initial vs. final answer) — never the ruler itself.

Bootstrap resampling unit is the DOCUMENT, not the individual field — fields from the same document
are correlated (they share one source text, one model call context), so resampling ~600 fields
independently when the real independent unit is ~50 documents would produce artificially tight
confidence intervals. See Key decision #9.
"""

import random
from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_MATCH_THRESHOLD = 85.0  # rapidfuzz 0-100 scale; a field counts as a fuzzy match at/above this


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(frozen=True)
class FieldScore:
    doc_id: str
    condition: str
    model: str
    stage: str  # "initial" | "final"
    question_id: int
    predicted_answer: str
    gold_answer: str
    exact_match: bool
    fuzzy_score: float
    fuzzy_match: bool
    omission: bool
    hallucination: bool
    misassociation: bool
    relation_following_error: bool | None


def score_field(
    doc_id: str,
    condition: str,
    model: str,
    stage: str,
    question_id: int,
    predicted_answer: str,
    gold_answer: str,
    other_source_texts: list[str],
    shown_answer_text: str | None = None,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> FieldScore:
    """Score one predicted answer against gold for one question.

    ``other_source_texts`` is every OTHER entity's text in the same document (excluding this
    question's own gold answer) — used to distinguish misassociation (a real value, wrong field)
    from hallucination (unsupported anywhere in the source).

    ``shown_answer_text`` is what the graph EDGE shown to the model implies as the answer for this
    question (see representations._common.shown_answer_text_by_question) -- ``None`` when this
    condition carries no edges at all (Raw/Flat) or this question had none (e.g. a Predicted-graph
    heuristic miss). Used only for ``relation_following_error`` (Key decision #9): among questions
    whose shown edge was itself WRONG (Predicted misfire, or Shuffled derangement -- Oracle's is
    never wrong by construction), did the model blindly relay that wrong edge's answer rather than
    the correct one? ``None`` when not applicable (no edge, or the edge shown was already correct --
    nothing to blindly follow either way); never conflated with the ``False`` case (edge was wrong,
    but the model didn't fall for it).

    ``threshold`` is the fuzzy-match ruler applied everywhere in this function -- defaults to the
    module's ``FUZZY_MATCH_THRESHOLD`` (the paper's primary scoring rule); an explicit override
    exists only for the scorer-sensitivity sweep (scripts/scorer_sensitivity.py), never for the
    primary reported numbers.
    """
    pred_norm = normalize(predicted_answer)
    gold_norm = normalize(gold_answer)

    exact_match = bool(pred_norm) and pred_norm == gold_norm
    fuzzy_score = fuzz.ratio(pred_norm, gold_norm) if (pred_norm or gold_norm) else 100.0
    fuzzy_match = fuzzy_score >= threshold

    omission = bool(gold_norm) and not pred_norm

    hallucination = False
    misassociation = False
    if pred_norm and not fuzzy_match:
        matches_other_source = any(
            fuzz.ratio(pred_norm, normalize(other)) >= threshold
            for other in other_source_texts
            if normalize(other)
        )
        if matches_other_source:
            misassociation = True
        else:
            hallucination = True

    relation_following_error = None
    if shown_answer_text is not None:
        shown_norm = normalize(shown_answer_text)
        # Fuzzy, not exact, equality -- this module's own ruler (fuzzy_match, above) is what decides
        # "correct" everywhere else; judging the edge itself by exact string match would flag a
        # near-identical shown answer (e.g. a one-character OCR/formatting difference) as a wrong
        # edge, then double-penalize the model for "following" what fuzzy_match would call the
        # right answer anyway.
        shown_vs_gold = fuzz.ratio(shown_norm, gold_norm) if (shown_norm or gold_norm) else 100.0
        edge_is_wrong = shown_vs_gold < threshold
        if edge_is_wrong:
            relation_following_error = bool(pred_norm) and (
                fuzz.ratio(pred_norm, shown_norm) >= threshold
            )

    return FieldScore(
        doc_id=doc_id,
        condition=condition,
        model=model,
        stage=stage,
        question_id=question_id,
        predicted_answer=predicted_answer,
        gold_answer=gold_answer,
        exact_match=exact_match,
        fuzzy_score=fuzzy_score,
        fuzzy_match=fuzzy_match,
        omission=omission,
        hallucination=hallucination,
        misassociation=misassociation,
        relation_following_error=relation_following_error,
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def aggregate(scores: list[FieldScore]) -> dict:
    """Summary stats over a flat list of FieldScore — the caller filters to whatever slice
    (condition, model, stage) it wants before calling this."""
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "exact_match_rate": None,
            "hallucination_rate": None,
            "omission_rate": None,
            "misassociation_rate": None,
            "n_relation_following_applicable": 0,
            "relation_following_error_rate": None,
        }
    applicable = [
        s.relation_following_error for s in scores if s.relation_following_error is not None
    ]
    return {
        "n": n,
        "accuracy": sum(s.fuzzy_match for s in scores) / n,
        "exact_match_rate": sum(s.exact_match for s in scores) / n,
        "hallucination_rate": sum(s.hallucination for s in scores) / n,
        "omission_rate": sum(s.omission for s in scores) / n,
        "misassociation_rate": sum(s.misassociation for s in scores) / n,
        # Denominator is questions whose shown edge was itself WRONG, not all `n` -- see
        # score_field's docstring. Oracle's should always be n=0 (sanity check: its edges are never
        # wrong by construction).
        "n_relation_following_applicable": len(applicable),
        "relation_following_error_rate": (
            (sum(applicable) / len(applicable)) if applicable else None
        ),
    }


# ── RQ5: correction / regression / net correction ───────────────────────────


def correction_regression(initial_scores: list[FieldScore], final_scores: list[FieldScore]) -> dict:
    """Per Key decision (RQ5): NOT a naive initial-vs-final accuracy delta (a ceiling-effect
    problem — a condition starting near 100% has little room to show improvement regardless of how
    good its self-verification is). Measures, among initially-wrong fields, what fraction became
    correct (correction rate); among initially-correct fields, what fraction became wrong
    (regression rate); and the net of the two.
    """
    initial_by_key = {(s.doc_id, s.question_id): s for s in initial_scores}
    final_by_key = {(s.doc_id, s.question_id): s for s in final_scores}
    keys = set(initial_by_key) & set(final_by_key)

    initially_wrong = [k for k in keys if not initial_by_key[k].fuzzy_match]
    initially_correct = [k for k in keys if initial_by_key[k].fuzzy_match]

    corrected = sum(1 for k in initially_wrong if final_by_key[k].fuzzy_match)
    regressed = sum(1 for k in initially_correct if not final_by_key[k].fuzzy_match)

    correction_rate = corrected / len(initially_wrong) if initially_wrong else None
    regression_rate = regressed / len(initially_correct) if initially_correct else None
    net_correction = corrected - regressed

    return {
        "n_initially_wrong": len(initially_wrong),
        "n_initially_correct": len(initially_correct),
        "n_corrected": corrected,
        "n_regressed": regressed,
        "correction_rate": correction_rate,
        "regression_rate": regression_rate,
        "net_correction": net_correction,
    }


# ── Document-level paired bootstrap ──────────────────────────────────────────


def _percentile_ci(deltas: list[float], alpha: float) -> tuple[float | None, float | None]:
    """Two-sided percentile CI at the given alpha (e.g. 0.05 for 95%, 0.0125 for a
    four-comparison-Bonferroni-corrected 98.75%), from an already-sorted delta list."""
    if not deltas:
        return None, None
    lo = deltas[int((alpha / 2) * len(deltas))]
    hi = deltas[int((1 - alpha / 2) * len(deltas)) - 1]
    return lo, hi


def bootstrap_document_level(
    scores_a: list[FieldScore],
    scores_b: list[FieldScore],
    statistic_fn,
    n_boot: int = 20000,
    seed: int = 42,
    correction_n: int = 1,
) -> dict:
    """Bootstrap CI on ``statistic_fn(scores_a) - statistic_fn(scores_b)`` (e.g. Predicted-vs-Flat
    accuracy delta), resampling whole DOCUMENTS with replacement — never individual fields — and
    keeping the pairing intact (the same sampled documents are used for both A and B on each
    resample), since both conditions are scored on the same underlying document set.

    Restricted to the INTERSECTION of documents present in both ``scores_a`` and ``scores_b`` —
    a document can be scored on one side but not the other if its cell hit a parse error under
    only one of the two conditions being compared. Including a one-sided document would break the
    pairing (it contributes real fields to one side and none to the other on every resample,
    silently biasing the statistic), so it's dropped from both the point estimate and the bootstrap
    consistently, and the drop count is reported rather than silently absorbed.

    ``correction_n`` > 1 additionally reports a Bonferroni-corrected CI (alpha / correction_n) as
    ``ci_low_bonferroni`` / ``ci_high_bonferroni``, alongside the uncorrected 95% ``ci_low`` /
    ``ci_high`` — for when this same hypothesis is being tested once per model (e.g. 4 models).
    """
    doc_ids_a = {s.doc_id for s in scores_a}
    doc_ids_b = {s.doc_id for s in scores_b}
    doc_ids = sorted(doc_ids_a & doc_ids_b)
    n_dropped_one_sided = len(doc_ids_a ^ doc_ids_b)

    scores_a = [s for s in scores_a if s.doc_id in doc_ids_a & doc_ids_b]
    scores_b = [s for s in scores_b if s.doc_id in doc_ids_a & doc_ids_b]

    by_doc_a: dict[str, list[FieldScore]] = {d: [] for d in doc_ids}
    by_doc_b: dict[str, list[FieldScore]] = {d: [] for d in doc_ids}
    for s in scores_a:
        by_doc_a[s.doc_id].append(s)
    for s in scores_b:
        by_doc_b[s.doc_id].append(s)

    rng = random.Random(seed)
    point_a = statistic_fn(scores_a)
    point_b = statistic_fn(scores_b)
    point_delta = point_a - point_b

    deltas = []
    for _ in range(n_boot):
        sample_docs = [rng.choice(doc_ids) for _ in doc_ids] if doc_ids else []
        resampled_a = [s for d in sample_docs for s in by_doc_a[d]]
        resampled_b = [s for d in sample_docs for s in by_doc_b[d]]
        if not resampled_a or not resampled_b:
            continue
        deltas.append(statistic_fn(resampled_a) - statistic_fn(resampled_b))

    deltas.sort()
    lo, hi = _percentile_ci(deltas, 0.05)

    result = {
        "point_estimate_a": point_a,
        "point_estimate_b": point_b,
        "point_delta": point_delta,
        "ci_low": lo,
        "ci_high": hi,
        "n_documents": len(doc_ids),
        "n_dropped_one_sided": n_dropped_one_sided,
        "n_boot": len(deltas),
    }
    if correction_n > 1:
        lo_b, hi_b = _percentile_ci(deltas, 0.05 / correction_n)
        result["ci_low_bonferroni"] = lo_b
        result["ci_high_bonferroni"] = hi_b
        result["bonferroni_n"] = correction_n
    return result


def bootstrap_interaction_contrast(
    scores_a_treatment: list[FieldScore],
    scores_a_control: list[FieldScore],
    scores_b_treatment: list[FieldScore],
    scores_b_control: list[FieldScore],
    statistic_fn,
    n_boot: int = 20000,
    seed: int = 42,
    correction_n: int = 1,
) -> dict:
    """Bootstrap CI on a difference-in-differences: whether group A's own
    ``statistic_fn(treatment) - statistic_fn(control)`` differs from group B's same contrast (e.g.
    "is GPT-5's Heuristic-Flat delta different from Claude's Heuristic-Flat delta", not just whether
    each is individually different from zero). Generalizes ``bootstrap_document_level`` to four
    FieldScore lists resampled from ONE shared set of documents per draw (the intersection across
    all four), so the point estimate and every resample stay pooled/micro-averaged -- the same
    convention ``accuracy_statistic``/``aggregate`` use everywhere else in this project, not a
    mean-of-per-document-rates macro-average, which is a different (and here, wrong) estimand.
    """
    doc_id_sets = [
        {s.doc_id for s in scores_a_treatment},
        {s.doc_id for s in scores_a_control},
        {s.doc_id for s in scores_b_treatment},
        {s.doc_id for s in scores_b_control},
    ]
    doc_ids = sorted(set.intersection(*doc_id_sets))
    n_dropped_one_sided = len(set.union(*doc_id_sets)) - len(doc_ids)

    def _by_doc(scores):
        by_doc = {d: [] for d in doc_ids}
        for s in scores:
            if s.doc_id in by_doc:
                by_doc[s.doc_id].append(s)
        return by_doc

    by_doc_a_t, by_doc_a_c = _by_doc(scores_a_treatment), _by_doc(scores_a_control)
    by_doc_b_t, by_doc_b_c = _by_doc(scores_b_treatment), _by_doc(scores_b_control)

    def _contrast(sample_docs):
        a_t = [s for d in sample_docs for s in by_doc_a_t[d]]
        a_c = [s for d in sample_docs for s in by_doc_a_c[d]]
        b_t = [s for d in sample_docs for s in by_doc_b_t[d]]
        b_c = [s for d in sample_docs for s in by_doc_b_c[d]]
        return (statistic_fn(a_t) - statistic_fn(a_c)) - (statistic_fn(b_t) - statistic_fn(b_c))

    rng = random.Random(seed)
    point_delta = _contrast(doc_ids)

    deltas = []
    for _ in range(n_boot):
        sample_docs = [rng.choice(doc_ids) for _ in doc_ids] if doc_ids else []
        if not sample_docs:
            continue
        deltas.append(_contrast(sample_docs))

    deltas.sort()
    lo, hi = _percentile_ci(deltas, 0.05)

    result = {
        "point_delta": point_delta,
        "ci_low": lo,
        "ci_high": hi,
        "n_documents": len(doc_ids),
        "n_dropped_one_sided": n_dropped_one_sided,
        "n_boot": len(deltas),
    }
    if correction_n > 1:
        lo_b, hi_b = _percentile_ci(deltas, 0.05 / correction_n)
        result["ci_low_bonferroni"] = lo_b
        result["ci_high_bonferroni"] = hi_b
        result["bonferroni_n"] = correction_n
    return result


# ── Question-text alignment ──────────────────────────────────────────────────


def align_answers_to_pairs(
    question_pairs: list[tuple[int, str]],
    returned: dict[str, str],
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> dict[int, str]:
    """Map each (question_id, question_text) to the model's best-matching returned answer, by
    FUZZY question-text similarity rather than an exact dict-key lookup.

    Models frequently reformat a question's punctuation/whitespace when echoing it back — this is
    especially common under the Raw condition, where the question text isn't given as a discrete,
    copyable field but has to be picked out of running prose. An exact-match join would silently
    score a correctly-answered field as an omission whenever the model's echoed label differs by
    even one character from FUNSD's gold ``question_text`` string. Matching is one-to-one and
    greedy by score (highest-confidence pairs claimed first) so two similarly-worded questions in
    the same document can't both claim the same returned answer.
    """
    candidates = [
        (fuzz.ratio(normalize(qtext), normalize(rkey)), qid, rkey)
        for qid, qtext in question_pairs
        for rkey in returned
    ]
    candidates.sort(key=lambda c: -c[0])

    result: dict[int, str] = {qid: "" for qid, _ in question_pairs}
    matched_qids: set[int] = set()
    matched_rkeys: set[str] = set()
    for score, qid, rkey in candidates:
        if score < threshold:
            break  # sorted descending — no remaining candidate can clear the threshold either
        if qid in matched_qids or rkey in matched_rkeys:
            continue
        result[qid] = returned[rkey]
        matched_qids.add(qid)
        matched_rkeys.add(rkey)
    return result


def accuracy_statistic(scores: list[FieldScore]) -> float:
    if not scores:
        return 0.0
    return sum(s.fuzzy_match for s in scores) / len(scores)


def relation_following_error_rate_statistic(scores: list[FieldScore]) -> float:
    """Pooled wrong-edge-consistency rate over a list of FieldScore -- same denominator convention
    as ``aggregate``'s ``relation_following_error_rate`` (questions whose shown edge was itself
    wrong, not all scores), just as a standalone statistic_fn for ``bootstrap_document_level``. Used
    for the cross-round delta (scripts/cross_round_analysis.py); NOT the mean of each document's own
    rate (a macro-average), the same pooled/micro convention every other headline in this project
    uses.
    """
    applicable = [
        s.relation_following_error for s in scores if s.relation_following_error is not None
    ]
    if not applicable:
        return 0.0
    return sum(applicable) / len(applicable)
