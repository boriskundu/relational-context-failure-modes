from agentic_docs.metrics import (
    aggregate,
    align_answers_to_pairs,
    bootstrap_document_level,
    bootstrap_interaction_contrast,
    correction_regression,
    relation_following_error_rate_statistic,
    score_field,
)


def _score(
    doc_id="d1",
    condition="flat",
    model="m",
    stage="final",
    qid=1,
    predicted="",
    gold="",
    others=None,
    shown=None,
):
    return score_field(
        doc_id, condition, model, stage, qid, predicted, gold, others or [], shown_answer_text=shown
    )


def test_exact_match_when_predicted_equals_gold():
    s = _score(predicted="John Smith", gold="John Smith")
    assert s.exact_match is True
    assert s.fuzzy_match is True
    assert s.omission is False
    assert s.hallucination is False
    assert s.misassociation is False


def test_case_and_whitespace_insensitive_exact_match():
    s = _score(predicted="  JOHN   smith ", gold="John Smith")
    assert s.exact_match is True


def test_blank_gold_and_blank_prediction_is_neither_omission_nor_hallucination():
    s = _score(predicted="", gold="")
    assert s.omission is False
    assert s.hallucination is False
    assert s.misassociation is False


def test_nonblank_gold_blank_prediction_is_omission():
    s = _score(predicted="", gold="John Smith")
    assert s.omission is True
    assert s.hallucination is False
    assert s.exact_match is False


def test_blank_gold_nonblank_prediction_unsupported_is_hallucination():
    s = _score(predicted="December 14, 2007", gold="", others=["Finance", "555-1234"])
    assert s.omission is False
    assert s.hallucination is True
    assert s.misassociation is False


def test_wrong_but_real_value_from_elsewhere_is_misassociation_not_hallucination():
    s = _score(predicted="Finance", gold="John Smith", others=["Finance", "555-1234"])
    assert s.hallucination is False
    assert s.misassociation is True
    assert s.exact_match is False


def test_fuzzy_match_tolerates_minor_formatting_difference():
    s = _score(predicted="J. Smith", gold="J Smith")
    assert s.fuzzy_score >= 85.0
    assert s.fuzzy_match is True


def test_score_field_threshold_override_changes_match_outcome():
    # "Jon Smith" vs "John Smith" scores in the high-80s/low-90s on rapidfuzz's 0-100 scale --
    # below a 95 threshold it's a mismatch, at the default 85 threshold it matches. Confirms the new
    # `threshold` override on score_field actually takes effect (used only by the scorer-sensitivity
    # sweep), rather than silently falling back to the module constant.
    default = score_field("d1", "flat", "m", "final", 1, "Jon Smith", "John Smith", [])
    assert default.fuzzy_match is True

    strict = score_field(
        "d1", "flat", "m", "final", 1, "Jon Smith", "John Smith", [], threshold=95.0
    )
    assert strict.fuzzy_match is False


def test_relation_following_error_is_none_when_no_edge_shown():
    # Raw/Flat carry no edges at all -- shown_answer_text is None -- so there's nothing to
    # blindly follow, not even a "correct edge" case.
    s = _score(predicted="John Smith", gold="John Smith", shown=None)
    assert s.relation_following_error is None


def test_relation_following_error_is_none_when_shown_edge_is_correct():
    # Oracle's edges are never wrong by construction -- shown == gold means nothing to blindly
    # follow either way, regardless of what the model actually answered.
    s = _score(predicted="wrong guess", gold="John Smith", shown="John Smith")
    assert s.relation_following_error is None


def test_relation_following_error_true_when_model_follows_wrong_edge():
    # Shuffled derangement points this question at "Finance" instead of the correct "John Smith" --
    # the model faithfully relayed the WRONG edge it was shown.
    s = _score(predicted="Finance", gold="John Smith", shown="Finance")
    assert s.relation_following_error is True


def test_relation_following_error_false_when_model_ignores_wrong_edge():
    # Edge shown was wrong ("Finance"), but the model answered something else entirely --
    # it did NOT blindly follow the bad edge (a different failure mode, already captured by
    # hallucination/misassociation/omission).
    s = _score(predicted="", gold="John Smith", shown="Finance")
    assert s.relation_following_error is False


def test_relation_following_error_false_when_model_gets_it_right_despite_wrong_edge():
    s = _score(predicted="John Smith", gold="John Smith", shown="Finance")
    assert s.relation_following_error is False


def test_aggregate_relation_following_error_rate_denominator_excludes_not_applicable():
    scores = [
        _score(qid=1, predicted="A", gold="A", shown=None),  # N/A: no edge
        _score(qid=2, predicted="A", gold="A", shown="A"),  # N/A: edge was correct
        _score(qid=3, predicted="Finance", gold="A", shown="Finance"),  # followed wrong edge
        _score(qid=4, predicted="", gold="A", shown="Finance"),  # wrong edge, but not followed
    ]
    result = aggregate(scores)
    assert result["n"] == 4
    assert result["n_relation_following_applicable"] == 2
    assert result["relation_following_error_rate"] == 0.5


def test_aggregate_empty_list_has_none_rates():
    result = aggregate([])
    assert result["n"] == 0
    assert result["accuracy"] is None


def test_aggregate_computes_rates_correctly():
    scores = [
        _score(qid=1, predicted="A", gold="A"),  # correct
        _score(qid=2, predicted="", gold="B"),  # omission
        _score(qid=3, predicted="X", gold="", others=["Y"]),  # hallucination
        _score(qid=4, predicted="Y", gold="C", others=["Y"]),  # misassociation
    ]
    result = aggregate(scores)
    assert result["n"] == 4
    assert result["accuracy"] == 0.25
    assert result["omission_rate"] == 0.25
    assert result["hallucination_rate"] == 0.25
    assert result["misassociation_rate"] == 0.25


def test_correction_regression_math():
    initial = [
        _score(doc_id="d1", qid=1, stage="initial", predicted="", gold="A"),  # wrong (omission)
        _score(doc_id="d1", qid=2, stage="initial", predicted="B", gold="B"),  # correct
        _score(doc_id="d1", qid=3, stage="initial", predicted="C", gold="C"),  # correct
    ]
    final = [
        _score(doc_id="d1", qid=1, stage="final", predicted="A", gold="A"),  # corrected
        _score(doc_id="d1", qid=2, stage="final", predicted="B", gold="B"),  # stayed correct
        _score(doc_id="d1", qid=3, stage="final", predicted="", gold="C"),  # regressed
    ]
    result = correction_regression(initial, final)
    assert result["n_initially_wrong"] == 1
    assert result["n_initially_correct"] == 2
    assert result["n_corrected"] == 1
    assert result["n_regressed"] == 1
    assert result["correction_rate"] == 1.0
    assert result["regression_rate"] == 0.5
    assert result["net_correction"] == 0


def test_correction_regression_no_initially_wrong_fields_gives_none_rate():
    initial = [_score(doc_id="d1", qid=1, stage="initial", predicted="A", gold="A")]
    final = [_score(doc_id="d1", qid=1, stage="final", predicted="A", gold="A")]
    result = correction_regression(initial, final)
    assert result["correction_rate"] is None
    assert result["n_initially_wrong"] == 0


def test_bootstrap_resamples_documents_not_individual_fields():
    # Two documents; scores_a has doc2 perfect, doc1 all wrong. A bootstrap sample that never
    # draws doc2 must show 0% accuracy, which is only possible if it resamples whole documents.
    scores_a = [_score(doc_id="d1", qid=i, predicted="", gold="X") for i in range(5)] + [
        _score(doc_id="d2", qid=i, predicted="X", gold="X") for i in range(5)
    ]
    scores_b = [_score(doc_id="d1", qid=i, predicted="", gold="X") for i in range(5)] + [
        _score(doc_id="d2", qid=i, predicted="", gold="X") for i in range(5)
    ]

    from agentic_docs.metrics import accuracy_statistic

    result = bootstrap_document_level(scores_a, scores_b, accuracy_statistic, n_boot=200, seed=1)
    assert result["n_documents"] == 2
    assert result["point_estimate_a"] == 0.5  # doc1 all wrong, doc2 all right -> 50%
    assert result["point_estimate_b"] == 0.0  # both docs all wrong
    assert result["point_delta"] == 0.5


def test_align_answers_matches_reformatted_question_text_not_exact_string():
    # Model echoed "Name / Phone Ext.:" (no space before colon); gold is "Name / Phone Ext. :" —
    # this is the real mismatch found during the live single-cell sanity check under Raw, where
    # the question text isn't given as a copyable field and the model retypes it from prose.
    pairs = [(1, "Name / Phone Ext. :"), (2, "Date:")]
    returned = {"Name / Phone Ext.:": "M. Hamann", "Date:": "9/3/92"}
    aligned = align_answers_to_pairs(pairs, returned)
    assert aligned == {1: "M. Hamann", 2: "9/3/92"}


def test_align_answers_is_one_to_one_not_double_claimed():
    # Two similarly-worded gold questions must not both grab the same returned answer.
    pairs = [(1, "Manager:"), (2, "Manager Comments:")]
    returned = {"Manager Comments:": "Approved"}
    aligned = align_answers_to_pairs(pairs, returned)
    # "Manager Comments:" is a much closer match to gold question 2 than gold question 1, and can
    # only be claimed once.
    assert aligned[2] == "Approved"
    assert aligned[1] == ""


def test_align_answers_below_threshold_is_unmatched():
    pairs = [(1, "Signature:")]
    returned = {"Completely unrelated field": "xyz"}
    aligned = align_answers_to_pairs(pairs, returned)
    assert aligned[1] == ""


def test_bootstrap_drops_documents_missing_from_either_side():
    # d1 is scored on both sides; d2 only exists in scores_a (e.g. its cell parse-errored under
    # condition B and was excluded from scoring entirely). Including d2 would break the pairing --
    # it contributes real fields to A and none to B on every resample.
    scores_a = [
        _score(doc_id="d1", qid=i, predicted="X", gold="X") for i in range(4)
    ] + [  # d1: 100% correct
        _score(doc_id="d2", qid=i, predicted="", gold="X") for i in range(4)
    ]  # d2: 0% correct
    scores_b = [
        _score(doc_id="d1", qid=i, predicted="", gold="X") for i in range(4)
    ]  # d1: 0% correct

    from agentic_docs.metrics import accuracy_statistic

    result = bootstrap_document_level(scores_a, scores_b, accuracy_statistic, n_boot=200, seed=1)
    assert result["n_documents"] == 1  # only d1 is common to both sides
    assert result["n_dropped_one_sided"] == 1  # d2
    # Point estimates must reflect ONLY the intersected document (d1), not d2's fields leaking in.
    assert result["point_estimate_a"] == 1.0
    assert result["point_estimate_b"] == 0.0


def test_bootstrap_point_delta_is_deterministic_given_seed():
    scores_a = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    scores_b = [_score(doc_id="d1", qid=1, predicted="", gold="A")]
    from agentic_docs.metrics import accuracy_statistic

    r1 = bootstrap_document_level(scores_a, scores_b, accuracy_statistic, n_boot=50, seed=7)
    r2 = bootstrap_document_level(scores_a, scores_b, accuracy_statistic, n_boot=50, seed=7)
    assert r1 == r2


def test_relation_following_error_rate_statistic_is_pooled_not_macro_averaged():
    # Doc d1 has 1 applicable question (error), doc d2 has 3 (all no error). Pooled = 1/4 = 25%,
    # NOT the mean of each document's own rate (100% and 0%, averaging to 50%) -- the distinction
    # this statistic exists to enforce, matching accuracy_statistic's own pooled convention.
    scores = [_score(doc_id="d1", qid=1, predicted="Finance", gold="A", shown="Finance")] + [
        _score(doc_id="d2", qid=i, predicted="A", gold="A", shown="WRONG") for i in range(3)
    ]
    assert relation_following_error_rate_statistic(scores) == 0.25


def test_relation_following_error_rate_statistic_zero_when_none_applicable():
    scores = [_score(doc_id="d1", qid=1, predicted="A", gold="A", shown=None)]
    assert relation_following_error_rate_statistic(scores) == 0.0


def test_bootstrap_interaction_contrast_zero_when_both_groups_have_equal_effect():
    from agentic_docs.metrics import accuracy_statistic

    # Both groups A and B show the identical +100pp (treatment) vs 0pp (control) effect -- the
    # interaction (difference of the two differences) must be exactly zero.
    treatment = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    control = [_score(doc_id="d1", qid=1, predicted="", gold="A")]
    result = bootstrap_interaction_contrast(
        treatment, control, treatment, control, accuracy_statistic, n_boot=50, seed=1
    )
    assert result["point_delta"] == 0.0


def test_bootstrap_interaction_contrast_detects_a_real_difference():
    from agentic_docs.metrics import accuracy_statistic

    # Group A: treatment perfect, control all wrong (a +100pp effect). Group B: no effect at all
    # (both perfect). The interaction must be +100pp, and its CI must exclude zero.
    a_treatment = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    a_control = [_score(doc_id="d1", qid=1, predicted="", gold="A")]
    b_treatment = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    b_control = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    result = bootstrap_interaction_contrast(
        a_treatment, a_control, b_treatment, b_control, accuracy_statistic, n_boot=50, seed=1
    )
    assert result["point_delta"] == 1.0


def test_bootstrap_interaction_contrast_drops_documents_missing_from_any_side():
    from agentic_docs.metrics import accuracy_statistic

    a_treatment = [
        _score(doc_id="d1", qid=1, predicted="A", gold="A"),
        _score(doc_id="d2", qid=1, predicted="A", gold="A"),
    ]  # d2 only here
    a_control = [_score(doc_id="d1", qid=1, predicted="", gold="A")]
    b_treatment = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    b_control = [_score(doc_id="d1", qid=1, predicted="A", gold="A")]
    result = bootstrap_interaction_contrast(
        a_treatment, a_control, b_treatment, b_control, accuracy_statistic, n_boot=50, seed=1
    )
    assert result["n_documents"] == 1
    assert result["n_dropped_one_sided"] == 1
