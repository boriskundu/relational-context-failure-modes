import json

from agentic_docs.analyze import build_summary, score_run
from agentic_docs.funsd.parse import DocumentRecord, Entity, ScorablePair
from agentic_docs.runner import run_grid


def _tiny_doc(doc_id: str) -> DocumentRecord:
    entities = [
        Entity(id=0, text="Name:", label="question", box=(0, 0, 10, 10)),
        Entity(id=1, text="Bob", label="answer", box=(20, 0, 30, 10)),
        Entity(id=2, text="Date:", label="question", box=(0, 20, 10, 30)),
        Entity(id=3, text="1999", label="answer", box=(20, 20, 30, 30)),
    ]
    return DocumentRecord(
        doc_id=doc_id,
        entities=entities,
        links=[(0, 1), (2, 3)],
        scorable_pairs=[
            ScorablePair(
                question_id=0, question_text="Name:", answer_entity_ids=(1,), gold_answer_text="Bob"
            ),
            ScorablePair(
                question_id=2,
                question_text="Date:",
                answer_entity_ids=(3,),
                gold_answer_text="1999",
            ),
        ],
    )


class ScriptedClient:
    """Extract gets Name right and Date wrong; verify corrects Date."""

    def __init__(self):
        self.call_count = 0

    def generate_text(self, prompt: str, max_tokens=None) -> str:
        self.call_count += 1
        if self.call_count == 1:  # extract
            return json.dumps(
                [{"question": "Name:", "answer": "Bob"}, {"question": "Date:", "answer": "1990"}]
            )
        return json.dumps(
            [{"question": "Name:", "answer": "Bob"}, {"question": "Date:", "answer": "1999"}]
        )


class ReformattedKeyClient:
    """Both extract and verify echo the question text with slightly different punctuation than
    gold — regression test for the exact-string-match join bug found in the live sanity check."""

    def generate_text(self, prompt: str, max_tokens=None) -> str:
        return json.dumps(
            [{"question": "Name", "answer": "Bob"}, {"question": "Date", "answer": "1999"}]
        )


class VerifyFailsClient:
    """Extract succeeds; verify returns unparseable garbage every time."""

    def __init__(self):
        self.call_count = 0

    def generate_text(self, prompt: str, max_tokens=None) -> str:
        self.call_count += 1
        if self.call_count % 2 == 1:  # extract call
            return json.dumps(
                [{"question": "Name:", "answer": "Bob"}, {"question": "Date:", "answer": "1999"}]
            )
        return "not valid json"  # verify call


class BlindRelayClient:
    """Answers each question with whatever the OTHER question's gold value is -- simulates an
    agent naively relaying a swapped/wrong graph edge rather than reasoning about content."""

    def generate_text(self, prompt: str, max_tokens=None) -> str:
        return json.dumps(
            [{"question": "Name:", "answer": "1999"}, {"question": "Date:", "answer": "Bob"}]
        )


def test_relation_following_error_flagged_for_shuffled_graph_blind_relay(tmp_path):
    # _tiny_doc's two questions are both degree-1, so build_shuffled_graph's only possible
    # derangement for a 2-item bucket is the swap: Name's shown answer becomes "1999" (Date's
    # gold), Date's becomes "Bob" (Name's gold) -- deterministic regardless of seed.
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(
        ["m1"], docs, ["shuffled_graph"], out_path, client_factory=lambda name: BlindRelayClient()
    )

    _, final_scores = score_run(merged, docs)
    assert len(final_scores) == 2
    for s in final_scores:
        assert s.fuzzy_match is False  # answered the swapped-wrong value, not gold
        assert s.relation_following_error is True  # ...because it blindly relayed the deranged edge


def test_relation_following_error_always_none_for_oracle_graph(tmp_path):
    # Oracle's edges are never wrong by construction -- nothing to blindly follow either way.
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(
        ["m1"], docs, ["oracle_graph"], out_path, client_factory=lambda name: ScriptedClient()
    )

    _, final_scores = score_run(merged, docs)
    assert len(final_scores) == 2
    assert all(s.relation_following_error is None for s in final_scores)


def test_verify_stage_failure_does_not_discard_extract_stage_data(tmp_path):
    # Regression test: score_run previously dropped the WHOLE cell (both initial and final scores)
    # whenever verify_parse_error was set, even though extract succeeded and final_answers correctly
    # falls back to the (valid) initial_answers. That silently removed good data from every metric,
    # including the RQ2 headline bootstrap and RQ5 correction/regression, on any verify-stage hiccup.
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(
        ["m1"], docs, ["flat"], out_path, client_factory=lambda name: VerifyFailsClient()
    )

    row = next(iter(merged.values()))
    assert row["extract_parse_error"] is False
    assert row["verify_parse_error"] is True

    initial_scores, final_scores = score_run(merged, docs)
    assert len(initial_scores) == 2  # NOT dropped
    assert len(final_scores) == 2
    assert all(s.fuzzy_match for s in initial_scores)
    assert all(
        s.fuzzy_match for s in final_scores
    )  # final falls back to the correct initial answers


def test_score_run_matches_reformatted_question_text(tmp_path):
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(
        ["m1"], docs, ["flat"], out_path, client_factory=lambda name: ReformattedKeyClient()
    )

    initial_scores, _ = score_run(merged, docs)
    # Gold question_text is "Name:"/"Date:"; the model dropped the trailing colon on both. A naive
    # exact-string join would score both as omissions despite correct answers.
    assert len(initial_scores) == 2
    assert all(s.fuzzy_match for s in initial_scores)


def test_end_to_end_runner_to_analyze_join(tmp_path):
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(
        ["m1"], docs, ["flat"], out_path, client_factory=lambda name: ScriptedClient()
    )

    initial_scores, final_scores = score_run(merged, docs)
    # 1 doc x 2 scorable questions = 2 scores per stage
    assert len(initial_scores) == 2
    assert len(final_scores) == 2

    date_initial = next(s for s in initial_scores if s.question_id == 2)
    date_final = next(s for s in final_scores if s.question_id == 2)
    assert date_initial.fuzzy_match is False  # "1990" vs gold "1999"
    assert date_final.fuzzy_match is True  # verify corrected it to "1999"

    summary = build_summary(initial_scores, final_scores)
    assert summary["by_condition_model"]["flat|m1"]["n"] == 2
    assert summary["by_condition_model"]["flat|m1"]["accuracy"] == 1.0  # both correct after verify
    correction = summary["rq5_correction"]["flat|m1"]
    assert correction["n_corrected"] == 1
    assert correction["n_regressed"] == 0
