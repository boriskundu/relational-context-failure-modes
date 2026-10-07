import json
import os

from agentic_docs.funsd.parse import DocumentRecord, Entity, ScorablePair
from agentic_docs.runner import _save_partial, cell_key, run_grid, run_model, status


def _tiny_doc(doc_id: str) -> DocumentRecord:
    entities = [
        Entity(id=0, text="Name:", label="question", box=(0, 0, 10, 10)),
        Entity(id=1, text="Bob", label="answer", box=(20, 0, 30, 10)),
    ]
    return DocumentRecord(
        doc_id=doc_id,
        entities=entities,
        links=[(0, 1)],
        scorable_pairs=[ScorablePair(question_id=0, question_text="Name:",
                                      answer_entity_ids=(1,), gold_answer_text="Bob")],
    )


class AlwaysSucceedsClient:
    def generate_text(self, prompt: str, max_tokens=None) -> str:
        return json.dumps([{"question": "Name:", "answer": "Bob"}])


class AlwaysFailsClient:
    def generate_text(self, prompt: str, max_tokens=None) -> str:
        return "not valid json"


class CountingClient:
    """Tracks how many times generate_text was called, to prove resumed cells are skipped."""
    calls = 0

    def generate_text(self, prompt: str, max_tokens=None) -> str:
        CountingClient.calls += 1
        return json.dumps([{"question": "Name:", "answer": "Bob"}])


def test_cell_key_format():
    assert cell_key("doc1", "flat", "gpt-5") == "doc1|flat|gpt-5"


def test_run_model_writes_partial_and_completes_all_cells(tmp_path):
    docs = [_tiny_doc("d1"), _tiny_doc("d2")]
    out_path = tmp_path / "run.json"
    rows = run_model("fake-model", docs, ["raw", "flat"], out_path,
                      client_factory=lambda name: AlwaysSucceedsClient())

    assert len(rows) == 4  # 2 docs x 2 conditions
    partial_path = tmp_path / "run.fake-model.json"
    assert partial_path.exists()
    saved = json.loads(partial_path.read_text())
    assert len(saved) == 4
    for row in rows.values():
        assert row["parse_error"] is False
        assert row["final_answers"] == {"Name:": "Bob"}


def test_run_model_resumes_and_skips_already_completed_cells(tmp_path):
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    CountingClient.calls = 0
    run_model("fake-model", docs, ["raw", "flat"], out_path, client_factory=lambda name: CountingClient())
    assert CountingClient.calls == 2 * 2  # 1 doc x 2 conditions x 2 calls per cell (extract+verify)

    # Re-run against the same out_path — every cell already succeeded, so no new calls should happen.
    run_model("fake-model", docs, ["raw", "flat"], out_path, client_factory=lambda name: CountingClient())
    assert CountingClient.calls == 4  # unchanged


def test_run_model_retries_failed_cells_on_resume(tmp_path):
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    run_model("fake-model", docs, ["raw"], out_path, client_factory=lambda name: AlwaysFailsClient())
    partial_path = tmp_path / "run.fake-model.json"
    saved = json.loads(partial_path.read_text())
    assert all(r["parse_error"] for r in saved.values())

    # Resume with a client that now succeeds — the previously-failed cell must be retried, not skipped.
    rows = run_model("fake-model", docs, ["raw"], out_path, client_factory=lambda name: AlwaysSucceedsClient())
    assert all(not r["parse_error"] for r in rows.values())


def test_status_reports_progress_without_api_calls(tmp_path):
    docs = [_tiny_doc("d1"), _tiny_doc("d2")]
    out_path = tmp_path / "run.json"
    run_model("m1", docs, ["raw", "flat"], out_path, client_factory=lambda name: AlwaysSucceedsClient())

    report = status(["m1"], docs, ["raw", "flat"], out_path)
    assert report["m1"]["done"] == 4
    assert report["m1"]["failed"] == 0
    assert report["m1"]["total"] == 4


def test_run_grid_merges_all_models_into_one_output_file(tmp_path):
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"
    merged = run_grid(["m1", "m2"], docs, ["raw"], out_path,
                       client_factory=lambda name: AlwaysSucceedsClient())
    assert len(merged) == 2  # 1 doc x 1 condition x 2 models
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert len(saved) == 2


def test_save_partial_retries_transient_permission_error(tmp_path, monkeypatch):
    # Regression test: observed live inside a OneDrive-synced working directory, where the sync
    # client briefly opens a just-written file, making os.replace fail with a transient
    # PermissionError even though the write itself succeeded. Without a retry, this propagates
    # out of run_model and gets an entire model's already-complete results excluded from
    # run_grid's merge (test_run_grid_isolates_one_models_crash_from_the_rest's failure mode) --
    # over a lock that clears almost immediately.
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    path = tmp_path / "run.json"
    _save_partial(path, {"a": 1})
    assert calls["n"] == 3
    assert json.loads(path.read_text()) == {"a": 1}


def test_save_partial_raises_after_exhausting_retries(tmp_path, monkeypatch):
    import agentic_docs.runner as runner_module

    def always_fails(src, dst):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(os, "replace", always_fails)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)  # keep the test fast
    path = tmp_path / "run.json"
    try:
        _save_partial(path, {"a": 1})
        assert False, "expected PermissionError to propagate after exhausting retries"
    except PermissionError:
        pass


def test_run_grid_isolates_one_models_crash_from_the_rest(tmp_path):
    # Regression test: one model's factory/thread raising (e.g. a missing API key surfacing only
    # once the thread starts) previously propagated out of run_grid entirely, so the merged output
    # file was never written -- discarding a fully-completed OTHER model's results along with it.
    docs = [_tiny_doc("d1")]
    out_path = tmp_path / "run.json"

    def factory(name):
        if name == "broken-model":
            raise KeyError("MISSING_API_KEY")
        return AlwaysSucceedsClient()

    merged = run_grid(["good-model", "broken-model"], docs, ["raw"], out_path, client_factory=factory)
    assert len(merged) == 1  # only good-model's cell
    assert all(row["model"] == "good-model" for row in merged.values())
    assert out_path.exists()  # the merged file must still be written for the model that succeeded
    saved = json.loads(out_path.read_text())
    assert len(saved) == 1
