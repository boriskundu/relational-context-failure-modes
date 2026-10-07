"""
Resumable extraction grid: cell = (document, condition, model). Adapts healthcare-eval's
runner.py resumable-checkpoint pattern — one partial JSON file per model, written by its own
worker thread, so a killed/interrupted run resumes by skipping already-completed cells and
retrying only failed ones.

``client_factory`` is injectable so tests can run the whole grid against a fake client with zero
network access.
"""
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agentic_docs.agent_graph import run_cell
from agentic_docs.config import CHECKPOINT_EVERY, HEARTBEAT_EVERY, RANDOM_SEED
from agentic_docs.funsd.parse import DocumentRecord
from agentic_docs.llm_clients import LLMClient, get_client
from agentic_docs.representations.flat import build_flat
from agentic_docs.representations.oracle_graph import build_oracle_graph
from agentic_docs.representations.predicted_graph import build_predicted_graph
from agentic_docs.representations.raw import build_raw
from agentic_docs.representations.shuffled_graph import build_shuffled_graph

CONDITIONS = ["raw", "flat", "predicted_graph", "oracle_graph", "shuffled_graph"]


def cell_key(document_id: str, condition: str, model: str) -> str:
    return f"{document_id}|{condition}|{model}"


def build_representation(condition: str, doc: DocumentRecord) -> tuple[object, set[int]]:
    """Returns (representation, excluded_question_ids) — the latter is only non-empty for
    shuffled_graph (Key decision #3's degree-preserving exclusion), empty for every other condition."""
    if condition == "raw":
        return build_raw(doc), set()
    if condition == "flat":
        return build_flat(doc), set()
    if condition == "predicted_graph":
        return build_predicted_graph(doc), set()
    if condition == "oracle_graph":
        return build_oracle_graph(doc), set()
    if condition == "shuffled_graph":
        return build_shuffled_graph(doc, seed=RANDOM_SEED)
    raise ValueError(f"Unknown condition: {condition}")


def _load_partial(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_partial(path: Path, rows: dict) -> None:
    """Write via a temp file + atomic rename -- a direct write can be killed mid-write (the exact
    scenario periodic checkpointing exists to protect against), leaving a truncated JSON file that
    `_load_partial` can't parse on the next run, losing not just this run but every previously
    checkpointed cell for that model.

    Retries ``os.replace`` on ``PermissionError``: observed live inside a OneDrive-synced working
    directory, where the sync client briefly opens a just-written file (indexing/upload), making
    the following rename fail with WinError 5 even though nothing is actually wrong with the data.
    Without this retry, that transient OS-level hiccup propagates out of `run_model` and gets the
    entire model excluded from `run_grid`'s merge (see its docstring) -- discarding a model's
    already-complete, already-durably-saved results over a lock that clears in well under a second.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (2 ** attempt))


def _needs_retry(row: dict) -> bool:
    # Retry on EITHER stage's failure (even a verify-only hiccup is worth a clean re-run), but
    # scoring (analyze.py) only skips the cell entirely on extract_parse_error -- see
    # agent_graph.py's module docstring for why the two are tracked separately. Module-level (not
    # nested in run_model) so status() can share the exact same predicate for its "done" count.
    return bool(row.get("extract_parse_error") or row.get("verify_parse_error"))


def run_model(
    model_name: str,
    documents: list[DocumentRecord],
    conditions: list[str],
    out_path: Path,
    client_factory: Callable[[str], LLMClient] = get_client,
) -> dict:
    """Run every (document, condition) cell for one model, checkpointing to its own partial file."""
    partial_path = out_path.with_name(f"{out_path.stem}.{model_name}{out_path.suffix}")
    rows = _load_partial(partial_path)
    client = client_factory(model_name)

    todo = [
        (doc, cond)
        for doc in documents
        for cond in conditions
        if cell_key(doc.doc_id, cond, model_name) not in rows
        or _needs_retry(rows[cell_key(doc.doc_id, cond, model_name)])
    ]
    total_cells = len(documents) * len(conditions)

    completed_since_checkpoint = 0
    start = time.monotonic()
    for i, (doc, cond) in enumerate(todo, start=1):
        cell_start = time.monotonic()
        representation, excluded_question_ids = build_representation(cond, doc)
        result = run_cell(client, cond, representation)
        key = cell_key(doc.doc_id, cond, model_name)
        rows[key] = {
            "cell": key,
            "document_id": doc.doc_id,
            "condition": cond,
            "model": model_name,
            "excluded_question_ids": sorted(excluded_question_ids),
            **result,
        }
        if i % HEARTBEAT_EVERY == 0:
            done = len(rows)
            elapsed = time.monotonic() - start
            print(
                f"  [{model_name}] {done}/{total_cells} done -- {key} "
                f"(cell took {time.monotonic() - cell_start:.0f}s, "
                f"parse_error={result['parse_error']}, elapsed {elapsed / 60:.1f}m)",
                flush=True,
            )
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= CHECKPOINT_EVERY:
            _save_partial(partial_path, rows)
            completed_since_checkpoint = 0

    _save_partial(partial_path, rows)
    return rows


def status(models: list[str], documents: list[DocumentRecord], conditions: list[str],
           out_path: Path) -> dict:
    """Report progress from existing partials only — no API calls.

    ``done`` means "a resume run will leave this cell alone" -- i.e. matches ``_needs_retry``'s own
    predicate exactly. A row with a verify-only failure has valid, scorable data (verify falls back
    to the extract answers), but ``_needs_retry`` still re-runs it (a clean verify attempt is worth
    it), so it must NOT be counted as done here either -- an earlier version counted it as done,
    letting --status report a cell as finished when the very next `docs run` would still re-execute
    it end to end.
    """
    total_per_model = len(documents) * len(conditions)
    report = {}
    for model_name in models:
        partial_path = out_path.with_name(f"{out_path.stem}.{model_name}{out_path.suffix}")
        rows = _load_partial(partial_path)
        done = sum(1 for r in rows.values() if not _needs_retry(r))
        failed = sum(1 for r in rows.values() if r.get("parse_error"))
        verify_failed = sum(1 for r in rows.values()
                             if not r.get("parse_error") and r.get("verify_parse_error"))
        report[model_name] = {
            "done": done, "failed": failed, "verify_failed": verify_failed, "total": total_per_model,
        }
    return report


def run_grid(
    models: list[str],
    documents: list[DocumentRecord],
    conditions: list[str],
    out_path: Path,
    client_factory: Callable[[str], LLMClient] = get_client,
) -> dict:
    """Run the full grid, one worker thread per model (parallel across models, sequential within a
    model, matching each provider's own rate limits).

    One model's exception (e.g. a missing API key surfacing as a KeyError only once its thread
    actually starts, bypassing get_available_models()'s upfront filter) must not discard every
    other model's completed work -- each model already checkpoints to its own partial file
    independently, so a per-model failure is isolated here too rather than left to propagate out
    of the whole ThreadPoolExecutor block before the merged file is ever written.
    """
    merged: dict = {}
    failed_models: dict[str, str] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(run_model, m, documents, conditions, out_path, client_factory): m
                for m in models
            }
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    merged.update(future.result())
                except Exception as exc:  # noqa: BLE001 -- isolate one model's failure from the rest
                    failed_models[model_name] = f"{type(exc).__name__}: {exc}"
                    print(f"[error] model '{model_name}' failed and is excluded from this merge: "
                          f"{failed_models[model_name]}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted -- partials are checkpointed, re-run to resume.")
        raise
    _save_partial(out_path, merged)
    if failed_models:
        print(f"[warn] {len(failed_models)}/{len(models)} model(s) missing from {out_path}: "
              f"{sorted(failed_models)} -- fix and re-run to resume just those models.", flush=True)
    return merged
