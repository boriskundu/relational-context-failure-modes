# Testing

`make test` runs the full offline suite — no API keys, no network access, no cost. This is the
default and what CI should run.

| Module | What's tested |
|---|---|
| `funsd/parse.py` | Real annotation fixture (a genuine FUNSD file, not synthetic) covering unlinked questions, header-to-multiple-questions chains, single-answer and multi-answer question→answer resolution |
| `representations/*.py` | Each of the five builders against the fixture, plus the content-set-equality guard (all five conditions must carry identical underlying text — only structure exposure may differ) and the box-parity guard (Flat and every graph condition carry identical bounding boxes) |
| `heuristic/proximity.py` | Deduplication, and a leakage-guard test asserting the heuristic's output is unchanged even if the document's gold links/scorable pairs are wiped out — it never reads them |
| `prompt_builder.py` | The three conditions' skill templates are byte-identical outside the `<!-- INPUT-SPECIFIC -->` block; the three graph conditions share the exact same template |
| `agent_graph.py` | Extract→verify wiring against a fake `LLMClient`, including parse-error handling at both stages and verify-node corrections overriding extract-node output |
| `metrics.py` | Every cell of the exact-match / fuzzy-match / omission / hallucination / misassociation taxonomy, correction/regression math, the question-text fuzzy-alignment join (`align_answers_to_pairs` — a model reformatting a question's punctuation must still match its gold pair, and matching is one-to-one), and the document-level bootstrap's resampling unit and cross-condition pairing (a document scored on only one side of a comparison must be dropped from both, not silently bias the statistic) |
| `runner.py` | Resume/retry against a fake client — a completed cell is skipped, a failed cell is retried, `--status` reports progress with zero API calls |
| `analyze.py` (`test_analyze.py`) | Full runner→analyze join, end to end, against a fake client — validates that the metrics pipeline correctly attaches gold answers to runner output rows, including when the model reformats a question's punctuation |
| `visualize.py` | Each figure/CSV export against a hand-built summary dict — writes a non-empty file without crashing |

## Requires real network/API access (not run by `make test`)

- `funsd/download.py` — downloads the actual FUNSD dataset. Run once via `make funsd-download`.
- Live LLM calls through the real `extract_graph` — needs API keys in `.env`. `make pilot` runs a
  small (20-document) end-to-end pilot.

## Leakage-prevention checklist (code review, not automated)

Before any commit touching the extraction path, grep-confirm:
- Nothing outside `heuristic/validate.py`, `representations/oracle_graph.py`, and tests reads a
  document's gold `linking` field — this is what keeps `predicted_graph.py`'s heuristic honest.

## Fuzzy-match scorer validation (do this once real pilot data exists)

`metrics.py`'s unit tests check the scoring function's boundary logic, not whether its fuzzy-match
threshold actually agrees with human judgment on real model outputs. After the first real pilot run
(`make pilot`), hand-check ~50-100 scored items against your own judgment of whether the match was
actually correct — this is the same discipline as any human-validated automated scorer, and it's
cheap to do once, early, before scaling to the full grid.
