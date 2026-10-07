# relational-context-failure-modes

Code, prompts, and aggregate analysis files for **"Failure Modes of Imperfect Relational Context in
LLM Document Extraction"** (Boris Kundu and Pooja Mehta, AACL-IJCNLP 2026 Workshop NORA).

The study asks how imperfect relations between document entities, supplied as input context,
change what an LLM extracts. On FUNSD (scanned forms) it holds entity text, labels, and layout
fixed and varies only the supplied relations, across four LLMs and a fixed extract-then-verify
pipeline.

## Conditions and names

The code uses earlier internal names for two of the conditions.

| Paper | Code (`--conditions`, result keys) | Relations shown to the model |
|---|---|---|
| Flat | `flat` | none |
| Heuristic | `predicted_graph` | layout-proximity heuristic |
| Oracle | `oracle_graph` | FUNSD gold links |
| Deranged | `shuffled_graph` | degree-preserving derangement, nearly all wrong |
| Raw (background only) | `raw` | text only, no labels or layout |

Code and result keys also use internal labels RQ1 to RQ5. RQ1 is Raw vs Flat, RQ2 is Heuristic vs
Flat, RQ3 is Oracle vs Heuristic, RQ4 is Deranged vs Flat, and RQ5 is the verify step's
correction and regression rates. The paper does not use these labels.

Prompt rounds in the paper map to files as follows. Round 1 is `docs/prompt_versions/run1_*` and
`results/analysis/summary.json`. Round 2 is `docs/prompt_versions/run2_*` and
`results/analysis/summary_condition_blind.json`. Round 3, the primary evaluation, is
`docs/prompt_versions/run3_final_SKILL.md` and `results/analysis/summary_run3.json`.

## Setup

Requires Python 3.10 or newer.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env   # then fill in the API keys you need
```

**Linux / macOS**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env   # then fill in the API keys you need
```

## Data

FUNSD is not redistributed here. It is licensed for non-commercial research and educational use
(EPFL-LTS5). Download it from its official source and build the combined document file:

```
python -m agentic_docs.funsd.download
```

This writes `data/documents_all.json` (199 documents). The code never sends document images to any
model; every representation is built from FUNSD's annotation text, boxes, labels, and links.

## Tests

The offline suite needs no API keys, network access, or credits.

```
python -m pytest tests/
```

or `make test` where `make` is available (set `PYTHON=.venv/Scripts/python` on Windows, or
`PYTHON=.venv/bin/python` on Linux and macOS, if the virtual environment is not activated). Other
targets: `make lint`, `make format`, `make type-check`, `make docker`. See [TESTING.md](TESTING.md).

## Running the experiment

Running calls paid APIs for Anthropic, OpenAI, Groq, and Google. Start with a small pilot.

```
docs run --status            # progress report, no API calls
docs run --limit 2           # tiny sanity run
docs run                     # full grid, resumable
docs analyze                 # score runs against gold
docs visualize               # figures and summary table
```

Use `docs run --help` for model, condition, and output-path options. Model identifiers and
reasoning settings are in `src/agentic_docs/config.py`; the paper's appendix lists the values used.
Hosted models change over time, so a rerun will not reproduce the paper's numbers exactly.

## What is included

- `src/agentic_docs/`: representation builders, heuristic linker, extract-then-verify agent,
  scoring, and analysis.
- `src/agentic_docs/skills/` and `docs/prompt_versions/`: the prompt templates for every round.
- `tests/`: offline test suite.
- `scripts/`: post-hoc analyses and figure generation used for the paper. Only
  `scripts/camera_ready_figures.py` runs from the released files (`python
  scripts/camera_ready_figures.py`). The others read per-question outputs under
  `results/extractions/`, which are not released, so they need a rerun of `docs run` first; their
  results are the JSON files in `results/analysis/`.
- `results/analysis/`: aggregate analysis files behind the paper's tables.

Not included: FUNSD itself, and per-question model outputs (they contain FUNSD text). The
per-item manual-validation samples described in `results/analysis/MANUAL_VALIDATION.md` are
likewise not released.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## License

The code and prompts are released under the [MIT License](LICENSE). FUNSD is covered by its own
license and is not part of this repository. `tests/fixtures/sample_funsd_annotation.json` is a
single FUNSD annotation file used as a parser test fixture.

## Citation

See [CITATION.cff](CITATION.cff), or cite:

```bibtex
@inproceedings{kundu2026failure,
  title     = {Failure Modes of Imperfect Relational Context in {LLM} Document Extraction},
  author    = {Kundu, Boris and Mehta, Pooja},
  booktitle = {Proceedings of the AACL-IJCNLP 2026 Workshop on Knowledge Graphs and Agentic Systems Interplay (NORA)},
  year      = {2026}
}
```
