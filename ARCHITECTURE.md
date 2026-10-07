# Architecture

## Research question

For a self-verifying LLM document-processing system, does explicit relational structure improve
field-completion accuracy beyond flat structured extraction — and does it change the system's
ability to catch and correct its own mistakes? Evaluated on FUNSD (Form Understanding in Noisy
Scanned Documents), across four models.

## Data

FUNSD: 199 real noisy scanned forms, 9,707 entities, 5,304 relations. FUNSD's own official release
splits these into 149 train / 50 test documents; this project merges both into one combined
199-document dataset (`data/documents_all.json`) rather than holding out a separate frozen test
split. License: non-commercial research and educational use only (EPFL-LTS5). Source images derive
from RVL-CDIP; this project never processes document images — every representation is built from
FUNSD's own annotation JSON (human-transcribed text, bounding boxes, type labels, and linking
edges), so the image-specific license clause doesn't apply.

**Scorable task unit**: a `question`-labeled entity with at least one linked `answer`-labeled
entity. Where a question links to multiple answer entities (a real, common pattern — e.g. one
answer split across line-wrapped boxes, or a genuine multi-option field), those are concatenated in
reading order into one scored answer. Header-to-question chains and unlinked questions are
structural context, never scored targets.

## The five conditions

| Condition | Entity text | Type labels | Bounding boxes | Relations |
|---|---|---|---|---|
| Raw | ✓ | ✗ | ✗ | ✗ |
| Flat | ✓ | Gold | Gold | ✗ |
| Predicted Graph | ✓ | Gold | Gold | Heuristic-predicted |
| Oracle Graph | ✓ | Gold | Gold | Gold |
| Shuffled Graph | ✓ | Gold | Gold | Wrong (degree-preserving derangement) |

All five are built from FUNSD's own text/entities (`src/agentic_docs/representations/`) — never
from a re-OCR'd image — so only the amount of structure exposed to the model varies, never the
underlying content (enforced by `test_representations.py`'s content-set-equality and box-parity
guards). Flat and every graph condition carry identical bounding boxes, so a Predicted-Graph result
can only be attributed to the relations, not to layout information Flat lacks.

**Oracle Graph** measures faithful relay under complete information (does the agent accurately
report what it's already been given) — not relational reasoning, since an agent handed the exact
answer mapping isn't reasoning about anything.

**Predicted Graph** uses a deterministic layout-proximity heuristic
(`src/agentic_docs/heuristic/proximity.py`): each answer entity is linked to the nearest question
entity that precedes it in reading order. Chosen over an LLM-based linker (would entangle results
with the extraction model's self-consistency) and over existing pretrained FUNSD-linking models
(opaque training history against FUNSD risks test-set contamination). Validated on an internal
holdout carved from the combined document dataset via `heuristic/validate.py`.

**Shuffled Graph** is a negative control: the scorable question→answer edges are replaced with a
degree-preserving derangement (same number of answers per question, guaranteed no correct edge
survives) within each document. Structural edges (e.g. headers) pass through unchanged since
they're never scored. A question whose degree is unique within its own document (no same-degree
partner to swap with) keeps its correct edge but is excluded from Shuffled-condition scoring.

## The self-verifying agent

A 2-node graph (`src/agentic_docs/agent_graph.py`, built with LangGraph for flow control only — the
LLM calls go through this project's own lightweight `LLMClient` abstraction, not a LangChain chat
model): an **extract** node produces initial question→answer pairs, a forced **verify** node
re-shows the same representation plus the extract node's own output and is instructed to correct
any ungrounded or inconsistent answers. Both the initial and final answers are retained and scored.

Prompts (`src/agentic_docs/skills/*/SKILL.md`) are byte-identical across all five conditions
(the final prompt round, preserved in `docs/prompt_versions/run3_final_SKILL.md`); only the inserted
document representation differs. This is enforced by `test_prompt_builder.py`. The model can tell
the conditions apart only from the data, never from the wording. Earlier prompt rounds are kept in
`docs/prompt_versions/` for provenance.

## Models

Claude Sonnet 5, GPT-5, Qwen3.6-27B (open-weight, via Groq), Gemini 3.7 Flash. Each model's
temperature/thinking config (`src/agentic_docs/config.py`) is fixed per-model and held constant
across all conditions. Some of these APIs reject an explicit temperature outright, and none can
fully disable reasoning, so each model runs at its lowest reasoning setting that keeps reasoning
enabled. The comparison is therefore always within-model (same model, same settings, only the
representation varies), never a cross-model comparison of whose configuration is more aggressive.

## Scoring

`src/agentic_docs/metrics.py` computes, per predicted answer against gold: exact match, a fuzzy/
token-similarity score (the practical "correct" threshold), omission (gold non-blank, prediction
blank), hallucination (a non-blank prediction unsupported anywhere in the source), and
misassociation (a non-blank prediction that matches some other field's real value — a distinct
failure mode from hallucination). Aggregate comparisons use a document-level paired bootstrap
(resampling whole documents, not individual fields, since fields from the same document are
correlated) rather than treating every scored field as an independent sample.

Correction/regression (not a naive initial-vs-final accuracy delta, which has a ceiling-effect
problem) measures, among initially-wrong answers, what fraction the verify step corrected, and
among initially-correct answers, what fraction it broke.

## Pipeline

```
docs download   -> data/documents_all.json
docs run        -> results/extractions/run.<model>.json (resumable, one partial per model)
docs analyze    -> results/analysis/summary.json
docs visualize  -> results/analysis/figures/*.png, results/analysis/figures/summary_table.csv
```

`docs run` is resumable — each model writes its own partial file, keyed by
`<document_id>|<condition>|<model>`; a failed cell is retried on the next run, a succeeded cell is
skipped. `--status` reports progress without making any API calls.

`docs analyze`'s document-level bootstrap (RQ2 headline) is restricted to documents scored on both
sides of a comparison; a document whose cell parse-errored under only one condition is dropped from
both sides rather than silently biasing the comparison, and the drop count is surfaced as a warning.
