# Prompt version history

Snapshots of the extract/verify input-specific prompt block used in each experimental run. The
live templates are `src/agentic_docs/skills/{extract_raw,extract_flat,extract_graph}/SKILL.md`,
which get overwritten in place each time the prompt changes — these files preserve what each run
actually used, independent of git history (kept even after any future repo cleanup, since git log
alone won't survive a code snapshot shared outside this repo).

Only the `<!-- INPUT-SPECIFIC:BEGIN -->...END` block differs between conditions/runs. The
surrounding instructions block has been identical since the project's first commit.

- **Run 1** (`run1_extract_{raw,flat,graph}_SKILL.md`, git commit `5eecbb5`): each condition had
  its own hand-written description of that representation's structure. `extract_graph`'s version
  told the model to use the shown links "as your primary guide" but warned they "may contain
  errors" and to use its "own judgment... instead of following the link blindly" — an instruction
  with no analog in `extract_raw`/`extract_flat`, later found to confound RQ2/RQ4/RQ5 (the paper's
  primary comparisons all assume representation is the only manipulated variable).

- **Run 2** (`run2_condition_blind_SKILL.md`, git commit `194ac3f`, one file — byte-identical
  across all 5 conditions by design): rewritten to remove all condition-specific wording, including
  the distrust instruction. Diagnosed afterward (see the paper's Appendix A)
  to have overcorrected: removing the link explanation entirely, with no guidance on what a
  missing link means, caused a `no_edge` blank-rate collapse in Qwen/Gemini (e.g. Qwen's no-edge
  blank rate rose from 18.1% to 53.2%) alongside sharply increased same-question wrong-edge
  trust (e.g. Qwen's wrong-edge-consistency rose from 5.2% to 25.0%).

- **Run 3** (`run3_final_SKILL.md`, one file — byte-identical across all 5 conditions, final
  locked design): reintroduces a neutral definition of what a link is and that some questions
  lack one, without any reliability-loaded language (no "correct/incorrect," "trust/distrust,"
  "plausible," "primary guide," "blindly," "override," or "if no link, then..." procedural
  gating — enforced by two automated regression tests in `tests/test_prompt_builder.py`). Designed
  and reviewed across three rounds with two independent external reviewers (Claude + ChatGPT)
  before execution.
