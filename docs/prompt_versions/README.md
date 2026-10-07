# Prompt version history

Snapshots of the extract/verify input-specific prompt block used in each experimental run. The
live templates are `src/agentic_docs/skills/{extract_raw,extract_flat,extract_graph}/SKILL.md`;
the files here preserve what each run actually used. Run numbers match the paper's prompt rounds
(Run 1 = Round 1, and so on).

Only the `<!-- INPUT-SPECIFIC:BEGIN -->...END` block differs between conditions and runs. The
surrounding instructions block is identical across runs.

- **Run 1** (`run1_extract_{raw,flat,graph}_SKILL.md`): each condition had its own hand-written
  description of that representation's structure. `extract_graph`'s version told the model to use
  the shown links "as your primary guide" but warned they "may contain errors" and to use its "own
  judgment... instead of following the link blindly". `extract_raw` and `extract_flat` had no
  analogous instruction, which confounds the comparisons, since representation is meant to be the
  only manipulated variable.

- **Run 2** (`run2_condition_blind_SKILL.md`, one file, byte-identical across all five conditions):
  rewritten to remove all condition-specific wording, including the skepticism instruction. It
  also removed the generic link explanation, so Run 1 and Run 2 differ in more than one way. Under
  this prompt Qwen and Gemini gave many more blank answers on questions with no shown edge, which
  is consistent with treating an unexplained absence of a links field as license to answer blank
  (see the paper's Appendix A).

- **Run 3** (`run3_final_SKILL.md`, one file, byte-identical across all five conditions, the
  primary evaluation in the paper): reintroduces a neutral definition of what a link is and that
  some questions have none, without reliability-loaded language (no "correct/incorrect",
  "trust/distrust", "plausible", "primary guide", "blindly", "override", or "if no link, then..."
  gating), enforced by automated regression tests in `tests/test_prompt_builder.py`. Restoring the
  explanation did not reliably remove the Run 2 blank-answer pattern (see the paper).
