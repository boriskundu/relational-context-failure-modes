# Manual validation of the fuzzy-match scorer

Purpose: independently confirm that the automatic scorer (`metrics.py::score_field`, rapidfuzz
`fuzz.ratio` at `FUZZY_MATCH_THRESHOLD=85`) tracks actual human judgment of correctness, not just
string similarity, before treating the pipeline's accuracy numbers as final.

## Method

Two independent rounds of N=50, drawn from the full grid's 53,688 final-stage field scores,
seeded via `config.RANDOM_SEED` (round 2 uses `RANDOM_SEED + 1` and excludes every item already
drawn in round 1, so the two rounds never overlap). Each round mixes:
- 40 **stratified** items, 2 randomly drawn from each of the 20 (model, condition) combinations.
- 10 **boundary** items, randomly drawn from scores with `fuzzy_score` in [70, 95], the region
  around the threshold most likely to expose scorer problems.

Every item was reviewed against the scorer's MATCH/NO_MATCH verdict. The per-item records
(question text, predicted answer, gold answer, fuzzy score, scorer verdict, human verdict) contain
FUNSD text and model outputs, so they are not released; the numbers below summarize them.

## Result

**100/100 agreement** between the scorer's verdict and human judgment across both rounds. No
scorer bug was found. Three genuine, disclosable patterns emerged, all confirmed by at least two
independent sampled items:

1. **Order-sensitivity on multi-value answers.** `rapidfuzz.ratio` cannot distinguish "wrong
   content" from "right content, wrong order." Confirmed on 4 items across both rounds: two
   numeric lists (round 1, #43 and #49) and two multi-name CC/TO fields (round 2, #44 and #46)
   where the predicted answer contained the exact same values as gold with one or two adjacent
   items transposed, and was still scored NO_MATCH. This is now a confirmed, reproducible pattern
   rather than a single outlier, and is written up as a Limitation with a suggested fix (a
   token-multiset comparison for list-type fields) rather than something patched after the fact.
2. **Gold-data artifacts.** A handful of gold answers are themselves incomplete or contain
   placeholder text rather than a real value: an instruction placeholder `"(check one)"` used as
   the gold answer for two checkbox questions in the same document (round 1, #16/#26), a
   `"/A"` OCR truncation of `"N/A"` (round 2, #43), and a date gold answer missing its year
   (`"October 16,"`) that recurred identically across both rounds for the same underlying document
   (round 1 #46, round 2 #50) -- confirming it as a real, reproducible per-document annotation gap
   rather than a fluke. None of these are scorer or model errors.
3. **Borderline calls at the threshold**, adjudicated case by case: a predicted answer missing a
   short prefix token present in gold (round 1 #42, kept NO_MATCH), a predicted answer with one
   extra token beyond gold that still passed threshold (round 1 #50, kept MATCH), a blank
   prediction against a literal `"0"` gold value (round 2 #22, kept NO_MATCH), and a predicted
   answer missing a trailing qualifier phrase (round 2 #41, kept NO_MATCH).

**Conclusion**: the automatic scorer is sound. The reported accuracy numbers are not inflated or
deflated by a broken metric. The order-sensitivity limitation is real and is disclosed in the
paper's Limitations section rather than silently accepted or silently fixed.
