"""Same ID-based bucket diagnostic as run3_bucket_diagnostic.py, re-run against Round 2's raw
extraction (results/extractions/run_condition_blind.json) instead of Round 3 -- needed for the
Round2-vs-Round3 no-edge blank-penalty and cross-round misassociation analyses requested in the
2026-09-03 review round (Section 4.4). Round 1 is deliberately excluded from this specific
comparison: its per-condition prompts weren't condition-invariant, the same reasoning the paper
already uses to exclude Round 1 from RQ2/RQ4.

Ad hoc script, not a permanent CLI capability. Run with:
    .venv/Scripts/python scripts/round2_bucket_diagnostic.py
"""
import json

from run3_bucket_diagnostic import MODELS, compute_bucket_records

RUN2_PATH = "results/extractions/run_condition_blind.json"
OUT_PATH = "results/analysis/round2_bucket_diagnostic.json"


def main():
    per_model_rows = compute_bucket_records(RUN2_PATH)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({m: per_model_rows[m] for m in MODELS}, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    for model in MODELS:
        print(f"{model}: n={len(per_model_rows[model])}")


if __name__ == "__main__":
    main()
