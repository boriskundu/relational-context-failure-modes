"""
Turns an analyze.py summary JSON into figures + a flat CSV table for the paper.

Pure functions over the summary dict (no I/O beyond writing the requested output path) so they're
offline-testable against a hand-built summary — never regenerate a summary by re-running the grid.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never requires a display, safe in CI/CLI use
import matplotlib.pyplot as plt

from agentic_docs.config import ANALYSIS_DIR
from agentic_docs.runner import CONDITIONS

CONDITION_ORDER = CONDITIONS  # raw, flat, predicted_graph, oracle_graph, shuffled_graph


def _condition_model_pairs(by_condition_model: dict) -> tuple[list[str], list[str]]:
    conditions = [
        c for c in CONDITION_ORDER if any(k.startswith(f"{c}|") for k in by_condition_model)
    ]
    models = sorted({k.split("|", 1)[1] for k in by_condition_model})
    return conditions, models


def plot_accuracy_by_condition_model(summary: dict, out_path: Path) -> Path:
    """Grouped bar chart: one group per condition, one bar per model, height = fuzzy-match accuracy
    on final answers. This is the main figure for RQ1-RQ4 (Raw/Flat/Predicted/Oracle/Shuffled)."""
    by_condition_model = summary["by_condition_model"]
    conditions, models = _condition_model_pairs(by_condition_model)

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.6), 5))
    width = 0.8 / max(len(models), 1)
    x = range(len(conditions))
    for i, model in enumerate(models):
        heights = [
            (by_condition_model.get(f"{c}|{model}", {}) or {}).get("accuracy") or 0.0
            for c in conditions
        ]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, heights, width=width, label=model)

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (fuzzy match, final answer)")
    ax.set_ylim(0, 1)
    ax.set_title("Field-completion accuracy by condition and model")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rq5_correction_regression(summary: dict, out_path: Path) -> Path:
    """Grouped bar chart: for each condition, correction rate and regression rate, averaged across
    models (unweighted mean of the per-model rates) — the RQ5 self-verification result."""
    rq5 = summary["rq5_correction"]
    conditions, models = _condition_model_pairs(rq5)

    def _mean(key: str, cond: str) -> float:
        vals = [rq5.get(f"{cond}|{m}", {}).get(key) for m in models]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    correction = [_mean("correction_rate", c) for c in conditions]
    regression = [_mean("regression_rate", c) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.6), 5))
    x = range(len(conditions))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        correction,
        width=width,
        label="Correction rate",
        color="tab:green",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        regression,
        width=width,
        label="Regression rate",
        color="tab:red",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Rate (mean across models)")
    ax.set_ylim(0, 1)
    ax.set_title("RQ5: self-verification correction vs. regression, by condition")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rq2_headline_bootstrap(summary: dict, out_path: Path) -> Path:
    """Bar chart with bootstrap CI error bars: per-model Predicted-vs-Flat accuracy delta — the
    prespecified primary RQ2 comparison."""
    rq2 = summary["rq2_headline_predicted_vs_flat"]
    models = sorted(rq2)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 5.5))
    deltas = [rq2[m]["point_delta"] for m in models]
    lo_err = [
        rq2[m]["point_delta"] - rq2[m]["ci_low"] if rq2[m]["ci_low"] is not None else 0
        for m in models
    ]
    hi_err = [
        rq2[m]["ci_high"] - rq2[m]["point_delta"] if rq2[m]["ci_high"] is not None else 0
        for m in models
    ]
    ax.bar(models, deltas, yerr=[lo_err, hi_err], capsize=5, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy delta (Predicted Graph - Flat)")
    ax.set_title("RQ2 (primary): Predicted Graph vs. Flat, 95% bootstrap CI", fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def export_summary_csv(summary: dict, out_path: Path) -> Path:
    """Flat per-(condition, model) table — accuracy + RQ5 rates — for the paper's appendix tables."""
    by_condition_model = summary["by_condition_model"]
    rq5 = summary["rq5_correction"]
    conditions, models = _condition_model_pairs(by_condition_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "model",
                "n",
                "accuracy",
                "exact_match_rate",
                "hallucination_rate",
                "omission_rate",
                "misassociation_rate",
                "n_relation_following_applicable",
                "relation_following_error_rate",
                "correction_rate",
                "regression_rate",
                "net_correction",
            ]
        )
        for cond in conditions:
            for model in models:
                key = f"{cond}|{model}"
                agg = by_condition_model.get(key, {})
                corr = rq5.get(key, {})
                writer.writerow(
                    [
                        cond,
                        model,
                        agg.get("n"),
                        agg.get("accuracy"),
                        agg.get("exact_match_rate"),
                        agg.get("hallucination_rate"),
                        agg.get("omission_rate"),
                        agg.get("misassociation_rate"),
                        agg.get("n_relation_following_applicable"),
                        agg.get("relation_following_error_rate"),
                        corr.get("correction_rate"),
                        corr.get("regression_rate"),
                        corr.get("net_correction"),
                    ]
                )
    return out_path


def generate_all(summary: dict, out_dir: Path) -> list[Path]:
    return [
        plot_accuracy_by_condition_model(summary, out_dir / "accuracy_by_condition_model.png"),
        plot_rq5_correction_regression(summary, out_dir / "rq5_correction_regression.png"),
        plot_rq2_headline_bootstrap(summary, out_dir / "rq2_headline_bootstrap.png"),
        export_summary_csv(summary, out_dir / "summary_table.csv"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docs visualize")
    parser.add_argument("--summary", default=None, help="Path to summary.json (default inferred)")
    parser.add_argument(
        "--out", default=None, help="Output directory (default: results/analysis/figures)"
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.summary) if args.summary else ANALYSIS_DIR / "summary.json"
    out_dir = Path(args.out) if args.out else ANALYSIS_DIR / "figures"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = generate_all(summary, out_dir)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
