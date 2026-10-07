"""Generates the two NEW figures the paper rewrite needs (2026-09-03), from already-computed
analysis JSON only -- no new experiments. Ad hoc script, same pattern as run3_bucket_diagnostic.py.

1. Failure-mode decomposition: Heuristic-Flat delta per bucket (edge_correct/edge_wrong/no_edge),
   per model, with 95% bootstrap error bars -- the paper's single strongest visual per both
   reviewers.
2. Exploratory prompt-regime sensitivity: wrong-edge-consistency rate across Run1/Run2/Run3, one
   panel per graph condition (Heuristic, Deranged), GROUPED BARS not a line plot -- a line would
   visually imply a continuous dose axis across three qualitatively different prompt regimes, which
   both external reviewers explicitly banned as "dose-response" framing.

Run with: .venv/Scripts/python scripts/final_paper_figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]
MODEL_LABELS = {"claude-sonnet-5": "Claude", "gpt-5": "GPT-5", "qwen3.6-27b": "Qwen", "gemini-3.7-flash": "Gemini"}
BUCKET_LABELS = {"edge_correct": "Edge correct", "edge_wrong": "Edge wrong", "no_edge": "No edge"}


def plot_bucket_failure_mode(out_path: str):
    d = json.load(open("results/analysis/final_paper_numbers.json", encoding="utf-8"))
    bucket_summary = d["bucket_summary_run3"]
    buckets = ["edge_correct", "edge_wrong", "no_edge"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(MODELS)
    x = range(len(buckets))
    colors = {"claude-sonnet-5": "tab:blue", "gpt-5": "tab:orange", "qwen3.6-27b": "tab:green",
              "gemini-3.7-flash": "tab:red"}

    for i, model in enumerate(MODELS):
        heights = [bucket_summary[model][b]["delta_point"] * 100 for b in buckets]
        ci_lo = [bucket_summary[model][b]["delta_ci"][0] * 100 for b in buckets]
        ci_hi = [bucket_summary[model][b]["delta_ci"][1] * 100 for b in buckets]
        err_lo = [h - lo for h, lo in zip(heights, ci_lo)]
        err_hi = [hi - h for h, hi in zip(heights, ci_hi)]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, heights, width=width, label=MODEL_LABELS[model], color=colors[model],
               yerr=[err_lo, err_hi], capsize=2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([xi + width * (len(MODELS) - 1) / 2 for xi in x])
    ax.set_xticklabels([BUCKET_LABELS[b] for b in buckets])
    ax.set_ylabel("Heuristic $-$ Flat accuracy ($\\Delta$ pp)")
    ax.set_title("Failure-mode decomposition, Run 3 (95% document-bootstrap CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_prompt_sensitivity(out_path: str):
    d = json.load(open("results/analysis/final_paper_numbers.json", encoding="utf-8"))
    traj = d["wrong_edge_consistency_trajectory"]
    runs = ["run1", "run2", "run3"]
    run_labels = ["Run 1", "Run 2", "Run 3"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, cond, title in zip(axes, ["predicted_graph", "shuffled_graph"], ["Heuristic", "Deranged"]):
        width = 0.8 / len(runs)
        x = range(len(MODELS))
        for i, run in enumerate(runs):
            heights = [traj[cond][m][run]["rate"] * 100 for m in MODELS]
            offsets = [xi + i * width for xi in x]
            ax.bar(offsets, heights, width=width, label=run_labels[i], color=colors[i])
        ax.set_xticks([xi + width * (len(runs) - 1) / 2 for xi in x])
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_title(f"{title} graph")
        ax.set_ylabel("Wrong-edge-consistent rate (%)")

    axes[1].legend(fontsize=8)
    fig.suptitle("Exploratory: wrong-edge-consistency across three prompt regimes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_bucket_failure_mode("paper/bucket_failure_mode.png")
    plot_prompt_sensitivity("paper/prompt_sensitivity_bars.png")
