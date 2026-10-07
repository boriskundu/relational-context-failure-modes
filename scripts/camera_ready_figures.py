"""Camera-ready version of the failure-mode figure, split into two panels.

Figure 2 (failure-mode decomposition): the Heuristic-minus-Flat accuracy delta per edge bucket, as
two panels with their own y-scales (a gain for correct edges, losses for wrong and absent edges).
Figure 3 (prompt sensitivity): wrong-edge-consistency rate across the three prompt rounds. Both
are drawn from the released analysis JSON; no new analysis.

Run with: python scripts/camera_ready_figures.py [output_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["claude-sonnet-5", "gpt-5", "qwen3.6-27b", "gemini-3.7-flash"]
LABELS = {"claude-sonnet-5": "Claude", "gpt-5": "GPT-5", "qwen3.6-27b": "Qwen",
          "gemini-3.7-flash": "Gemini"}
COLORS = {"claude-sonnet-5": "tab:blue", "gpt-5": "tab:orange", "qwen3.6-27b": "tab:green",
          "gemini-3.7-flash": "tab:red"}
HATCHES = {"claude-sonnet-5": "", "gpt-5": "//", "qwen3.6-27b": "..", "gemini-3.7-flash": "xx"}
ANALYSIS_JSON = Path("results/analysis/final_paper_numbers.json")


def _draw(ax, summary, buckets, names):
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        pts = [summary[model][b]["delta_point"] * 100 for b in buckets]
        lo = [p - summary[model][b]["delta_ci"][0] * 100 for p, b in zip(pts, buckets)]
        hi = [summary[model][b]["delta_ci"][1] * 100 - p for p, b in zip(pts, buckets)]
        xs = [k + i * width for k in range(len(buckets))]
        ax.bar(xs, pts, width=width, color=COLORS[model], hatch=HATCHES[model],
               edgecolor="white", yerr=[lo, hi], capsize=2, label=LABELS[model])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([k + width * (len(MODELS) - 1) / 2 for k in range(len(buckets))])
    ax.set_xticklabels(names)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)


def plot_failure_modes(out_path: Path) -> None:
    data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))["bucket_summary_run3"]
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 2]})
    _draw(ax_a, data, ["edge_correct"], ["Correct edge"])
    _draw(ax_b, data, ["edge_wrong", "no_edge"], ["Wrong edge", "No edge"])
    ax_a.set_title("(a) Gain with a correct edge", fontsize=9)
    ax_b.set_title("(b) Loss with a wrong or absent edge", fontsize=9)
    ax_a.set_ylabel("Heuristic $-$ Flat accuracy (pp)", fontsize=9)
    ax_a.set_ylim(-1, 14)
    ax_b.set_ylim(-40, 1)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_prompt_sensitivity(out_path: Path) -> None:
    """Same data as the submitted appendix figure, labeled Round (not Run) to match the text."""
    traj = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))["wrong_edge_consistency_trajectory"]
    rounds = ["run1", "run2", "run3"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.6), sharey=True)
    for ax, cond, title in zip(axes, ["predicted_graph", "shuffled_graph"], ["Heuristic", "Deranged"]):
        width = 0.8 / len(rounds)
        for i, rnd in enumerate(rounds):
            heights = [traj[cond][m][rnd]["rate"] * 100 for m in MODELS]
            ax.bar([k + i * width for k in range(len(MODELS))], heights, width=width,
                   color=colors[i], label=f"Round {i + 1}")
        ax.set_xticks([k + width * (len(rounds) - 1) / 2 for k in range(len(MODELS))])
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        ax.set_title(f"{title} graph", fontsize=9)
        ax.set_ylabel("Wrong-edge-consistent (%)", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].set_ylim(0, 38)
    axes[0].legend(fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_failure_modes(out_dir / "bucket_failure_mode.png")
    plot_prompt_sensitivity(out_dir / "prompt_sensitivity_bars.png")
