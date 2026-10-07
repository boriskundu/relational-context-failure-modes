import csv

from agentic_docs.visualize import (
    export_summary_csv,
    generate_all,
    plot_accuracy_by_condition_model,
    plot_rq2_headline_bootstrap,
    plot_rq5_correction_regression,
)

_SUMMARY = {
    "by_condition_model": {
        "flat|m1": {"n": 10, "accuracy": 0.5, "exact_match_rate": 0.4, "hallucination_rate": 0.1,
                    "omission_rate": 0.1, "misassociation_rate": 0.0},
        "predicted_graph|m1": {"n": 10, "accuracy": 0.7, "exact_match_rate": 0.6,
                                "hallucination_rate": 0.05, "omission_rate": 0.05,
                                "misassociation_rate": 0.0},
    },
    "rq5_correction": {
        "flat|m1": {"correction_rate": 0.3, "regression_rate": 0.1, "net_correction": 2},
        "predicted_graph|m1": {"correction_rate": 0.5, "regression_rate": 0.05, "net_correction": 4},
    },
    "rq2_headline_predicted_vs_flat": {
        "m1": {"point_estimate_a": 0.7, "point_estimate_b": 0.5, "point_delta": 0.2,
               "ci_low": 0.05, "ci_high": 0.35, "n_documents": 10, "n_boot": 1000},
    },
}


def test_plot_accuracy_by_condition_model_writes_a_file(tmp_path):
    out = plot_accuracy_by_condition_model(_SUMMARY, tmp_path / "acc.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_rq5_correction_regression_writes_a_file(tmp_path):
    out = plot_rq5_correction_regression(_SUMMARY, tmp_path / "rq5.png")
    assert out.exists()


def test_plot_rq2_headline_bootstrap_writes_a_file(tmp_path):
    out = plot_rq2_headline_bootstrap(_SUMMARY, tmp_path / "rq2.png")
    assert out.exists()


def test_export_summary_csv_has_expected_rows(tmp_path):
    out = export_summary_csv(_SUMMARY, tmp_path / "summary.csv")
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    flat_row = next(r for r in rows if r["condition"] == "flat")
    assert flat_row["accuracy"] == "0.5"
    assert flat_row["correction_rate"] == "0.3"


def test_generate_all_writes_four_outputs(tmp_path):
    paths = generate_all(_SUMMARY, tmp_path)
    assert len(paths) == 4
    assert all(p.exists() for p in paths)
