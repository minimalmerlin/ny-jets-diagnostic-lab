import pandas as pd

from jets_project.reporting import render_markdown_report


def test_render_markdown_report_includes_model_and_root_cause_findings() -> None:
    jets_season_summary = pd.DataFrame(
        [
            {
                "season": 2025,
                "team": "NYJ",
                "wins": 5,
                "losses": 12,
                "win_pct": 5 / 17,
                "avg_point_diff": -6.2,
                "league_avg_win_pct": 0.5,
                "league_avg_point_diff": 0.0,
            }
        ]
    )
    root_causes = pd.DataFrame(
        [
            {
                "metric": "offense_sack_rate",
                "jets_value": 0.08,
                "league_avg": 0.05,
                "problem_score": 12.5,
            }
        ]
    )
    evaluation_metrics = pd.DataFrame(
        [
            {"model_name": "gradient_boosting", "test_season": 2024, "log_loss": 0.64, "brier_score": 0.22, "roc_auc": 0.66},
            {"model_name": "gradient_boosting", "test_season": 2025, "log_loss": 0.63, "brier_score": 0.21, "roc_auc": 0.67},
            {"model_name": "logistic_regression", "test_season": 2024, "log_loss": 0.69, "brier_score": 0.24, "roc_auc": 0.61},
            {"model_name": "logistic_regression", "test_season": 2025, "log_loss": 0.68, "brier_score": 0.23, "roc_auc": 0.62},
            {"model_name": "naive_home_rate", "test_season": 2024, "log_loss": 0.70, "brier_score": 0.25, "roc_auc": 0.50},
            {"model_name": "naive_home_rate", "test_season": 2025, "log_loss": 0.71, "brier_score": 0.26, "roc_auc": 0.50},
        ]
    )
    backtest_diagnostics = pd.DataFrame(
        [
            {
                "model_name": "gradient_boosting",
                "test_season": 2024,
                "train_log_loss": 0.60,
                "test_log_loss": 0.64,
                "train_brier_score": 0.20,
                "test_brier_score": 0.22,
                "log_loss_gap": 0.04,
                "brier_score_gap": 0.02,
            }
        ]
    )

    report = render_markdown_report(
        jets_season_summary=jets_season_summary,
        root_causes=root_causes,
        evaluation_metrics=evaluation_metrics,
        backtest_diagnostics=backtest_diagnostics,
        focus_team="NYJ",
    )

    assert "Gradient-Boosting-Modell" in report
    assert "Offense-Sack-Rate" in report
    assert "2/2" in report
    assert "0.0400" in report
