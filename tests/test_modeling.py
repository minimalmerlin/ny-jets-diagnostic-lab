import numpy as np
import pandas as pd

from jets_project.modeling import run_training_pipeline, select_training_features, top_actionable_metrics


def test_run_training_pipeline_returns_predictions_and_models() -> None:
    rows = []
    rng = np.random.default_rng(7)
    for season in range(2020, 2025):
        for week in range(1, 11):
            signal = rng.normal()
            rows.append(
                {
                    "game_id": f"{season}_{week}",
                    "season": season,
                    "week": week,
                    "home_team": "AAA",
                    "away_team": "BBB",
                    "home_win": int(signal > -0.2),
                    "pregame_signal_season_avg_diff": signal,
                    "pregame_turnover_rate_season_avg_diff": -signal / 2,
                }
            )
    frame = pd.DataFrame(rows)
    feature_cols = [
        "pregame_signal_season_avg_diff",
        "pregame_turnover_rate_season_avg_diff",
    ]

    results = run_training_pipeline(frame, feature_cols=feature_cols, min_train_seasons=2)

    assert not results.backtest_predictions.empty
    assert {"logistic_regression", "gradient_boosting"} <= set(results.final_models)
    assert not results.metrics.empty
    assert not results.diagnostics.empty
    assert not results.calibration_summary.empty
    assert not results.feature_importance.empty
    assert "raw_predicted_home_win_probability" in results.backtest_predictions.columns
    assert "ece" in results.metrics.columns


def test_top_actionable_metrics_excludes_non_informative_tracking_fields() -> None:
    importance = pd.DataFrame(
        [
            {"feature": "pregame_ngs_passing_player_jersey_number_season_avg_diff", "importance": 99.0},
            {"feature": "pregame_ngs_passing_weight_sum_season_avg_diff", "importance": 88.0},
            {"feature": "pregame_offense_sack_rate_season_avg_diff", "importance": 70.0},
            {"feature": "pregame_defense_takeaway_rate_season_avg_diff", "importance": 65.0},
        ]
    )

    top = top_actionable_metrics(importance, top_n=4)

    assert "offense_sack_rate" in top
    assert "defense_takeaway_rate" in top
    assert not any("player_jersey_number" in metric for metric in top)
    assert not any("weight_sum" in metric for metric in top)


def test_top_actionable_metrics_excludes_proxy_context_metrics() -> None:
    importance = pd.DataFrame(
        [
            {"feature": "pregame_point_diff_rolling4_diff", "importance": 99.0},
            {"feature": "pregame_win_season_avg_diff", "importance": 90.0},
            {"feature": "pregame_strength_of_schedule_diff", "importance": 80.0},
            {"feature": "pregame_offense_success_rate_rolling4_diff", "importance": 70.0},
        ]
    )

    top = top_actionable_metrics(importance, top_n=4)

    assert "offense_success_rate" in top
    assert "point_diff" not in top
    assert "win" not in top
    assert "strength_of_schedule" not in top


def test_select_training_features_prefers_stable_high_coverage_metrics() -> None:
    frame = pd.DataFrame(
        {
            "season": [2024] * 10,
            "pregame_offense_sack_rate_season_avg_diff": np.linspace(-0.2, 0.2, 10),
            "pregame_offense_sack_rate_rolling4_diff": np.linspace(-0.1, 0.1, 10),
            "pregame_ngs_passing_avg_air_distance_season_avg_diff": np.linspace(-1.0, 1.0, 10),
            "pregame_ngs_passing_avg_air_distance_rolling4_diff": np.linspace(-2.0, 2.0, 10),
            "pregame_ngs_rushing_expected_rush_yards_season_avg_diff": np.linspace(-3.0, 3.0, 10),
            "pregame_ngs_passing_player_jersey_number_season_avg_diff": np.arange(10),
            "pregame_constant_diff": np.ones(10),
            "pregame_low_coverage_season_avg_diff": [0.0, 1.0] + [np.nan] * 8,
            "pregame_strength_of_schedule_diff": np.linspace(-0.5, 0.5, 10),
        }
    )

    selected = select_training_features(frame, min_coverage=0.8)

    assert "pregame_offense_sack_rate_season_avg_diff" in selected
    assert "pregame_offense_sack_rate_rolling4_diff" in selected
    assert "pregame_ngs_passing_avg_air_distance_season_avg_diff" in selected
    assert "pregame_strength_of_schedule_diff" in selected
    assert "pregame_ngs_passing_avg_air_distance_rolling4_diff" not in selected
    assert "pregame_ngs_rushing_expected_rush_yards_season_avg_diff" not in selected
    assert "pregame_ngs_passing_player_jersey_number_season_avg_diff" not in selected
    assert "pregame_constant_diff" not in selected
    assert "pregame_low_coverage_season_avg_diff" not in selected
