import pandas as pd

from jets_project.analysis_utils import build_root_causes


def test_build_root_causes_ignores_non_actionable_tracking_metrics() -> None:
    team_season_summary = pd.DataFrame(
        [
            {
                "season": 2025,
                "team": "NYJ",
                "wins": 3,
                "avg_point_diff": -10.0,
                "offense_sack_rate": 0.05,
                "point_diff": -10.0,
                "win": 0.2,
                "ngs_passing_player_jersey_number": 12.0,
            },
            {
                "season": 2025,
                "team": "BUF",
                "wins": 11,
                "avg_point_diff": 7.0,
                "offense_sack_rate": 0.02,
                "point_diff": 7.0,
                "win": 0.8,
                "ngs_passing_player_jersey_number": 17.0,
            },
        ]
    )
    feature_importance = pd.DataFrame(
        [
            {
                "feature": "pregame_offense_sack_rate_season_avg_diff",
                "importance": 10.0,
                "direction": -1.0,
            },
            {
                "feature": "pregame_ngs_passing_player_jersey_number_season_avg_diff",
                "importance": 100.0,
                "direction": 1.0,
            },
            {
                "feature": "pregame_point_diff_rolling4_diff",
                "importance": 90.0,
                "direction": 1.0,
            },
            {
                "feature": "pregame_win_season_avg_diff",
                "importance": 80.0,
                "direction": 1.0,
            },
        ]
    )

    root_causes = build_root_causes(team_season_summary, feature_importance, focus_team="NYJ")

    assert "offense_sack_rate" in root_causes["metric"].tolist()
    assert "ngs_passing_player_jersey_number" not in root_causes["metric"].tolist()
    assert "point_diff" not in root_causes["metric"].tolist()
    assert "win" not in root_causes["metric"].tolist()
