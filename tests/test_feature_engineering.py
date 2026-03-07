import pandas as pd

from jets_project.feature_engineering import (
    build_model_matrix,
    build_team_week_features,
    prepare_contract_features,
    prepare_injury_features,
)


def _sample_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2024,
                "week": 1,
                "game_date": pd.Timestamp("2024-09-10"),
                "game_type": "REG",
                "home_team": "NYJ",
                "away_team": "BUF",
                "home_score": 20,
                "away_score": 10,
                "stadium": "A",
                "score_diff": 10,
                "home_win": 1,
            },
            {
                "game_id": "g2",
                "season": 2024,
                "week": 2,
                "game_date": pd.Timestamp("2024-09-17"),
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "NYJ",
                "home_score": 14,
                "away_score": 7,
                "stadium": "B",
                "score_diff": 7,
                "home_win": 1,
            },
        ]
    )


def test_pregame_features_are_shifted() -> None:
    team_week = build_team_week_features(
        games=_sample_games(),
        pbp_team_game=pd.DataFrame(),
        team_stats=pd.DataFrame(),
        player_week_features=pd.DataFrame(),
        injury_features=pd.DataFrame(),
        ngs_team_week=pd.DataFrame(),
        contract_features=pd.DataFrame(),
        staff_features=pd.DataFrame(),
    )

    jets_week_two = team_week[(team_week["team"] == "NYJ") & (team_week["week"] == 2)].iloc[0]
    bills_week_two = team_week[(team_week["team"] == "BUF") & (team_week["week"] == 2)].iloc[0]

    assert jets_week_two["pregame_win_season_avg"] == 1.0
    assert jets_week_two["pregame_point_diff_season_avg"] == 10.0
    assert bills_week_two["pregame_win_season_avg"] == 0.0
    assert bills_week_two["pregame_point_diff_season_avg"] == -10.0


def test_build_model_matrix_creates_home_away_diffs() -> None:
    team_week = build_team_week_features(
        games=_sample_games(),
        pbp_team_game=pd.DataFrame(),
        team_stats=pd.DataFrame(),
        player_week_features=pd.DataFrame(),
        injury_features=pd.DataFrame(),
        ngs_team_week=pd.DataFrame(),
        contract_features=pd.DataFrame(),
        staff_features=pd.DataFrame(),
    )

    matrix = build_model_matrix(team_week)
    week_two = matrix[matrix["week"] == 2].iloc[0]

    assert len(matrix) == 2
    assert week_two["home_team"] == "BUF"
    assert week_two["away_team"] == "NYJ"
    assert week_two["home_win"] == 1
    assert week_two["pregame_win_season_avg_diff"] == -1.0


def test_prepare_injury_features_aggregates_statuses() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "team": "NYJ",
                "gsis_id": "1",
                "position": "QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate In Practice",
            },
            {
                "season": 2024,
                "week": 1,
                "team": "NYJ",
                "gsis_id": "2",
                "position": "LT",
                "report_status": "Questionable",
                "practice_status": "Limited Participation in Practice",
            },
        ]
    )

    features = prepare_injury_features(injuries)
    row = features.iloc[0]

    assert row["injury_report_count"] == 2
    assert row["injury_out_count"] == 1
    assert row["injury_questionable_count"] == 1
    assert row["injury_dnp_count"] == 1
    assert row["injury_limited_count"] == 1
    assert row["injury_skill_out_count"] == 1
    assert row["injury_oline_out_count"] == 0


def test_prepare_contract_features_builds_team_season_rollup() -> None:
    contracts = pd.DataFrame(
        [
            {
                "gsis_id": "qb1",
                "position": "QB",
                "cols": [
                    {"year": "2024", "team": "New York Jets", "cap_number": 30.0, "guaranteed_salary": 20.0, "cash_paid": 25.0, "cap_percent": 0.12}
                ],
            },
            {
                "gsis_id": "wr1",
                "position": "WR",
                "cols": [
                    {"year": "2024", "team": "New York Jets", "cap_number": 10.0, "guaranteed_salary": 5.0, "cash_paid": 8.0, "cap_percent": 0.04}
                ],
            },
        ]
    )

    features = prepare_contract_features(contracts)
    row = features.iloc[0]

    assert row["team"] == "NYJ"
    assert row["contract_player_count"] == 2
    assert row["contract_total_cap_number"] == 40.0
    assert row["contract_qb_cap_share"] == 0.75
    assert row["contract_top3_cap_share"] == 1.0


def test_build_team_week_features_keeps_current_week_injury_signals() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": "NYJ",
                "gsis_id": "1",
                "position": "QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate In Practice",
            },
        ]
    )

    team_week = build_team_week_features(
        games=_sample_games(),
        pbp_team_game=pd.DataFrame(),
        team_stats=pd.DataFrame(),
        player_week_features=pd.DataFrame(),
        injury_features=prepare_injury_features(injuries),
        ngs_team_week=pd.DataFrame(),
        contract_features=pd.DataFrame(),
        staff_features=pd.DataFrame(),
    )

    jets_week_one = team_week[(team_week["team"] == "NYJ") & (team_week["week"] == 1)].iloc[0]
    jets_week_two = team_week[(team_week["team"] == "NYJ") & (team_week["week"] == 2)].iloc[0]

    assert jets_week_one["pregame_injury_out_count"] == 0.0
    assert jets_week_two["pregame_injury_out_count"] == 1.0
    assert jets_week_two["pregame_injury_impact_score"] > 0.0
