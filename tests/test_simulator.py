import numpy as np
import pandas as pd

from jets_project.modeling import CalibratedModel, EnsembleModel, IdentityCalibrator
from jets_project.simulator import simulate_team_improvement


class FixedModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.repeat(self.probability, len(features))
        return np.column_stack([1 - probabilities, probabilities])


def test_simulate_team_improvement_returns_uncertainty_columns_and_attrs() -> None:
    features = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_1_NYJ_BUF",
                "home_team": "NYJ",
                "away_team": "BUF",
                "home_score": 0,
                "away_score": 0,
                "pregame_offense_sack_rate_rolling4_diff": -0.03,
                "pregame_offense_sack_rate_season_avg_diff": -0.02,
            }
        ]
    )
    members = [
        CalibratedModel(FixedModel(0.40), IdentityCalibrator(), "gradient_boosting"),
        CalibratedModel(FixedModel(0.50), IdentityCalibrator(), "gradient_boosting"),
        CalibratedModel(FixedModel(0.60), IdentityCalibrator(), "gradient_boosting"),
    ]
    model_bundle = {
        "models": {"gradient_boosting": members[0]},
        "decision_support_models": {"gradient_boosting": EnsembleModel(members, "gradient_boosting")},
        "preferred_model_name": "gradient_boosting",
        "feature_columns": [
            "pregame_offense_sack_rate_rolling4_diff",
            "pregame_offense_sack_rate_season_avg_diff",
        ],
        "feature_ranges": {
            "pregame_offense_sack_rate_rolling4_diff": {"q05": -0.2, "q95": 0.2},
            "pregame_offense_sack_rate_season_avg_diff": {"q05": -0.2, "q95": 0.2},
        },
    }

    scenario = simulate_team_improvement(
        game_features=features,
        model_bundle=model_bundle,
        focus_team="NYJ",
        season=2025,
        adjustments={"offense_sack_rate": -0.05},
    )

    assert not scenario.empty
    assert "baseline_win_probability_p10" in scenario.columns
    assert "scenario_win_probability_p90" in scenario.columns
    assert "win_probability_delta_p10" in scenario.columns
    assert "decision_confidence" in scenario.columns
    assert "positive_scenario_share" in scenario.columns
    assert scenario.attrs["ensemble_size"] == 3
    assert len(scenario.attrs["baseline_expected_wins_distribution"]) == 3
    assert len(scenario.attrs["scenario_expected_wins_distribution"]) == 3
    assert isinstance(scenario.attrs["baseline_expected_wins_distribution"], list)
    assert isinstance(scenario.attrs["scenario_expected_wins_distribution"], list)
    melted = scenario[["week", "baseline_win_probability", "scenario_win_probability"]].copy()
    melted.attrs = {}
    assert not melted.melt(
        id_vars="week",
        value_vars=["baseline_win_probability", "scenario_win_probability"],
        var_name="series",
        value_name="value",
    ).empty
