from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .modeling import EnsembleModel


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


def _matching_diff_features(feature_columns: list[str], base_metric: str) -> list[str]:
    pattern = re.compile(rf"^pregame_{re.escape(base_metric)}_(rolling4|season_avg)_diff$")
    return [column for column in feature_columns if pattern.match(column)]


def _clip_to_ranges(frame: pd.DataFrame, feature_ranges: dict[str, dict[str, float]], columns: list[str]) -> pd.DataFrame:
    clipped = frame.copy()
    for column in columns:
        bounds = feature_ranges.get(column)
        if bounds is None:
            continue
        clipped[column] = clipped[column].clip(lower=bounds["q05"], upper=bounds["q95"])
    return clipped


def _predict_probability_distribution(
    model_bundle: dict,
    model_name: str,
    features: pd.DataFrame,
) -> np.ndarray:
    decision_support_models = model_bundle.get("decision_support_models", {})
    if model_name in decision_support_models:
        model = decision_support_models[model_name]
        if isinstance(model, EnsembleModel):
            return model.predict_member_probabilities(features)
    model = model_bundle["models"][model_name]
    probabilities = model.predict_proba(features)[:, 1]
    return probabilities.reshape(1, -1)


def _decision_confidence(positive_share: float) -> str:
    if positive_share >= 0.8:
        return "hoch"
    if positive_share >= 0.65:
        return "mittel"
    return "niedrig"


def simulate_team_improvement(
    game_features: pd.DataFrame,
    model_bundle: dict,
    focus_team: str,
    season: int | None,
    adjustments: dict[str, float],
    model_name: str | None = None,
) -> pd.DataFrame:
    if model_name is None:
        model_name = model_bundle.get("preferred_model_name", "gradient_boosting")
    feature_columns = model_bundle["feature_columns"]
    feature_ranges = model_bundle["feature_ranges"]

    season_frame = game_features.copy()
    if season is not None:
        season_frame = season_frame[season_frame["season"] == season].copy()
    season_frame = season_frame[
        (season_frame["home_team"] == focus_team) | (season_frame["away_team"] == focus_team)
    ].copy()
    if season_frame.empty:
        return pd.DataFrame(columns=["week", "baseline_win_probability", "scenario_win_probability"])

    scenario = season_frame.copy()
    affected_columns: list[str] = []
    for metric, delta in adjustments.items():
        for diff_feature in _matching_diff_features(feature_columns, metric):
            jets_home_mask = scenario["home_team"] == focus_team
            jets_away_mask = scenario["away_team"] == focus_team
            scenario.loc[jets_home_mask, diff_feature] = scenario.loc[jets_home_mask, diff_feature] + delta
            scenario.loc[jets_away_mask, diff_feature] = scenario.loc[jets_away_mask, diff_feature] - delta
            affected_columns.append(diff_feature)
    scenario = _clip_to_ranges(scenario, feature_ranges, affected_columns)

    baseline_home_probs_distribution = _predict_probability_distribution(
        model_bundle=model_bundle,
        model_name=model_name,
        features=season_frame[feature_columns],
    )
    scenario_home_probs_distribution = _predict_probability_distribution(
        model_bundle=model_bundle,
        model_name=model_name,
        features=scenario[feature_columns],
    )

    baseline_home_probs = baseline_home_probs_distribution.mean(axis=0)
    scenario_home_probs = scenario_home_probs_distribution.mean(axis=0)

    baseline_win_probability = np.where(
        season_frame["home_team"] == focus_team,
        baseline_home_probs,
        1 - baseline_home_probs,
    )
    scenario_win_probability = np.where(
        season_frame["home_team"] == focus_team,
        scenario_home_probs,
        1 - scenario_home_probs,
    )
    baseline_win_distribution = np.where(
        (season_frame["home_team"] == focus_team).to_numpy()[None, :],
        baseline_home_probs_distribution,
        1 - baseline_home_probs_distribution,
    )
    scenario_win_distribution = np.where(
        (season_frame["home_team"] == focus_team).to_numpy()[None, :],
        scenario_home_probs_distribution,
        1 - scenario_home_probs_distribution,
    )
    delta_distribution = scenario_win_distribution - baseline_win_distribution

    result = season_frame[
        ["season", "week", "game_id", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    result["baseline_win_probability"] = baseline_win_probability
    result["scenario_win_probability"] = scenario_win_probability
    result["baseline_win_probability_p10"] = np.quantile(baseline_win_distribution, 0.10, axis=0)
    result["baseline_win_probability_p90"] = np.quantile(baseline_win_distribution, 0.90, axis=0)
    result["scenario_win_probability_p10"] = np.quantile(scenario_win_distribution, 0.10, axis=0)
    result["scenario_win_probability_p90"] = np.quantile(scenario_win_distribution, 0.90, axis=0)
    result["win_probability_delta"] = result["scenario_win_probability"] - result["baseline_win_probability"]
    result["win_probability_delta_p10"] = np.quantile(delta_distribution, 0.10, axis=0)
    result["win_probability_delta_p90"] = np.quantile(delta_distribution, 0.90, axis=0)
    positive_share = (delta_distribution > 0).mean(axis=0)
    result["positive_scenario_share"] = positive_share
    result["decision_confidence"] = [
        _decision_confidence(value) for value in positive_share
    ]
    result["opponent"] = np.where(
        result["home_team"] == focus_team,
        result["away_team"],
        result["home_team"],
    )
    result["location"] = np.where(result["home_team"] == focus_team, "home", "away")
    result.attrs["model_name"] = model_name
    result.attrs["ensemble_size"] = int(baseline_win_distribution.shape[0])
    result.attrs["baseline_expected_wins_distribution"] = baseline_win_distribution.sum(axis=1)
    result.attrs["scenario_expected_wins_distribution"] = scenario_win_distribution.sum(axis=1)
    return result.sort_values(["season", "week"]).reset_index(drop=True)
