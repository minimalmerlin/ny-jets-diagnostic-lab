from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd

from .modeling import (
    _base_metric_name,
    build_decision_support_ensemble,
    run_training_pipeline,
    select_training_features,
    top_actionable_metrics,
)
from .logging_utils import configure_logging
from .settings import get_default_config
from .storage import list_sqlite_tables, read_sqlite_table, write_parquet, write_sqlite_table

LOGGER = logging.getLogger(__name__)
CORE_MODEL_DATASETS = (
    "pbp",
    "team_stats",
    "player_stats",
    "snap_counts",
    "rosters",
    "ngs_passing",
    "ngs_receiving",
    "injuries",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Win-Probability-Modelle trainieren.")
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--preferred-end-season", type=int, default=2025)
    return parser.parse_args()


def _select_training_end_season(model_frame: pd.DataFrame, preferred_end_season: int) -> int:
    season_counts = model_frame.groupby("season")["game_id"].nunique().sort_index()
    if season_counts.empty:
        raise RuntimeError("Keine Spiele fuer das Training vorhanden.")
    chosen = int(min(preferred_end_season, season_counts.index.max()))
    prior = season_counts[season_counts.index < chosen].tail(3)
    if chosen in season_counts.index and not prior.empty and season_counts.loc[chosen] < 0.8 * prior.median():
        return int(chosen - 1)
    return chosen


def _select_complete_training_end_season(
    sqlite_path,
    model_frame: pd.DataFrame,
    preferred_end_season: int,
) -> int:
    table_names = set(list_sqlite_tables(sqlite_path))
    if "dataset_availability" not in table_names:
        return _select_training_end_season(model_frame, preferred_end_season)

    availability = read_sqlite_table(sqlite_path, "dataset_availability")
    if availability.empty or "dataset_name" not in availability.columns:
        return _select_training_end_season(model_frame, preferred_end_season)

    seasonal = availability[availability["dataset_name"].isin(CORE_MODEL_DATASETS)].copy()
    if seasonal.empty:
        return _select_training_end_season(model_frame, preferred_end_season)

    loaded_by_dataset: dict[str, set[int]] = {}
    for row in seasonal.itertuples(index=False):
        raw_loaded = getattr(row, "loaded_seasons", "[]")
        try:
            loaded = {int(value) for value in json.loads(raw_loaded or "[]")}
        except (TypeError, ValueError, json.JSONDecodeError):
            loaded = set()
        loaded_by_dataset[getattr(row, "dataset_name")] = loaded

    season_counts = model_frame.groupby("season")["game_id"].nunique().sort_index()
    candidate_seasons = [int(season) for season in season_counts.index if int(season) <= preferred_end_season]
    for candidate in reversed(candidate_seasons):
        if all(candidate in loaded_by_dataset.get(dataset_name, set()) for dataset_name in CORE_MODEL_DATASETS):
            return _select_training_end_season(model_frame, candidate)
    return _select_training_end_season(model_frame, preferred_end_season)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = get_default_config()

    team_week_features = pd.read_parquet(config.paths.data_processed / "team_week_features.parquet")
    from .feature_engineering import build_model_matrix

    game_features = build_model_matrix(team_week_features, regular_season_only=config.regular_season_only)
    write_parquet(game_features, config.paths.model_matrix_path)
    write_sqlite_table(game_features, "game_features", config.paths.sqlite_path)

    training_end_season = _select_complete_training_end_season(
        config.paths.sqlite_path,
        game_features,
        args.preferred_end_season,
    )
    training_frame = game_features[game_features["season"] <= training_end_season].copy()
    feature_cols = select_training_features(training_frame)
    if not feature_cols:
        raise RuntimeError("Keine geeigneten Features fuer das Modelltraining gefunden.")
    results = run_training_pipeline(
        model_frame=training_frame,
        feature_cols=feature_cols,
        min_train_seasons=args.min_train_seasons,
    )

    feature_importance = results.feature_importance.copy()
    feature_importance["base_metric"] = feature_importance["feature"].map(_base_metric_name)

    predictions = results.backtest_predictions.merge(
        training_frame[["game_id", "season", "week", "home_team", "away_team", "home_win"]],
        on=["game_id", "season", "week"],
        how="left",
    )
    predictions["actual_home_win"] = predictions["home_win"]
    predictions = predictions.drop(columns=["home_win"])

    write_sqlite_table(predictions, "predictions", config.paths.sqlite_path)
    write_sqlite_table(results.metrics, "evaluation_metrics", config.paths.sqlite_path)
    write_sqlite_table(results.diagnostics, "backtest_diagnostics", config.paths.sqlite_path)
    write_sqlite_table(results.calibration_summary, "calibration_summary", config.paths.sqlite_path)
    write_sqlite_table(feature_importance, "feature_importance", config.paths.sqlite_path)
    write_parquet(predictions, config.paths.data_processed / "predictions.parquet")

    contender_metrics = results.metrics[results.metrics["model_name"] != "naive_home_rate"].copy()
    preferred_model_name = (
        contender_metrics.groupby("model_name", as_index=False)["log_loss"].mean().sort_values("log_loss").iloc[0][
            "model_name"
        ]
        if not contender_metrics.empty
        else "gradient_boosting"
    )
    decision_support_model = build_decision_support_ensemble(
        model_name=preferred_model_name,
        model_frame=training_frame,
        feature_cols=feature_cols,
        min_train_seasons=args.min_train_seasons,
    )

    model_bundle = {
        "models": results.final_models,
        "decision_support_models": {preferred_model_name: decision_support_model},
        "feature_columns": feature_cols,
        "feature_ranges": results.feature_ranges,
        "training_end_season": training_end_season,
        "preferred_model_name": preferred_model_name,
        "ensemble_size": len(decision_support_model.members),
        "top_actionable_metrics": top_actionable_metrics(feature_importance),
    }
    joblib.dump(model_bundle, config.paths.model_bundle_path)

    model_run = pd.DataFrame(
        [
            {
                "run_timestamp": pd.Timestamp.utcnow().isoformat(),
                "training_end_season": training_end_season,
                "rows_used": len(training_frame),
                "feature_count": len(feature_cols),
                "preferred_model_name": preferred_model_name,
                "ensemble_size": len(decision_support_model.members),
                "top_actionable_metrics": json.dumps(model_bundle["top_actionable_metrics"]),
            }
        ]
    )
    write_sqlite_table(model_run, "model_runs", config.paths.sqlite_path)
    LOGGER.info("Modelle trainiert bis Saison %s mit %s Features.", training_end_season, len(feature_cols))


if __name__ == "__main__":
    main()
