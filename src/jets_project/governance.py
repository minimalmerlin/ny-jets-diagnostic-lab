from __future__ import annotations

import numpy as np
import pandas as pd


def build_benchmark_comparison(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_metrics.empty or "naive_home_rate" not in evaluation_metrics["model_name"].unique():
        return pd.DataFrame()
    naive = evaluation_metrics[evaluation_metrics["model_name"] == "naive_home_rate"][
        ["test_season", "log_loss", "brier_score"]
    ].rename(columns={"log_loss": "naive_log_loss", "brier_score": "naive_brier"})
    comparison = evaluation_metrics[evaluation_metrics["model_name"] != "naive_home_rate"].merge(
        naive, on="test_season", how="left"
    )
    comparison["beat_log_loss"] = comparison["log_loss"] < comparison["naive_log_loss"]
    comparison["beat_brier"] = comparison["brier_score"] < comparison["naive_brier"]
    comparison["log_loss_gain"] = comparison["naive_log_loss"] - comparison["log_loss"]
    comparison["brier_gain"] = comparison["naive_brier"] - comparison["brier_score"]
    return comparison


def build_model_governance(
    evaluation_metrics: pd.DataFrame,
    backtest_diagnostics: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    model_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if evaluation_metrics.empty:
        return pd.DataFrame()
    evaluation_metrics = evaluation_metrics.copy()
    if "ece" not in evaluation_metrics.columns:
        evaluation_metrics["ece"] = np.nan

    summary = (
        evaluation_metrics.groupby("model_name", as_index=False)
        .agg(
            avg_log_loss=("log_loss", "mean"),
            avg_brier=("brier_score", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_ece=("ece", "mean"),
        )
        .sort_values("avg_log_loss")
    )
    if not backtest_diagnostics.empty:
        diagnostic_summary = (
            backtest_diagnostics.groupby("model_name", as_index=False)
            .agg(
                avg_log_loss_gap=("log_loss_gap", "mean"),
                avg_brier_gap=("brier_score_gap", "mean"),
            )
        )
        summary = summary.merge(diagnostic_summary, on="model_name", how="left")
    if not calibration_summary.empty:
        calibration_rollup = (
            calibration_summary.groupby("model_name", as_index=False)
            .agg(
                calibrator_type=("calibrator_type", lambda values: pd.Series(values).mode().iloc[0]),
                avg_calibration_rows=("calibration_rows", "mean"),
                avg_calibration_seasons=("calibration_seasons", "mean"),
            )
        )
        summary = summary.merge(calibration_rollup, on="model_name", how="left")

    benchmark = build_benchmark_comparison(evaluation_metrics)
    if not benchmark.empty:
        benchmark_rollup = (
            benchmark.groupby("model_name", as_index=False)
            .agg(
                beat_log_loss_seasons=("beat_log_loss", "sum"),
                beat_brier_seasons=("beat_brier", "sum"),
                total_test_seasons=("test_season", "nunique"),
                avg_log_loss_gain=("log_loss_gain", "mean"),
                avg_brier_gain=("brier_gain", "mean"),
            )
        )
        summary = summary.merge(benchmark_rollup, on="model_name", how="left")

    preferred_model_name = None
    ensemble_size = np.nan
    if model_runs is not None and not model_runs.empty:
        latest_run = model_runs.sort_values("run_timestamp").iloc[-1]
        preferred_model_name = latest_run.get("preferred_model_name")
        ensemble_size = latest_run.get("ensemble_size", np.nan)
    if preferred_model_name is not None:
        summary["is_preferred_model"] = summary["model_name"].eq(preferred_model_name)
    else:
        summary["is_preferred_model"] = False
    summary["decision_support_ensemble_size"] = np.where(
        summary["is_preferred_model"], ensemble_size, np.nan
    )
    return summary


def build_calibration_bins(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    required = {"model_name", "predicted_home_win_probability", "actual_home_win"}
    if predictions.empty or not required <= set(predictions.columns):
        return pd.DataFrame()

    frames = []
    bins = np.linspace(0, 1, n_bins + 1)
    for model_name, model_frame in predictions.groupby("model_name"):
        frame = model_frame.copy()
        frame["bin"] = pd.cut(
            frame["predicted_home_win_probability"],
            bins=bins,
            include_lowest=True,
            duplicates="drop",
        )
        grouped = (
            frame.groupby("bin", observed=False)
            .agg(
                count=("actual_home_win", "size"),
                mean_predicted=("predicted_home_win_probability", "mean"),
                observed_rate=("actual_home_win", "mean"),
            )
            .reset_index()
        )
        grouped["model_name"] = model_name
        grouped["bin_lower"] = grouped["bin"].map(lambda interval: float(interval.left) if pd.notna(interval) else np.nan)
        grouped["bin_upper"] = grouped["bin"].map(lambda interval: float(interval.right) if pd.notna(interval) else np.nan)
        grouped["calibration_gap"] = grouped["mean_predicted"] - grouped["observed_rate"]
        frames.append(grouped.drop(columns=["bin"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
