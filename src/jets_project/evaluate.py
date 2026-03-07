from __future__ import annotations

import logging

from .analysis_utils import build_jets_season_summary, build_root_causes, build_team_season_summary, build_tracking_lens
from .governance import build_calibration_bins, build_model_governance
from .logging_utils import configure_logging
from .reporting import write_markdown_report
from .settings import get_default_config
from .storage import export_sqlite_tables, list_sqlite_tables, read_sqlite_table, write_parquet, write_sqlite_table

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = get_default_config()
    team_week_features = read_sqlite_table(config.paths.sqlite_path, "team_week_features")
    feature_importance = read_sqlite_table(config.paths.sqlite_path, "feature_importance")

    team_summary = build_team_season_summary(team_week_features)
    jets_summary = build_jets_season_summary(team_summary, focus_team=config.focus_team)
    root_causes = build_root_causes(team_summary, feature_importance, focus_team=config.focus_team)
    tracking_lens = build_tracking_lens(team_summary, focus_team=config.focus_team)

    outputs = {
        "team_season_summary": team_summary,
        "jets_season_summary": jets_summary,
        "root_causes": root_causes,
        "tracking_lens": tracking_lens,
    }
    for name, frame in outputs.items():
        write_sqlite_table(frame, name, config.paths.sqlite_path)
        write_parquet(frame, config.paths.artifacts_reports / f"{name}.parquet")
        LOGGER.info("Report-Artefakt gespeichert: %s", name)

    available_tables = set(list_sqlite_tables(config.paths.sqlite_path))
    evaluation_metrics = (
        read_sqlite_table(config.paths.sqlite_path, "evaluation_metrics")
        if "evaluation_metrics" in available_tables
        else team_summary.iloc[0:0]
    )
    predictions = (
        read_sqlite_table(config.paths.sqlite_path, "predictions")
        if "predictions" in available_tables
        else team_summary.iloc[0:0]
    )
    backtest_diagnostics = (
        read_sqlite_table(config.paths.sqlite_path, "backtest_diagnostics")
        if "backtest_diagnostics" in available_tables
        else team_summary.iloc[0:0]
    )
    calibration_summary = (
        read_sqlite_table(config.paths.sqlite_path, "calibration_summary")
        if "calibration_summary" in available_tables
        else team_summary.iloc[0:0]
    )
    model_runs = (
        read_sqlite_table(config.paths.sqlite_path, "model_runs")
        if "model_runs" in available_tables
        else team_summary.iloc[0:0]
    )
    model_governance = build_model_governance(
        evaluation_metrics=evaluation_metrics,
        backtest_diagnostics=backtest_diagnostics,
        calibration_summary=calibration_summary,
        model_runs=model_runs,
    )
    calibration_bins = build_calibration_bins(predictions)
    for name, frame in {
        "model_governance": model_governance,
        "calibration_bins": calibration_bins,
    }.items():
        write_sqlite_table(frame, name, config.paths.sqlite_path)
        write_parquet(frame, config.paths.artifacts_reports / f"{name}.parquet")
        LOGGER.info("Report-Artefakt gespeichert: %s", name)
    report_path = config.paths.reports_dir / "final_report.md"
    write_markdown_report(
        output_path=report_path,
        jets_season_summary=jets_summary,
        root_causes=root_causes,
        evaluation_metrics=evaluation_metrics,
        backtest_diagnostics=backtest_diagnostics,
        focus_team=config.focus_team,
    )
    LOGGER.info("Markdown-Report gespeichert: %s", report_path.name)

    export_sqlite_tables(
        source_path=config.paths.sqlite_path,
        destination_path=config.paths.app_sqlite_path,
        table_names=[
            "team_season_summary",
            "jets_season_summary",
            "root_causes",
            "tracking_lens",
            "game_features",
            "evaluation_metrics",
            "backtest_diagnostics",
            "calibration_bins",
            "model_governance",
            "model_runs",
            "dataset_availability",
        ],
    )
    LOGGER.info("App-SQLite gespeichert: %s", config.paths.app_sqlite_path.name)


if __name__ == "__main__":
    main()
