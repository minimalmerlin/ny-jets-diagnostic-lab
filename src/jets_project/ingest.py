from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
import numpy as np

from .feature_engineering import prepare_games
from .loaders import NFLDataClient
from .logging_utils import configure_logging
from .reference import load_staff_tenure
from .settings import get_default_config
from .storage import write_parquet, write_sqlite_table

LOGGER = logging.getLogger(__name__)


def _seasonal_availability(name: str, frame: pd.DataFrame, seasons: list[int]) -> dict[str, object]:
    if "season" not in frame.columns:
        return {
            "dataset_name": name,
            "requested_start_season": min(seasons),
            "requested_end_season": max(seasons),
            "loaded_start_season": pd.NA,
            "loaded_end_season": pd.NA,
            "loaded_seasons": json.dumps([]),
            "missing_seasons": json.dumps([]),
            "row_count": int(len(frame)),
            "status": "nonseasonal",
        }

    loaded = sorted(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).unique().tolist())
    missing = sorted(set(seasons) - set(loaded))
    if not loaded:
        status = "empty"
    elif missing:
        status = "partial"
    else:
        status = "complete"
    return {
        "dataset_name": name,
        "requested_start_season": min(seasons),
        "requested_end_season": max(seasons),
        "loaded_start_season": loaded[0] if loaded else pd.NA,
        "loaded_end_season": loaded[-1] if loaded else pd.NA,
        "loaded_seasons": json.dumps(loaded),
        "missing_seasons": json.dumps(missing),
        "row_count": int(len(frame)),
        "status": status,
    }


def _serialize_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    if "cols" not in frame.columns:
        return frame
    serialized = frame.copy()
    serialized["cols"] = serialized["cols"].map(
        lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
    )
    return serialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL-Daten laden und in SQLite/Parquet speichern.")
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = get_default_config()
    config.start_season = args.start_season
    config.end_season = args.end_season
    config.ensure_directories()
    seasons = config.seasons()

    client = NFLDataClient()
    datasets = {
        "schedules": client.load_schedules(seasons),
        "pbp": client.load_pbp(seasons),
        "team_stats": client.load_team_stats(seasons),
        "player_stats": client.load_player_stats(seasons),
        "snap_counts": client.load_snap_counts(seasons),
        "rosters": client.load_rosters(seasons),
        "injuries": client.load_injuries(seasons),
        "contracts": client.load_contracts(),
        "ngs_passing": client.load_nextgen("passing", seasons),
        "ngs_rushing": client.load_nextgen("rushing", seasons),
        "ngs_receiving": client.load_nextgen("receiving", seasons),
    }
    datasets["contracts"] = _serialize_contracts(datasets["contracts"])

    for name, frame in datasets.items():
        write_parquet(frame, config.paths.data_raw / f"{name}_{args.start_season}_{args.end_season}.parquet")
        write_sqlite_table(frame, name, config.paths.sqlite_path)
        LOGGER.info("Gespeichert: %s (%s Zeilen)", name, len(frame))

    games = prepare_games(datasets["schedules"])
    staff = load_staff_tenure(config.paths.staff_tenure_path)
    availability = pd.DataFrame(
        [_seasonal_availability(name=name, frame=frame, seasons=seasons) for name, frame in datasets.items()]
    )
    write_sqlite_table(games, "games", config.paths.sqlite_path)
    write_sqlite_table(staff, "staff_tenure", config.paths.sqlite_path)
    write_sqlite_table(availability, "dataset_availability", config.paths.sqlite_path)
    write_parquet(availability, config.paths.artifacts_reports / "dataset_availability.parquet")
    LOGGER.info("Games und Staff-Tenure gespeichert.")


if __name__ == "__main__":
    main()
