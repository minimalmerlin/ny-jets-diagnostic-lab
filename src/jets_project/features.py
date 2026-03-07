from __future__ import annotations

import logging

import pandas as pd

from .feature_engineering import (
    aggregate_nextgen_team_week,
    aggregate_pbp_team_game,
    build_player_week_features,
    build_team_week_features,
    prepare_contract_features,
    prepare_injury_features,
    prepare_team_stats,
)
from .logging_utils import configure_logging
from .settings import get_default_config
from .storage import list_sqlite_tables, read_sqlite_table, write_parquet, write_sqlite_table

LOGGER = logging.getLogger(__name__)


def _optional_table(table_names: list[str], sqlite_path, name: str):
    if name not in table_names:
        return None
    return read_sqlite_table(sqlite_path, name)


def main() -> None:
    configure_logging()
    config = get_default_config()
    table_names = list_sqlite_tables(config.paths.sqlite_path)
    games = read_sqlite_table(config.paths.sqlite_path, "games")
    pbp = read_sqlite_table(config.paths.sqlite_path, "pbp")
    team_stats = _optional_table(table_names, config.paths.sqlite_path, "team_stats")
    player_stats = _optional_table(table_names, config.paths.sqlite_path, "player_stats")
    snap_counts = _optional_table(table_names, config.paths.sqlite_path, "snap_counts")
    rosters = _optional_table(table_names, config.paths.sqlite_path, "rosters")
    injuries = _optional_table(table_names, config.paths.sqlite_path, "injuries")
    contracts = _optional_table(table_names, config.paths.sqlite_path, "contracts")
    staff_tenure = _optional_table(table_names, config.paths.sqlite_path, "staff_tenure")

    pbp_team_game = aggregate_pbp_team_game(pbp)
    team_stats_features = prepare_team_stats(team_stats) if team_stats is not None else prepare_team_stats(games.iloc[0:0])
    player_week_features = build_player_week_features(
        player_stats if player_stats is not None else games.iloc[0:0],
        snap_counts if snap_counts is not None else games.iloc[0:0],
        rosters if rosters is not None else games.iloc[0:0],
    )
    injury_features = prepare_injury_features(injuries) if injuries is not None else prepare_injury_features(games.iloc[0:0])
    contract_features = (
        prepare_contract_features(contracts) if contracts is not None else prepare_contract_features(games.iloc[0:0])
    )
    ngs_passing = _optional_table(table_names, config.paths.sqlite_path, "ngs_passing")
    ngs_rushing = _optional_table(table_names, config.paths.sqlite_path, "ngs_rushing")
    ngs_receiving = _optional_table(table_names, config.paths.sqlite_path, "ngs_receiving")
    ngs_team_week = aggregate_nextgen_team_week(
        {
            "passing": ngs_passing if ngs_passing is not None else pd.DataFrame(),
            "rushing": ngs_rushing if ngs_rushing is not None else pd.DataFrame(),
            "receiving": ngs_receiving if ngs_receiving is not None else pd.DataFrame(),
        }
    )
    team_week_features = build_team_week_features(
        games=games,
        pbp_team_game=pbp_team_game,
        team_stats=team_stats_features,
        player_week_features=player_week_features,
        injury_features=injury_features,
        ngs_team_week=ngs_team_week,
        contract_features=contract_features,
        staff_features=staff_tenure if staff_tenure is not None else games.iloc[0:0],
    )

    outputs = {
        "pbp_team_game": pbp_team_game,
        "player_week_features": player_week_features,
        "injury_features": injury_features,
        "contract_features": contract_features,
        "ngs_team_week": ngs_team_week,
        "team_week_features": team_week_features,
    }
    for name, frame in outputs.items():
        write_sqlite_table(frame, name, config.paths.sqlite_path)
        write_parquet(frame, config.paths.data_processed / f"{name}.parquet")
        LOGGER.info("Feature-Tabelle gespeichert: %s (%s Zeilen)", name, len(frame))


if __name__ == "__main__":
    main()
