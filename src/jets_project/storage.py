from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_sqlite_table(frame: pd.DataFrame, table_name: str, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        frame.to_sql(table_name, connection, if_exists="replace", index=False)


def read_sqlite_table(sqlite_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(sqlite_path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


def list_sqlite_tables(sqlite_path: Path) -> list[str]:
    with sqlite3.connect(sqlite_path) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            connection,
        )
    return tables["name"].tolist()


def export_sqlite_tables(source_path: Path, destination_path: Path, table_names: list[str]) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_path.unlink()
    with sqlite3.connect(source_path) as source_connection, sqlite3.connect(destination_path) as destination_connection:
        for table_name in table_names:
            frame = pd.read_sql_query(f"SELECT * FROM {table_name}", source_connection)
            frame.to_sql(table_name, destination_connection, if_exists="replace", index=False)
