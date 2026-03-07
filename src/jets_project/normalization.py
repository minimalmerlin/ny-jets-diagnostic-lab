from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TEAM_ALIASES = {
    "JAX": "JAC",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "CARDINALS": "ARI",
    "ARIZONA CARDINALS": "ARI",
    "FALCONS": "ATL",
    "ATLANTA FALCONS": "ATL",
    "RAVENS": "BAL",
    "BALTIMORE RAVENS": "BAL",
    "BILLS": "BUF",
    "BUFFALO BILLS": "BUF",
    "PANTHERS": "CAR",
    "CAROLINA PANTHERS": "CAR",
    "BEARS": "CHI",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "BENGALS": "CIN",
    "BROWNS": "CLE",
    "CLEVELAND BROWNS": "CLE",
    "COWBOYS": "DAL",
    "DALLAS COWBOYS": "DAL",
    "BRONCOS": "DEN",
    "DENVER BRONCOS": "DEN",
    "LIONS": "DET",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "PACKERS": "GB",
    "TEXANS": "HOU",
    "HOUSTON TEXANS": "HOU",
    "COLTS": "IND",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAC",
    "JAGUARS": "JAC",
    "CHIEFS": "KC",
    "KANSAS CITY CHIEFS": "KC",
    "RAIDERS": "LV",
    "LAS VEGAS RAIDERS": "LV",
    "CHARGERS": "LAC",
    "LOS ANGELES CHARGERS": "LAC",
    "RAMS": "LAR",
    "LOS ANGELES RAMS": "LAR",
    "DOLPHINS": "MIA",
    "MIAMI DOLPHINS": "MIA",
    "VIKINGS": "MIN",
    "MINNESOTA VIKINGS": "MIN",
    "PATRIOTS": "NE",
    "NEW ENGLAND PATRIOTS": "NE",
    "SAINTS": "NO",
    "NEW ORLEANS SAINTS": "NO",
    "GIANTS": "NYG",
    "NEW YORK GIANTS": "NYG",
    "JETS": "NYJ",
    "NEW YORK JETS": "NYJ",
    "EAGLES": "PHI",
    "PHILADELPHIA EAGLES": "PHI",
    "STEELERS": "PIT",
    "PITTSBURGH STEELERS": "PIT",
    "49ERS": "SF",
    "SAN FRANCISCO 49ERS": "SF",
    "SEAHAWKS": "SEA",
    "SEATTLE SEAHAWKS": "SEA",
    "BUCCANEERS": "TB",
    "BUCS": "TB",
    "TAMPA BAY BUCCANEERS": "TB",
    "TITANS": "TEN",
    "OILERS": "TEN",
    "TENNESSEE TITANS": "TEN",
    "COMMANDERS": "WAS",
    "WASHINGTON": "WAS",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON REDSKINS": "WAS",
    "REDSKINS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
}


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [
        str(column)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in frame.columns
    ]
    return renamed


def normalize_team_abbr(value: object) -> object:
    if value is None or pd.isna(value):
        return value
    text = str(value).upper().strip()
    if "/" in text:
        text = text.split("/")[0].strip()
    return TEAM_ALIASES.get(text, text)


def normalize_team_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_team_abbr)
    return normalized


def first_present(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None
