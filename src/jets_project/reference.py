from __future__ import annotations

import pandas as pd

from .normalization import clean_columns, normalize_team_columns

STAFF_COLUMNS = [
    "season",
    "team",
    "head_coach",
    "offensive_coordinator",
    "defensive_coordinator",
    "general_manager",
]


def _tenure_years(series: pd.Series) -> pd.Series:
    result = []
    prior = None
    tenure = 0
    for value in series:
        if pd.isna(value):
            result.append(pd.NA)
            prior = None
            tenure = 0
            continue
        if value == prior:
            tenure += 1
        else:
            tenure = 1
            prior = value
        result.append(tenure)
    return pd.Series(result, index=series.index, dtype="Int64")


def load_staff_tenure(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STAFF_COLUMNS)

    staff = clean_columns(pd.read_csv(path))
    for column in STAFF_COLUMNS:
        if column not in staff.columns:
            staff[column] = pd.NA

    staff = normalize_team_columns(staff[STAFF_COLUMNS], ["team"])
    staff = staff.sort_values(["team", "season"]).reset_index(drop=True)

    for label, source in (
        ("hc", "head_coach"),
        ("oc", "offensive_coordinator"),
        ("dc", "defensive_coordinator"),
        ("gm", "general_manager"),
    ):
        shifted = staff.groupby("team", sort=False)[source].shift(1)
        staff[f"{label}_changed"] = (
            staff[source].notna() & shifted.notna() & (staff[source] != shifted)
        ).astype(int)
        staff[f"{label}_tenure_years"] = (
            staff.groupby("team", sort=False)[source].transform(_tenure_years).fillna(0)
        )

    return staff
