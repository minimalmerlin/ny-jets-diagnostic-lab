from __future__ import annotations

import re

import numpy as np
import pandas as pd

EXCLUDED_METRIC_TOKENS = (
    "player_jersey_number",
    "weight_sum",
)
NON_ACTIONABLE_METRICS = {
    "point_diff",
    "win",
}
NON_ACTIONABLE_METRIC_TOKENS = (
    "games_played",
    "rest_days",
    "strength_of_schedule",
)


def _base_metric_name(feature_name: str) -> str:
    metric = re.sub(r"^pregame_", "", feature_name)
    if re.search(r"_(rolling4|season_avg)_diff$", metric):
        return re.sub(r"_(rolling4|season_avg)_diff$", "", metric)
    return re.sub(r"_diff$", "", metric)


def _metric_direction(metric_name: str) -> int:
    negative_keywords = (
        "allowed",
        "turnover",
        "sack",
        "penalty",
        "fluctuation",
        "changed",
        "instability",
    )
    return -1 if any(token in metric_name for token in negative_keywords) else 1


def _is_actionable_metric(metric_name: str) -> bool:
    if metric_name in NON_ACTIONABLE_METRICS:
        return False
    if any(token in metric_name for token in [*EXCLUDED_METRIC_TOKENS, *NON_ACTIONABLE_METRIC_TOKENS]):
        return False
    return True


def build_team_season_summary(team_week_features: pd.DataFrame) -> pd.DataFrame:
    exclusions = {
        "season",
        "week",
        "game_id",
        "game_date",
        "game_type",
        "team",
        "opponent",
        "stadium",
        "home_team",
        "away_team",
    }
    numeric_cols = [
        column
        for column in team_week_features.select_dtypes(include="number").columns
        if not column.startswith("pregame_")
        and column not in exclusions
        and column not in {"is_home", "home_score", "away_score", "points_for", "points_against"}
    ]
    summary = (
        team_week_features.groupby(["season", "team"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("win", "sum"),
            avg_point_diff=("point_diff", "mean"),
            **{column: (column, "mean") for column in numeric_cols},
        )
        .sort_values(["season", "wins", "avg_point_diff"], ascending=[True, False, False])
    )
    summary["losses"] = summary["games"] - summary["wins"]
    summary["win_pct"] = summary["wins"] / summary["games"].replace(0, np.nan)
    return summary


def build_jets_season_summary(team_season_summary: pd.DataFrame, focus_team: str = "NYJ") -> pd.DataFrame:
    seasons = sorted(team_season_summary["season"].unique())
    rows = []
    for season in seasons:
        season_frame = team_season_summary[team_season_summary["season"] == season].copy()
        jets_row = season_frame[season_frame["team"] == focus_team]
        if jets_row.empty:
            continue
        jets = jets_row.iloc[0]
        season_frame["win_rank"] = season_frame["wins"].rank(method="dense", ascending=False)
        season_frame["point_diff_rank"] = season_frame["avg_point_diff"].rank(method="dense", ascending=False)
        rows.append(
            {
                "season": int(season),
                "team": focus_team,
                "wins": float(jets["wins"]),
                "losses": float(jets["losses"]),
                "win_pct": float(jets["win_pct"]),
                "avg_point_diff": float(jets["avg_point_diff"]),
                "league_avg_win_pct": float(season_frame["win_pct"].mean()),
                "league_avg_point_diff": float(season_frame["avg_point_diff"].mean()),
                "win_rank": int(season_frame.loc[season_frame["team"] == focus_team, "win_rank"].iloc[0]),
                "point_diff_rank": int(
                    season_frame.loc[season_frame["team"] == focus_team, "point_diff_rank"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_root_causes(
    team_season_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    focus_team: str = "NYJ",
) -> pd.DataFrame:
    if team_season_summary.empty or feature_importance.empty:
        return pd.DataFrame(columns=["metric", "problem_score"])
    latest_season = int(team_season_summary["season"].max())
    season_frame = team_season_summary[team_season_summary["season"] == latest_season].copy()
    jets = season_frame[season_frame["team"] == focus_team]
    if jets.empty:
        return pd.DataFrame(columns=["metric", "problem_score"])
    jets_row = jets.iloc[0]

    importance = feature_importance.copy()
    importance["base_metric"] = importance["feature"].map(_base_metric_name)
    importance_summary = (
        importance.groupby("base_metric", as_index=False)
        .agg(
            importance=("importance", "max"),
            direction=("direction", lambda values: pd.Series(values).dropna().iloc[0] if pd.Series(values).dropna().any() else np.nan),
        )
    )
    importance_summary = importance_summary[
        importance_summary["base_metric"].map(_is_actionable_metric)
    ].copy()

    rows = []
    numeric_cols = season_frame.select_dtypes(include="number").columns.tolist()
    for metric in importance_summary["base_metric"]:
        if metric not in numeric_cols:
            continue
        jets_value = float(jets_row[metric])
        league_avg = float(season_frame[metric].mean())
        league_std = float(season_frame[metric].std(ddof=0))
        if not np.isfinite(jets_value) or not np.isfinite(league_avg):
            continue
        direction_row = importance_summary[importance_summary["base_metric"] == metric].iloc[0]
        beneficial_direction = (
            int(np.sign(direction_row["direction"]))
            if pd.notna(direction_row["direction"]) and direction_row["direction"] != 0
            else _metric_direction(metric)
        )
        z_gap = 0.0 if league_std == 0 or np.isnan(league_std) else (jets_value - league_avg) / league_std
        performance_gap = beneficial_direction * z_gap
        problem_score = max(0.0, -performance_gap) * float(direction_row["importance"])
        metric_rank = int(season_frame[metric].rank(method="dense", ascending=False).loc[jets_row.name])
        rows.append(
            {
                "season": latest_season,
                "team": focus_team,
                "metric": metric,
                "jets_value": jets_value,
                "league_avg": league_avg,
                "importance": float(direction_row["importance"]),
                "direction": beneficial_direction,
                "metric_rank": metric_rank,
                "problem_score": problem_score,
            }
        )
    root_causes = pd.DataFrame(rows).sort_values("problem_score", ascending=False)
    return root_causes[root_causes["problem_score"] > 0].reset_index(drop=True)


def build_tracking_lens(team_season_summary: pd.DataFrame, focus_team: str = "NYJ") -> pd.DataFrame:
    tracking_cols = [
        column
        for column in team_season_summary.columns
        if column.startswith("ngs_") and _is_actionable_metric(column)
    ]
    if not tracking_cols:
        return pd.DataFrame(columns=["season", "team"])
    columns = ["season", "team", *tracking_cols]
    lens = team_season_summary[columns].copy()
    jets = lens[lens["team"] == focus_team].copy()
    league = lens.groupby("season", as_index=False)[tracking_cols].mean(numeric_only=True)
    league = league.rename(columns={column: f"league_avg_{column}" for column in tracking_cols})
    return jets.merge(league, on="season", how="left")
