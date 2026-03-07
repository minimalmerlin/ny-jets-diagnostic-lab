from __future__ import annotations

from collections.abc import Iterable
import json
import re

import numpy as np
import pandas as pd

from .normalization import clean_columns, first_present, normalize_team_columns


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _to_numeric(series: pd.Series | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _contains_tokens(series: pd.Series, tokens: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(token) for token in tokens)
    return series.fillna("").astype(str).str.lower().str.contains(pattern, regex=True, na=False)


def prepare_games(schedules: pd.DataFrame) -> pd.DataFrame:
    schedules = clean_columns(schedules)
    home_col = first_present(schedules, ["home_team", "home"])
    away_col = first_present(schedules, ["away_team", "away"])
    season_col = first_present(schedules, ["season"])
    week_col = first_present(schedules, ["week"])
    if home_col is None or away_col is None or season_col is None or week_col is None:
        raise RuntimeError("Schedules enthalten nicht die erwarteten Basisfelder fuer home/away, season und week.")

    game_id_col = first_present(schedules, ["game_id", "gsis_id", "old_game_id", "id"])
    game_type_col = first_present(schedules, ["game_type", "season_type"])
    date_col = first_present(schedules, ["gameday", "game_date", "game_datetime", "date"])
    stadium_col = first_present(schedules, ["stadium", "venue"])
    home_score_col = first_present(schedules, ["home_score", "score_home"])
    away_score_col = first_present(schedules, ["away_score", "score_away"])

    games = pd.DataFrame(
        {
            "game_id": schedules[game_id_col]
            if game_id_col is not None
            else schedules[season_col].astype(str)
            + "_"
            + schedules[week_col].astype(str)
            + "_"
            + schedules[home_col].astype(str)
            + "_"
            + schedules[away_col].astype(str),
            "season": pd.to_numeric(schedules[season_col], errors="coerce"),
            "week": pd.to_numeric(schedules[week_col], errors="coerce"),
            "game_date": pd.to_datetime(schedules[date_col], errors="coerce")
            if date_col is not None
            else pd.NaT,
            "game_type": schedules[game_type_col].fillna("REG").astype(str).str.upper()
            if game_type_col is not None
            else "REG",
            "home_team": schedules[home_col],
            "away_team": schedules[away_col],
            "home_score": _to_numeric(schedules[home_score_col], np.nan)
            if home_score_col is not None
            else np.nan,
            "away_score": _to_numeric(schedules[away_score_col], np.nan)
            if away_score_col is not None
            else np.nan,
            "stadium": schedules[stadium_col].astype(str) if stadium_col is not None else pd.NA,
        }
    )
    games = normalize_team_columns(games, ["home_team", "away_team"])
    games = games.sort_values(["season", "week", "game_date", "game_id"]).drop_duplicates("game_id")
    games["score_diff"] = games["home_score"] - games["away_score"]
    games["home_win"] = (games["home_score"] > games["away_score"]).astype("Int64")
    return games.reset_index(drop=True)


def build_team_game_frame(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    for column in ("home_score", "away_score", "score_diff"):
        if column in games.columns:
            games[column] = pd.to_numeric(games[column], errors="coerce")
    home = games.assign(
        team=games["home_team"],
        opponent=games["away_team"],
        is_home=1,
        points_for=games["home_score"],
        points_against=games["away_score"],
    )
    away = games.assign(
        team=games["away_team"],
        opponent=games["home_team"],
        is_home=0,
        points_for=games["away_score"],
        points_against=games["home_score"],
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype("Int64")
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    team_games = team_games.sort_values(["team", "season", "game_date", "week", "game_id"]).reset_index(drop=True)
    team_games["rest_days"] = (
        team_games.groupby(["team", "season"], sort=False)["game_date"].diff().dt.days
    )
    return team_games


def aggregate_pbp_team_game(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = clean_columns(pbp)
    season_col = first_present(pbp, ["season"])
    week_col = first_present(pbp, ["week"])
    game_id_col = first_present(pbp, ["game_id", "old_game_id", "gsis_id"])
    posteam_col = first_present(pbp, ["posteam"])
    defteam_col = first_present(pbp, ["defteam"])
    if None in (season_col, week_col, game_id_col, posteam_col, defteam_col):
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    play_type_col = first_present(pbp, ["play_type"])
    epa_col = first_present(pbp, ["epa"])
    yards_col = first_present(pbp, ["yards_gained"])
    down_col = first_present(pbp, ["down"])
    sack_col = first_present(pbp, ["sack"])
    interception_col = first_present(pbp, ["interception"])
    fumble_lost_col = first_present(pbp, ["fumble_lost"])
    penalty_col = first_present(pbp, ["penalty"])
    first_down_col = first_present(pbp, ["first_down", "first_down_rush", "first_down_pass"])
    touchdown_col = first_present(pbp, ["touchdown"])
    yardline_col = first_present(pbp, ["yardline_100"])

    work = pd.DataFrame(
        {
            "season": pd.to_numeric(pbp[season_col], errors="coerce"),
            "week": pd.to_numeric(pbp[week_col], errors="coerce"),
            "game_id": pbp[game_id_col].astype(str),
            "team": pbp[posteam_col],
            "opponent": pbp[defteam_col],
            "play_type": pbp[play_type_col].astype(str).str.lower() if play_type_col else "unknown",
            "epa": _to_numeric(pbp[epa_col], np.nan) if epa_col else np.nan,
            "yards_gained": _to_numeric(pbp[yards_col]) if yards_col else 0.0,
            "down": _to_numeric(pbp[down_col], np.nan) if down_col else np.nan,
            "sack": _to_numeric(pbp[sack_col]) if sack_col else 0.0,
            "interception": _to_numeric(pbp[interception_col]) if interception_col else 0.0,
            "fumble_lost": _to_numeric(pbp[fumble_lost_col]) if fumble_lost_col else 0.0,
            "penalty": _to_numeric(pbp[penalty_col]) if penalty_col else 0.0,
            "first_down": _to_numeric(pbp[first_down_col]) if first_down_col else 0.0,
            "touchdown": _to_numeric(pbp[touchdown_col]) if touchdown_col else 0.0,
            "yardline_100": _to_numeric(pbp[yardline_col], np.nan) if yardline_col else np.nan,
        }
    )
    work = normalize_team_columns(work, ["team", "opponent"])
    work = work[work["team"].notna() & work["opponent"].notna() & work["game_id"].notna()].copy()
    work["valid_play"] = work["epa"].notna() | work["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])
    work = work[work["valid_play"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    work["_play"] = 1
    work["success"] = (work["epa"] > 0).astype(int)
    work["turnover"] = ((work["interception"] > 0) | (work["fumble_lost"] > 0)).astype(int)
    work["explosive"] = (
        ((work["play_type"] == "pass") & (work["yards_gained"] >= 20))
        | ((work["play_type"] == "run") & (work["yards_gained"] >= 10))
    ).astype(int)
    work["red_zone"] = (work["yardline_100"] <= 20).fillna(False).astype(int)
    work["red_zone_touchdown"] = ((work["red_zone"] == 1) & (work["touchdown"] > 0)).astype(int)
    work["third_down"] = (work["down"] == 3).astype(int)
    work["fourth_down"] = (work["down"] == 4).astype(int)
    work["third_down_conversion"] = (
        (work["third_down"] == 1) & ((work["first_down"] > 0) | (work["touchdown"] > 0))
    ).astype(int)
    work["fourth_down_conversion"] = (
        (work["fourth_down"] == 1) & ((work["first_down"] > 0) | (work["touchdown"] > 0))
    ).astype(int)
    work["early_down"] = work["down"].isin([1, 2]).astype(int)
    work["early_down_pass"] = (
        (work["early_down"] == 1) & (work["play_type"] == "pass")
    ).astype(int)

    keys = ["season", "week", "game_id", "team"]
    offense = work.groupby(keys, dropna=False).agg(
        offensive_plays=("_play", "sum"),
        offense_epa_per_play=("epa", "mean"),
        offense_success_rate=("success", "mean"),
        offense_turnover_rate=("turnover", "mean"),
        offense_sack_rate=("sack", "mean"),
        offense_explosive_play_rate=("explosive", "mean"),
        offense_penalty_rate=("penalty", "mean"),
        offense_yards_per_play=("yards_gained", "mean"),
        red_zone_plays=("red_zone", "sum"),
        red_zone_touchdowns=("red_zone_touchdown", "sum"),
        third_down_plays=("third_down", "sum"),
        third_down_conversions=("third_down_conversion", "sum"),
        fourth_down_plays=("fourth_down", "sum"),
        fourth_down_conversions=("fourth_down_conversion", "sum"),
        early_down_plays=("early_down", "sum"),
        early_down_passes=("early_down_pass", "sum"),
    )
    offense["offense_red_zone_td_rate"] = _safe_divide(
        offense["red_zone_touchdowns"], offense["red_zone_plays"]
    ).fillna(0.0)
    offense["offense_third_down_conversion_rate"] = _safe_divide(
        offense["third_down_conversions"], offense["third_down_plays"]
    ).fillna(0.0)
    offense["offense_fourth_down_conversion_rate"] = _safe_divide(
        offense["fourth_down_conversions"], offense["fourth_down_plays"]
    ).fillna(0.0)
    offense["offense_early_down_pass_rate"] = _safe_divide(
        offense["early_down_passes"], offense["early_down_plays"]
    ).fillna(0.0)
    offense = offense.drop(
        columns=[
            "red_zone_plays",
            "red_zone_touchdowns",
            "third_down_plays",
            "third_down_conversions",
            "fourth_down_plays",
            "fourth_down_conversions",
            "early_down_plays",
            "early_down_passes",
        ]
    ).reset_index()

    defense_work = work.rename(columns={"team": "offense_team", "opponent": "team"})
    defense = defense_work.groupby(keys, dropna=False).agg(
        defense_plays_faced=("_play", "sum"),
        defense_epa_per_play_allowed=("epa", "mean"),
        defense_success_rate_allowed=("success", "mean"),
        defense_takeaway_rate=("turnover", "mean"),
        defense_sack_rate=("sack", "mean"),
        defense_explosive_play_rate_allowed=("explosive", "mean"),
        defense_yards_per_play_allowed=("yards_gained", "mean"),
    )
    defense = defense.reset_index()

    return offense.merge(defense, on=keys, how="outer")


def prepare_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    team_stats = clean_columns(team_stats)
    season_col = first_present(team_stats, ["season"])
    week_col = first_present(team_stats, ["week"])
    team_col = first_present(team_stats, ["team", "recent_team", "team_abbr", "club"])
    if None in (season_col, week_col, team_col):
        return pd.DataFrame(columns=["season", "week", "team"])

    selected = pd.DataFrame(
        {
            "season": pd.to_numeric(team_stats[season_col], errors="coerce"),
            "week": pd.to_numeric(team_stats[week_col], errors="coerce"),
            "team": team_stats[team_col],
        }
    )
    selected = normalize_team_columns(selected, ["team"])
    mappings = {
        "teamstat_total_yards": ["total_yards", "yards"],
        "teamstat_passing_yards": ["passing_yards", "pass_yards"],
        "teamstat_rushing_yards": ["rushing_yards", "rush_yards"],
        "teamstat_first_downs": ["first_downs", "first_down"],
        "teamstat_turnovers": ["turnovers", "giveaways"],
        "teamstat_penalty_yards": ["penalty_yards"],
        "teamstat_time_of_possession_seconds": ["time_of_possession", "possession_seconds"],
        "teamstat_red_zone_pct": ["red_zone_pct", "red_zone_touchdown_pct"],
        "teamstat_third_down_pct": ["third_down_pct", "third_down_conversion_pct"],
        "teamstat_fourth_down_pct": ["fourth_down_pct", "fourth_down_conversion_pct"],
    }
    for output, candidates in mappings.items():
        source = first_present(team_stats, candidates)
        if source is not None:
            selected[output] = _to_numeric(team_stats[source], np.nan)

    numeric_cols = [col for col in selected.columns if col not in {"season", "week", "team"}]
    if not numeric_cols:
        return selected.drop_duplicates(["season", "week", "team"])
    return (
        selected.groupby(["season", "week", "team"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )


def prepare_injury_features(injuries: pd.DataFrame) -> pd.DataFrame:
    injuries = clean_columns(injuries)
    season_col = first_present(injuries, ["season"])
    week_col = first_present(injuries, ["week"])
    team_col = first_present(injuries, ["team", "recent_team", "team_abbr", "club"])
    player_col = first_present(injuries, ["gsis_id", "player_id", "full_name"])
    if None in (season_col, week_col, team_col, player_col):
        return pd.DataFrame(columns=["season", "week", "team"])

    position_col = first_present(injuries, ["position"])
    report_status_col = first_present(injuries, ["report_status"])
    practice_status_col = first_present(injuries, ["practice_status"])
    frame = pd.DataFrame(
        {
            "season": pd.to_numeric(injuries[season_col], errors="coerce"),
            "week": pd.to_numeric(injuries[week_col], errors="coerce"),
            "team": injuries[team_col],
            "player_key": injuries[player_col].astype(str),
            "position": injuries[position_col].astype(str).str.upper() if position_col is not None else pd.NA,
            "report_status": injuries[report_status_col].astype(str) if report_status_col is not None else "",
            "practice_status": injuries[practice_status_col].astype(str) if practice_status_col is not None else "",
        }
    )
    frame = normalize_team_columns(frame, ["team"])
    frame = frame.drop_duplicates(["season", "week", "team", "player_key"])
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    oline_positions = {"LT", "LG", "C", "RG", "RT", "OL", "G", "T"}
    skill_positions = {"QB", "RB", "FB", "WR", "TE"}
    defense_positions = {"DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB", "CB", "S", "SS", "FS", "DB"}

    frame["injury_report_count"] = 1
    frame["injury_out_count"] = _contains_tokens(frame["report_status"], ("out", "injured reserve", "reserve/injured")).astype(int)
    frame["injury_doubtful_count"] = _contains_tokens(frame["report_status"], ("doubtful",)).astype(int)
    frame["injury_questionable_count"] = _contains_tokens(frame["report_status"], ("questionable",)).astype(int)
    frame["injury_dnp_count"] = _contains_tokens(
        frame["practice_status"], ("did not participate", "did not practice")
    ).astype(int)
    frame["injury_limited_count"] = _contains_tokens(frame["practice_status"], ("limited",)).astype(int)
    frame["injury_oline_out_count"] = (
        frame["injury_out_count"] * frame["position"].isin(oline_positions).astype(int)
    )
    frame["injury_skill_out_count"] = (
        frame["injury_out_count"] * frame["position"].isin(skill_positions).astype(int)
    )
    frame["injury_defense_out_count"] = (
        frame["injury_out_count"] * frame["position"].isin(defense_positions).astype(int)
    )

    grouped = (
        frame.groupby(["season", "week", "team"], as_index=False)
        .agg(
            injury_report_count=("injury_report_count", "sum"),
            injury_out_count=("injury_out_count", "sum"),
            injury_doubtful_count=("injury_doubtful_count", "sum"),
            injury_questionable_count=("injury_questionable_count", "sum"),
            injury_dnp_count=("injury_dnp_count", "sum"),
            injury_limited_count=("injury_limited_count", "sum"),
            injury_oline_out_count=("injury_oline_out_count", "sum"),
            injury_skill_out_count=("injury_skill_out_count", "sum"),
            injury_defense_out_count=("injury_defense_out_count", "sum"),
        )
        .sort_values(["season", "week", "team"])
    )
    grouped["injury_out_rate"] = _safe_divide(
        grouped["injury_out_count"], grouped["injury_report_count"]
    ).fillna(0.0)
    grouped["injury_dnp_rate"] = _safe_divide(
        grouped["injury_dnp_count"], grouped["injury_report_count"]
    ).fillna(0.0)
    grouped["injury_impact_score"] = (
        grouped["injury_out_count"]
        + 0.75 * grouped["injury_doubtful_count"]
        + 0.4 * grouped["injury_questionable_count"]
        + 0.5 * grouped["injury_dnp_count"]
        + 0.2 * grouped["injury_limited_count"]
    )
    return grouped.reset_index(drop=True)


def prepare_contract_features(contracts: pd.DataFrame) -> pd.DataFrame:
    contracts = clean_columns(contracts)
    nested_col = first_present(contracts, ["cols"])
    position_col = first_present(contracts, ["position"])
    player_col = first_present(contracts, ["gsis_id", "player_id", "player"])
    if None in (nested_col, position_col, player_col):
        return pd.DataFrame(columns=["season", "team"])

    working = pd.DataFrame(
        {
            "player_key": contracts[player_col].astype(str),
            "position": contracts[position_col].astype(str).str.upper(),
            "cols": contracts[nested_col],
        }
    )

    def _contract_rows(value) -> list[dict]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        return []

    working["cols"] = working["cols"].map(_contract_rows)
    working = working[working["cols"].map(bool)].copy()
    if working.empty:
        return pd.DataFrame(columns=["season", "team"])
    working = working.explode("cols").reset_index(drop=True)
    details = clean_columns(pd.json_normalize(working["cols"]))
    detail_frame = pd.concat([working.drop(columns=["cols"]).reset_index(drop=True), details], axis=1)

    year_col = first_present(detail_frame, ["year"])
    team_col = first_present(detail_frame, ["team"])
    cap_col = first_present(detail_frame, ["cap_number"])
    guaranteed_col = first_present(detail_frame, ["guaranteed_salary", "guaranteed"])
    cash_col = first_present(detail_frame, ["cash_paid"])
    cap_pct_col = first_present(detail_frame, ["cap_percent"])
    if None in (year_col, team_col, cap_col):
        return pd.DataFrame(columns=["season", "team"])

    detail_frame = pd.DataFrame(
        {
            "season": pd.to_numeric(detail_frame[year_col], errors="coerce"),
            "team": detail_frame[team_col],
            "player_key": detail_frame["player_key"].astype(str),
            "position": detail_frame["position"].astype(str).str.upper(),
            "cap_number": _to_numeric(detail_frame[cap_col], np.nan),
            "guaranteed_salary": _to_numeric(detail_frame[guaranteed_col], np.nan)
            if guaranteed_col is not None
            else np.nan,
            "cash_paid": _to_numeric(detail_frame[cash_col], np.nan) if cash_col is not None else np.nan,
            "cap_percent": _to_numeric(detail_frame[cap_pct_col], np.nan) if cap_pct_col is not None else np.nan,
        }
    )
    detail_frame = normalize_team_columns(detail_frame, ["team"])
    detail_frame = detail_frame.dropna(subset=["season", "team", "cap_number"])
    if detail_frame.empty:
        return pd.DataFrame(columns=["season", "team"])

    offense_positions = {"QB", "RB", "FB", "WR", "TE", "OL", "LT", "LG", "C", "RG", "RT", "G", "T"}
    defense_positions = {"DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB", "CB", "S", "SS", "FS", "DB"}

    rows = []
    for (season, team), group in detail_frame.groupby(["season", "team"], dropna=False):
        total_cap = float(group["cap_number"].sum())
        qb_cap = float(group.loc[group["position"] == "QB", "cap_number"].sum())
        offense_cap = float(group.loc[group["position"].isin(offense_positions), "cap_number"].sum())
        defense_cap = float(group.loc[group["position"].isin(defense_positions), "cap_number"].sum())
        top3_cap = float(group["cap_number"].nlargest(3).sum())
        rows.append(
            {
                "season": season,
                "team": team,
                "contract_player_count": float(group["player_key"].nunique()),
                "contract_total_cap_number": total_cap,
                "contract_total_guaranteed_salary": float(group["guaranteed_salary"].sum()),
                "contract_total_cash_paid": float(group["cash_paid"].sum()),
                "contract_avg_cap_percent": float(group["cap_percent"].mean()),
                "contract_max_cap_number": float(group["cap_number"].max()),
                "contract_top3_cap_share": 0.0 if total_cap == 0 else top3_cap / total_cap,
                "contract_qb_cap_share": 0.0 if total_cap == 0 else qb_cap / total_cap,
                "contract_offense_cap_share": 0.0 if total_cap == 0 else offense_cap / total_cap,
                "contract_defense_cap_share": 0.0 if total_cap == 0 else defense_cap / total_cap,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "team"]).reset_index(drop=True)


def _player_sets(frame: pd.DataFrame, top_k: int, positions: set[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season", "team", "week", "player_set"])
    subset = frame
    if positions is not None and "position" in frame.columns:
        subset = frame[frame["position"].isin(positions)].copy()
    rows = []
    for (season, team, week), group in subset.groupby(["season", "team", "week"], dropna=False):
        players = (
            group.sort_values("snap_value", ascending=False)["player_id"]
            .dropna()
            .astype(str)
            .head(top_k)
            .tolist()
        )
        rows.append(
            {
                "season": season,
                "team": team,
                "week": week,
                "player_set": tuple(players),
            }
        )
    return pd.DataFrame(rows)


def _overlap_feature(set_frame: pd.DataFrame, feature_name: str, invert: bool = False) -> pd.DataFrame:
    if set_frame.empty:
        return pd.DataFrame(columns=["season", "team", "week", feature_name])
    set_frame = set_frame.sort_values(["team", "season", "week"]).reset_index(drop=True)
    overlaps = []
    previous_by_team: dict[tuple[str, int], set[str]] = {}
    for row in set_frame.itertuples(index=False):
        key = (row.team, row.season)
        current = set(row.player_set)
        previous = previous_by_team.get(key)
        if not previous or not current:
            value = np.nan
        else:
            value = len(previous & current) / max(1, min(len(previous), len(current)))
        overlaps.append(1 - value if invert and pd.notna(value) else value)
        previous_by_team[key] = current
    result = set_frame[["season", "team", "week"]].copy()
    result[feature_name] = overlaps
    return result


def build_player_week_features(
    player_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    rosters: pd.DataFrame,
) -> pd.DataFrame:
    player_stats = clean_columns(player_stats)
    snap_counts = clean_columns(snap_counts)
    rosters = clean_columns(rosters)

    outputs: list[pd.DataFrame] = []
    season_stat_col = first_present(player_stats, ["season"])
    week_stat_col = first_present(player_stats, ["week"])
    team_col = first_present(player_stats, ["team", "recent_team", "team_abbr", "club"])
    player_id_col = first_present(player_stats, ["player_id", "gsis_id", "nflverse_player_id"])
    attempts_col = first_present(player_stats, ["pass_attempts", "attempts", "passing_attempts"])
    if None not in (season_stat_col, week_stat_col, team_col, player_id_col, attempts_col):
        qb = pd.DataFrame(
            {
                "season": pd.to_numeric(player_stats[season_stat_col], errors="coerce"),
                "week": pd.to_numeric(player_stats[week_stat_col], errors="coerce"),
                "team": player_stats[team_col],
                "player_id": player_stats[player_id_col].astype(str),
                "attempts": _to_numeric(player_stats[attempts_col]),
            }
        )
        qb = normalize_team_columns(qb, ["team"])
        qb = (
            qb.groupby(["season", "week", "team", "player_id"], as_index=False)["attempts"]
            .sum()
            .sort_values(["season", "week", "team", "attempts"], ascending=[True, True, True, False])
            .drop_duplicates(["season", "week", "team"])
        )
        qb = qb.sort_values(["team", "season", "week"]).reset_index(drop=True)
        shifted = qb.groupby(["team", "season"], sort=False)["player_id"].shift(1)
        qb["qb_primary_changed"] = (
            shifted.notna() & qb["player_id"].notna() & (qb["player_id"] != shifted)
        ).astype(int)
        qb["qb_stability_score"] = 1 - qb["qb_primary_changed"]
        outputs.append(qb[["season", "team", "week", "qb_primary_changed", "qb_stability_score"]])

    season_col = first_present(snap_counts, ["season"])
    week_col = first_present(snap_counts, ["week"])
    snap_team_col = first_present(snap_counts, ["team", "recent_team", "team_abbr", "club"])
    snap_player_col = first_present(snap_counts, ["player_id", "gsis_id", "nflverse_player_id"])
    snap_value_col = first_present(
        snap_counts,
        ["offense_snaps", "off_snaps", "snaps", "offensive_snaps", "snaps_offense"],
    )
    if None not in (season_col, week_col, snap_team_col, snap_player_col, snap_value_col):
        snap = pd.DataFrame(
            {
                "season": pd.to_numeric(snap_counts[season_col], errors="coerce"),
                "week": pd.to_numeric(snap_counts[week_col], errors="coerce"),
                "team": snap_counts[snap_team_col],
                "player_id": snap_counts[snap_player_col].astype(str),
                "snap_value": _to_numeric(snap_counts[snap_value_col]),
            }
        )
        snap = normalize_team_columns(snap, ["team"])
        roster_player_col = first_present(rosters, ["player_id", "gsis_id", "nflverse_player_id"])
        roster_team_col = first_present(rosters, ["team", "recent_team", "team_abbr", "club"])
        roster_position_col = first_present(rosters, ["position", "official_position"])
        if None not in (roster_player_col, roster_position_col):
            roster_subset = pd.DataFrame(
                {
                    "player_id": rosters[roster_player_col].astype(str),
                    "position": rosters[roster_position_col].astype(str).str.upper(),
                }
            ).drop_duplicates("player_id")
            snap = snap.merge(roster_subset, on="player_id", how="left")
        if roster_team_col is not None and "team" not in snap.columns:
            snap["team"] = rosters[roster_team_col]

        oline_positions = {"LT", "LG", "C", "RG", "RT", "OL", "G", "T"}
        skill_positions = {"RB", "FB", "WR", "TE"}
        outputs.extend(
            [
                _overlap_feature(_player_sets(snap, top_k=5, positions=oline_positions), "oline_continuity"),
                _overlap_feature(_player_sets(snap, top_k=5, positions=skill_positions), "skill_group_continuity"),
                _overlap_feature(_player_sets(snap, top_k=15), "roster_fluctuation", invert=True),
            ]
        )

    if not outputs:
        return pd.DataFrame(columns=["season", "team", "week"])

    merged = outputs[0]
    for frame in outputs[1:]:
        merged = merged.merge(frame, on=["season", "team", "week"], how="outer")
    return merged.sort_values(["season", "week", "team"]).reset_index(drop=True)


def aggregate_nextgen_team_week(ngs_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    weight_preferences = {
        "passing": ["attempts", "pass_attempts", "dropbacks"],
        "rushing": ["carries", "rushing_attempts", "rush_attempts"],
        "receiving": ["targets", "receptions"],
    }
    excluded = {
        "season",
        "week",
        "age",
        "game_id",
        "jersey_number",
        "status",
    }
    for kind, frame in ngs_frames.items():
        frame = clean_columns(frame)
        season_col = first_present(frame, ["season"])
        week_col = first_present(frame, ["week"])
        team_col = first_present(frame, ["team", "recent_team", "team_abbr", "club"])
        if None in (season_col, week_col, team_col):
            continue

        dataset = pd.DataFrame(
            {
                "season": pd.to_numeric(frame[season_col], errors="coerce"),
                "week": pd.to_numeric(frame[week_col], errors="coerce"),
                "team": frame[team_col],
            }
        )
        dataset = normalize_team_columns(dataset, ["team"])
        weight_col = first_present(frame, weight_preferences.get(kind, []))
        if weight_col is not None:
            dataset["_weight"] = _to_numeric(frame[weight_col])
        else:
            dataset["_weight"] = 1.0

        metric_cols = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in excluded
            and column not in {season_col, week_col, weight_col}
            and not column.endswith("_id")
            and column not in {"player_id", "gsis_id"}
        ]
        if not metric_cols:
            continue
        for column in metric_cols:
            dataset[column] = _to_numeric(frame[column], np.nan)

        grouped_rows = []
        for (season, team, week), group in dataset.groupby(["season", "team", "week"], dropna=False):
            row = {
                "season": season,
                "team": team,
                "week": week,
                f"ngs_{kind}_weight_sum": group["_weight"].sum(),
            }
            weights = group["_weight"].clip(lower=0)
            for metric in metric_cols:
                values = group[metric]
                if weights.sum() > 0 and values.notna().any():
                    row[f"ngs_{kind}_{metric}"] = np.average(
                        values.fillna(values.mean()).fillna(0.0),
                        weights=weights,
                    )
                else:
                    row[f"ngs_{kind}_{metric}"] = values.mean()
            grouped_rows.append(row)
        outputs.append(pd.DataFrame(grouped_rows))

    if not outputs:
        return pd.DataFrame(columns=["season", "team", "week"])
    merged = outputs[0]
    for frame in outputs[1:]:
        merged = merged.merge(frame, on=["season", "team", "week"], how="outer")
    return merged.sort_values(["season", "week", "team"]).reset_index(drop=True)


def build_team_week_features(
    games: pd.DataFrame,
    pbp_team_game: pd.DataFrame,
    team_stats: pd.DataFrame,
    player_week_features: pd.DataFrame,
    injury_features: pd.DataFrame,
    ngs_team_week: pd.DataFrame,
    contract_features: pd.DataFrame,
    staff_features: pd.DataFrame,
) -> pd.DataFrame:
    team_games = build_team_game_frame(games)
    team_week = team_games.copy()
    if not pbp_team_game.empty and {"season", "week", "game_id", "team"} <= set(pbp_team_game.columns):
        team_week = team_week.merge(
            pbp_team_game,
            on=["season", "week", "game_id", "team"],
            how="left",
        )
    if not team_stats.empty and {"season", "week", "team"} <= set(team_stats.columns):
        team_week = team_week.merge(team_stats, on=["season", "week", "team"], how="left")
    if not player_week_features.empty and {"season", "week", "team"} <= set(player_week_features.columns):
        team_week = team_week.merge(player_week_features, on=["season", "week", "team"], how="left")
    if not injury_features.empty and {"season", "week", "team"} <= set(injury_features.columns):
        team_week = team_week.merge(injury_features, on=["season", "week", "team"], how="left")
        injury_cols = [column for column in injury_features.columns if column.startswith("injury_")]
        available_injury_seasons = set(pd.to_numeric(injury_features["season"], errors="coerce").dropna().astype(int))
        if injury_cols and available_injury_seasons:
            in_available_season = team_week["season"].isin(available_injury_seasons)
            team_week.loc[in_available_season, injury_cols] = team_week.loc[in_available_season, injury_cols].fillna(0.0)
    if not ngs_team_week.empty and {"season", "week", "team"} <= set(ngs_team_week.columns):
        team_week = team_week.merge(ngs_team_week, on=["season", "week", "team"], how="left")
    if not contract_features.empty and {"season", "team"} <= set(contract_features.columns):
        team_week = team_week.merge(contract_features, on=["season", "team"], how="left")
    if not staff_features.empty and {"season", "team"} <= set(staff_features.columns):
        keep_cols = [
            "season",
            "team",
            "hc_changed",
            "oc_changed",
            "dc_changed",
            "gm_changed",
            "hc_tenure_years",
            "gm_tenure_years",
        ]
        existing = [column for column in keep_cols if column in staff_features.columns]
        team_week = team_week.merge(staff_features[existing], on=["season", "team"], how="left")

    team_week = team_week.sort_values(["team", "season", "week", "game_date", "game_id"]).reset_index(drop=True)
    staff_cols = [
        column
        for column in ["hc_changed", "oc_changed", "dc_changed", "gm_changed", "hc_tenure_years", "gm_tenure_years"]
        if column in team_week.columns
    ]
    injury_cols = [column for column in team_week.columns if column.startswith("injury_")]
    if staff_cols:
        team_week[staff_cols] = team_week[staff_cols].fillna(0.0)
        team_week["staff_instability_score"] = team_week[
            [column for column in staff_cols if column.endswith("_changed")]
        ].sum(axis=1)

    rolling_exclusions = {
        "season",
        "week",
        "is_home",
        "home_score",
        "away_score",
        "points_for",
        "points_against",
        "win",
        "point_diff",
        "score_diff",
        "rest_days",
        "home_win",
    }
    rolling_exclusions.update(staff_cols)
    rolling_exclusions.add("staff_instability_score")
    contract_cols = [column for column in team_week.columns if column.startswith("contract_")]
    rolling_exclusions.update(contract_cols)
    rolling_source_cols = [
        column
        for column in team_week.select_dtypes(include="number").columns
        if column not in rolling_exclusions and not column.startswith("pregame_")
    ]

    grouped = team_week.groupby(["team", "season"], sort=False)
    new_columns: dict[str, pd.Series] = {}
    for column in [*rolling_source_cols, "win", "point_diff"]:
        new_columns[f"pregame_{column}_rolling4"] = grouped[column].transform(
            lambda series: series.shift(1).rolling(4, min_periods=1).mean()
        )
        new_columns[f"pregame_{column}_season_avg"] = grouped[column].transform(
            lambda series: series.shift(1).expanding().mean()
        )

    new_columns["pregame_games_played"] = grouped.cumcount()
    new_columns["pregame_rest_days"] = team_week["rest_days"]
    new_columns["pregame_week"] = team_week["week"]
    for column in [*injury_cols, *staff_cols, "staff_instability_score", *contract_cols]:
        if column in team_week.columns:
            new_columns[f"pregame_{column}"] = team_week[column]

    team_week = pd.concat([team_week, pd.DataFrame(new_columns, index=team_week.index)], axis=1)

    opponent_strength = team_week[
        ["season", "week", "team", "pregame_win_season_avg", "pregame_point_diff_season_avg"]
    ].rename(
        columns={
            "team": "opponent",
            "pregame_win_season_avg": "opponent_pregame_win_pct",
            "pregame_point_diff_season_avg": "opponent_pregame_point_diff",
        }
    )
    team_week = team_week.merge(opponent_strength, on=["season", "week", "opponent"], how="left")
    team_week["pregame_strength_of_schedule"] = team_week["opponent_pregame_win_pct"]
    return team_week


def build_model_matrix(team_week_features: pd.DataFrame, regular_season_only: bool = True) -> pd.DataFrame:
    team_week = team_week_features.copy()
    if regular_season_only and "game_type" in team_week.columns:
        team_week = team_week[team_week["game_type"].fillna("REG") == "REG"].copy()

    feature_cols = [
        column
        for column in team_week.select_dtypes(include="number").columns
        if column.startswith("pregame_") and column not in {"pregame_week"}
    ]
    merge_keys = [column for column in ["season", "week", "game_id", "game_date", "game_type"] if column in team_week.columns]

    home = team_week[team_week["is_home"] == 1].copy()
    away = team_week[team_week["is_home"] == 0].copy()
    home = home[merge_keys + ["team", "opponent", "points_for", "points_against", *feature_cols]].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "points_for": "home_score",
            "points_against": "away_score",
            **{column: f"home_{column}" for column in feature_cols},
        }
    )
    away = away[merge_keys + ["team", "opponent", *feature_cols]].rename(
        columns={
            "team": "away_team_check",
            "opponent": "home_team_check",
            **{column: f"away_{column}" for column in feature_cols},
        }
    )
    matrix = home.merge(away, on=merge_keys, how="inner")
    matrix["home_win"] = (matrix["home_score"] > matrix["away_score"]).astype(int)
    diff_columns = {
        f"{column}_diff": matrix[f"home_{column}"] - matrix[f"away_{column}"]
        for column in feature_cols
    }
    matrix = pd.concat([matrix, pd.DataFrame(diff_columns, index=matrix.index)], axis=1)
    if {"away_team", "away_team_check", "home_team", "home_team_check"} <= set(matrix.columns):
        matrix = matrix[
            matrix["away_team"].eq(matrix["away_team_check"]) & matrix["home_team"].eq(matrix["home_team_check"])
        ].copy()
        matrix = matrix.drop(columns=["away_team_check", "home_team_check"])
    return matrix.sort_values(["season", "week", "game_date", "game_id"]).reset_index(drop=True)
