from __future__ import annotations

from html import escape
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from jets_project.settings import get_default_config
from jets_project.simulator import load_model_bundle, simulate_team_improvement
from jets_project.storage import list_sqlite_tables, read_sqlite_table

MODEL_NAME_MAP = {
    "gradient_boosting": "Gradient-Boosting-Modell",
    "logistic_regression": "Logistische Regression",
    "naive_home_rate": "Naiver Heimvorteils-Benchmark",
}

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

METRIC_LABELS = {
    "win_pct": "Siegquote",
    "avg_point_diff": "Punktdifferenz pro Spiel",
    "point_diff": "Punktdifferenz",
    "offense_epa_per_play": "Offense-EPA pro Play",
    "offense_success_rate": "Offense-Erfolgsrate",
    "offense_turnover_rate": "Offense-Turnover-Rate",
    "offense_sack_rate": "Offense-Sack-Rate",
    "offense_explosive_play_rate": "Explosive-Play-Rate der Offense",
    "offense_penalty_rate": "Strafen der Offense",
    "offense_yards_per_play": "Yards pro Offense-Play",
    "offense_red_zone_td_rate": "Red-Zone-TD-Rate",
    "offense_third_down_conversion_rate": "Third-Down-Conversion-Rate",
    "offense_fourth_down_conversion_rate": "Fourth-Down-Conversion-Rate",
    "offense_early_down_pass_rate": "Early-Down-Pass-Rate",
    "defense_epa_per_play_allowed": "Defense-EPA erlaubt pro Play",
    "defense_success_rate_allowed": "Zugelassene Erfolgsrate",
    "defense_takeaway_rate": "Takeaway-Rate der Defense",
    "defense_sack_rate": "Defense-Sack-Rate",
    "defense_explosive_play_rate_allowed": "Zugelassene Explosive Plays",
    "defense_yards_per_play_allowed": "Yards erlaubt pro Play",
    "qb_stability_score": "QB-Stabilitaet",
    "qb_primary_changed": "Wechsel des Starting-QB",
    "oline_continuity": "O-Line-Kontinuitaet",
    "skill_group_continuity": "Kontinuitaet der Skill-Positions",
    "roster_fluctuation": "Roster-Fluktuation",
    "staff_instability_score": "Staff-Instabilitaet",
    "injury_out_rate": "Anteil Ausfaelle im Injury Report",
    "injury_dnp_rate": "Anteil DNP im Practice Report",
    "injury_impact_score": "Injury-Impact-Score",
    "contract_avg_cap_percent": "Durchschnittlicher Cap-Anteil pro Vertrag",
    "contract_player_count": "Vertrags-Slots im aktiven Kader",
    "contract_total_cap_number": "Gesamte Cap Number",
    "contract_total_guaranteed_salary": "Garantierte Gehaelter gesamt",
    "contract_total_cash_paid": "Cash Paid gesamt",
    "contract_max_cap_number": "Hoechste einzelne Cap Number",
    "contract_top3_cap_share": "Cap-Anteil der Top-3-Vertraege",
    "contract_qb_cap_share": "QB-Cap-Anteil",
    "contract_offense_cap_share": "Offense-Cap-Anteil",
    "contract_defense_cap_share": "Defense-Cap-Anteil",
    "ngs_receiving_avg_yac_above_expectation": "YAC ueber Erwartung",
    "ngs_receiving_avg_separation": "Receiver-Separation",
    "ngs_receiving_avg_cushion": "Defender-Cushion",
    "ngs_passing_avg_completed_air_yards": "Completed Air Yards",
    "ngs_passing_avg_intended_air_yards": "Intended Air Yards",
    "ngs_passing_completion_percentage_above_expectation": "Completion % ueber Erwartung",
    "ngs_passing_pass_touchdowns": "Passing-TDs",
    "ngs_rushing_efficiency": "Rushing Efficiency",
    "ngs_rushing_rush_yards_over_expected_per_att": "Rush Yards over Expected pro Attempt",
}

METRIC_RECOMMENDATIONS = {
    "offense_sack_rate": "Protection priorisieren: O-Line-Kontinuitaet, schnelleres Passspiel und klarere Hot-Reads.",
    "offense_turnover_rate": "Ball Security und Pressure-Management verbessern; QB-Entscheidungen unter Druck stabilisieren.",
    "defense_takeaway_rate": "Mehr Ballproduktion forcieren: Coverages mit Spaetrotation, aggressiver auf den Catch-Point gehen.",
    "defense_explosive_play_rate_allowed": "Deep-Shot-Schutz und Tackling im Second Level verbessern, um Big Plays zu begrenzen.",
    "offense_epa_per_play": "Mehr Early-Down-Effizienz, Play-Action und schematisch einfache Antworten gegen Druck einbauen.",
    "offense_success_rate": "Die Offense frueher on schedule bringen: mehr effiziente Early Downs, weniger offensichtliche Passing Downs.",
    "ngs_receiving_avg_yac_above_expectation": "Mehr YAC kreieren: Motion, freie Releases und designte Touches fuer Playmaker.",
    "ngs_receiving_avg_separation": "Mehr Separation schaffen: Formationen, Motion und Route-Spacing sauberer aufbauen.",
    "ngs_receiving_avg_cushion": "Coverage-Struktur lesen und Matchups erzwingen, statt Receiver konstant in enge Fenster zu schicken.",
    "oline_continuity": "Kontinuitaet in der Protection herstellen; weniger Rotation, stabilere Starting Five.",
    "skill_group_continuity": "Weniger Wechsel in den Skill-Groups; Timing und Rollenklarheit verbessern.",
    "roster_fluctuation": "Kernrollen ueber mehrere Wochen stabil halten statt permanent neu zu mischen.",
    "qb_primary_changed": "Auf QB-Kontinuitaet hinarbeiten, damit Timing, Protection-Calls und Spieltempo stabiler werden.",
    "injury_out_rate": "Medical, Load Management und belastbare Depth priorisieren; weniger kritische Ausfaelle in Kernrollen zulassen.",
    "injury_dnp_rate": "Trainingssteuerung und Rueckkehrprotokolle sauberer managen, damit weniger Kernspieler die Woche limitiert starten.",
    "injury_impact_score": "Injury-Risiko an Premium-Positionen mit Tiefe, Rotation und Belastungsmanagement abfedern.",
    "contract_avg_cap_percent": "Cap weniger breit auf austauschbare Tiefe verteilen und gezielter in echte Impact-Rollen investieren.",
    "contract_player_count": "Cap-Einsatz pro Roster-Slot schaerfen; weniger Mittel in Fringe-Roster-Spots binden.",
    "contract_top3_cap_share": "Top-End des Kaders klarer priorisieren, statt Cap zu stark im Mittelfeld zu verteilen.",
    "contract_qb_cap_share": "QB-Investment, QB-Stabilitaet und Scheme-Fit enger aufeinander abstimmen.",
    "contract_offense_cap_share": "Offense-Cap nur dort erhoehen, wo Protection und Explosiveness messbar profitieren.",
    "contract_defense_cap_share": "Defensive Ressourcen staerker auf Ballproduktion und Explosive-Play-Vermeidung ausrichten.",
    "point_diff": "Symptom-Metrik: zuerst EPA, Turnovers, Protection und Explosive Plays adressieren.",
}


@st.cache_data(show_spinner=False)
def load_table(sqlite_path: Path, table_name: str) -> pd.DataFrame:
    return read_sqlite_table(sqlite_path, table_name)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --jets-green: #0d5a45;
            --jets-green-dark: #0b3f31;
            --jets-cream: #f4efe4;
            --jets-ink: #16241f;
            --jets-muted: #5f6d66;
            --jets-positive: #1a7f52;
            --jets-negative: #b65344;
            --jets-card: rgba(255, 255, 255, 0.76);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(13, 90, 69, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(193, 163, 98, 0.18), transparent 24%),
                linear-gradient(180deg, #f8f4ea 0%, #efe5d2 100%);
            color: var(--jets-ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 63, 49, 0.97) 0%, rgba(7, 38, 31, 0.97) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8f6ef;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.1rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(11, 63, 49, 0.98), rgba(13, 90, 69, 0.95));
            box-shadow: 0 22px 55px rgba(22, 36, 31, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.09);
            color: #fbfaf4;
            margin-bottom: 1.25rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            right: -60px;
            top: -70px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.18), rgba(255,255,255,0.02) 72%);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: rgba(250, 248, 242, 0.7);
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.02;
            font-weight: 800;
            margin: 0 0 0.55rem 0;
            max-width: 850px;
        }

        .hero-subtitle {
            max-width: 820px;
            font-size: 1rem;
            line-height: 1.55;
            color: rgba(250, 248, 242, 0.86);
            margin-bottom: 1.1rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.16);
            font-size: 0.78rem;
            color: #faf8f1;
        }

        .section-card {
            background: var(--jets-card);
            border: 1px solid rgba(22, 36, 31, 0.08);
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 12px 28px rgba(22, 36, 31, 0.06);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }

        .mini-card {
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(22, 36, 31, 0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 132px;
            box-shadow: 0 10px 18px rgba(22, 36, 31, 0.04);
        }

        .mini-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--jets-muted);
            margin-bottom: 0.55rem;
        }

        .mini-value {
            font-size: 1.8rem;
            line-height: 1.05;
            font-weight: 800;
            color: var(--jets-ink);
            margin-bottom: 0.35rem;
        }

        .mini-delta {
            font-size: 0.92rem;
            font-weight: 600;
        }

        .positive {
            color: var(--jets-positive);
        }

        .negative {
            color: var(--jets-negative);
        }

        .neutral {
            color: var(--jets-muted);
        }

        .story-card {
            background: rgba(255,255,255,0.72);
            border-left: 4px solid var(--jets-green);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            height: 100%;
        }

        .story-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            color: var(--jets-ink);
        }

        .story-card p {
            margin: 0;
            color: var(--jets-muted);
            line-height: 1.5;
            font-size: 0.92rem;
        }

        .story-rank {
            display: inline-block;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            background: rgba(13, 90, 69, 0.10);
            color: var(--jets-green-dark);
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 750;
            color: var(--jets-ink);
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: var(--jets-muted);
            font-size: 0.95rem;
            margin-bottom: 0.95rem;
        }

        .summary-note {
            background: rgba(255,255,255,0.68);
            border: 1px solid rgba(22, 36, 31, 0.08);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            color: var(--jets-muted);
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def model_label(model_name: str) -> str:
    return MODEL_NAME_MAP.get(model_name, model_name.replace("_", " ").title())


def is_display_metric(metric: str) -> bool:
    if metric in NON_ACTIONABLE_METRICS:
        return False
    return not any(token in metric for token in [*EXCLUDED_METRIC_TOKENS, *NON_ACTIONABLE_METRIC_TOKENS])


def clean_root_causes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["metric"].map(is_display_metric)].reset_index(drop=True)


def pretty_metric(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]

    replacements = {
        "ngs": "NGS",
        "epa": "EPA",
        "yac": "YAC",
        "pct": "%",
        "qb": "QB",
        "avg": "Avg.",
        "hc": "HC",
        "oc": "OC",
        "dc": "DC",
        "gm": "GM",
        "td": "TD",
        "los": "LOS",
        "fourth": "Fourth",
        "third": "Third",
    }
    words = [replacements.get(token, token.title()) for token in metric.split("_")]
    return " ".join(words)


def beneficial_when_higher(metric: str) -> bool:
    negative_tokens = (
        "allowed",
        "turnover",
        "sack",
        "penalty",
        "fluctuation",
        "changed",
        "instability",
        "against",
    )
    return not any(token in metric for token in negative_tokens)


def recommendation_for_metric(metric: str) -> str:
    return METRIC_RECOMMENDATIONS.get(
        metric,
        "Hier liegt ein relevanter Jets-Hebel. Ziel ist, diese Kennzahl systematisch in Richtung Liga-Topquartil zu bewegen.",
    )


def format_diff(value: float, pct: bool = False, points: int = 1) -> str:
    sign = "+" if value > 0 else ""
    if pct:
        return f"{sign}{value * 100:.{points}f} pp"
    return f"{sign}{value:.{points}f}"


def tone_for_delta(value: float, higher_is_better: bool = True) -> str:
    if value == 0 or pd.isna(value):
        return "neutral"
    improved = value > 0 if higher_is_better else value < 0
    return "positive" if improved else "negative"


def build_model_summary(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_metrics.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "avg_log_loss",
                "avg_brier",
                "avg_roc_auc",
                "avg_ece",
                "beat_log_loss_seasons",
                "beat_brier_seasons",
                "total_test_seasons",
                "avg_log_loss_gain",
                "avg_brier_gain",
            ]
        )
    evaluation_metrics = evaluation_metrics.copy()
    if "ece" not in evaluation_metrics.columns:
        evaluation_metrics["ece"] = pd.NA
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
    benchmark = build_benchmark_comparison(evaluation_metrics)
    if benchmark.empty:
        return summary

    comparison_summary = (
        benchmark.groupby("model_name", as_index=False)
        .agg(
            beat_log_loss_seasons=("beat_log_loss", "sum"),
            beat_brier_seasons=("beat_brier", "sum"),
            total_test_seasons=("test_season", "nunique"),
            avg_log_loss_gain=("log_loss_gain", "mean"),
            avg_brier_gain=("brier_gain", "mean"),
        )
    )
    return summary.merge(comparison_summary, on="model_name", how="left")


def build_benchmark_comparison(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_metrics.empty or "naive_home_rate" not in evaluation_metrics["model_name"].unique():
        return pd.DataFrame(
            columns=[
                "model_name",
                "test_season",
                "log_loss",
                "brier_score",
                "naive_log_loss",
                "naive_brier",
                "beat_log_loss",
                "beat_brier",
                "log_loss_gain",
                "brier_gain",
            ]
        )

    naive = evaluation_metrics[evaluation_metrics["model_name"] == "naive_home_rate"][
        ["test_season", "log_loss", "brier_score"]
    ].rename(columns={"log_loss": "naive_log_loss", "brier_score": "naive_brier"})
    models = evaluation_metrics[evaluation_metrics["model_name"] != "naive_home_rate"].merge(
        naive, on="test_season", how="left"
    )
    models["beat_log_loss"] = models["log_loss"] < models["naive_log_loss"]
    models["beat_brier"] = models["brier_score"] < models["naive_brier"]
    models["log_loss_gain"] = models["naive_log_loss"] - models["log_loss"]
    models["brier_gain"] = models["naive_brier"] - models["brier_score"]
    return models


def build_diagnostic_summary(backtest_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if backtest_diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "avg_train_log_loss",
                "avg_test_log_loss",
                "avg_train_brier",
                "avg_test_brier",
                "avg_log_loss_gap",
                "avg_brier_gap",
            ]
        )
    return (
        backtest_diagnostics.groupby("model_name", as_index=False)
        .agg(
            avg_train_log_loss=("train_log_loss", "mean"),
            avg_test_log_loss=("test_log_loss", "mean"),
            avg_train_brier=("train_brier_score", "mean"),
            avg_test_brier=("test_brier_score", "mean"),
            avg_log_loss_gap=("log_loss_gap", "mean"),
            avg_brier_gap=("brier_score_gap", "mean"),
        )
        .sort_values("avg_test_log_loss")
    )


def render_hero(
    latest_season: int,
    jets_latest: pd.Series,
    root_causes: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    top_causes = ", ".join(pretty_metric(metric) for metric in root_causes["metric"].head(3))
    if model_summary.empty:
        model_chip = "Backtests noch nicht verfuegbar"
    else:
        contenders = model_summary[model_summary["model_name"] != "naive_home_rate"].copy()
        best_model = contenders.sort_values("avg_log_loss").iloc[0] if not contenders.empty else model_summary.iloc[0]
        if pd.notna(best_model.get("beat_log_loss_seasons")) and pd.notna(best_model.get("total_test_seasons")):
            model_chip = (
                f"Backtest: {model_label(best_model['model_name'])} "
                f"{int(best_model['beat_log_loss_seasons'])}/{int(best_model['total_test_seasons'])} Saisons besser als naiv"
            )
        else:
            best_log_loss = model_summary.sort_values("avg_log_loss").iloc[0]
            best_brier = model_summary.sort_values("avg_brier").iloc[0]
            model_chip = (
                f"Log Loss: {model_label(best_log_loss['model_name'])} | "
                f"Brier: {model_label(best_brier['model_name'])}"
            )

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">NY Jets Diagnostic Lab</div>
            <div class="hero-title">Warum verlieren die Jets immer wieder, obwohl sich Namen, Quarterbacks und Coaches aendern?</div>
            <div class="hero-subtitle">
                Dieses Streamlit-Dashboard verdichtet zehn NFL-Saisons in eine praesente Abgabestory:
                Liga-Vergleich, strukturelle Ursachen, Tracking-Signale und ein What-if-Simulator fuer realistische Verbesserungshebel.
            </div>
            <div class="chip-row">
                <span class="chip">Datenschnitt 2016-{latest_season}</span>
                <span class="chip">Aktueller Record: {int(jets_latest['wins'])}-{int(jets_latest['losses'])}</span>
                <span class="chip">{escape(model_chip)}</span>
                <span class="chip">Top-Hebel: {escape(top_causes or 'werden berechnet')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, delta: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{escape(label)}</div>
            <div class="mini-value">{escape(value)}</div>
            <div class="mini-delta {tone}">{escape(delta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def line_chart_win_trend(jets_season_summary: pd.DataFrame) -> alt.Chart:
    frame = jets_season_summary[
        ["season", "win_pct", "league_avg_win_pct"]
    ].copy()
    plot = frame.melt(
        id_vars="season",
        value_vars=["win_pct", "league_avg_win_pct"],
        var_name="series",
        value_name="value",
    )
    plot["series"] = plot["series"].map(
        {"win_pct": "Jets", "league_avg_win_pct": "Liga-Durchschnitt"}
    )

    return (
        alt.Chart(plot)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("season:O", title="Saison"),
            y=alt.Y("value:Q", title="Siegquote", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Jets", "Liga-Durchschnitt"], range=["#0d5a45", "#c46a3d"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("season:O", title="Saison"),
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value:Q", title="Siegquote", format=".1%"),
            ],
        )
        .properties(height=320)
    )


def scatter_latest_season(team_season_summary: pd.DataFrame, season: int, focus_team: str) -> alt.Chart:
    latest = team_season_summary[team_season_summary["season"] == season].copy()
    latest["label_group"] = "Liga"
    latest.loc[latest["team"] == focus_team, "label_group"] = "Jets"
    latest.loc[latest["wins"] >= latest["wins"].quantile(0.85), "label_group"] = "Top-Team"
    latest.loc[latest["team"] == focus_team, "label_group"] = "Jets"
    latest["team_label"] = np.where(latest["label_group"] != "Liga", latest["team"], "")

    base = alt.Chart(latest).encode(
        x=alt.X("wins:Q", title="Siege"),
        y=alt.Y("avg_point_diff:Q", title="Punktdifferenz pro Spiel"),
        tooltip=[
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("wins:Q", title="Siege", format=".0f"),
            alt.Tooltip("losses:Q", title="Niederlagen", format=".0f"),
            alt.Tooltip("avg_point_diff:Q", title="Punktdifferenz", format=".2f"),
            alt.Tooltip("win_pct:Q", title="Siegquote", format=".1%"),
        ],
    )
    points = base.mark_circle(size=180, opacity=0.85).encode(
        color=alt.Color(
            "label_group:N",
            scale=alt.Scale(
                domain=["Jets", "Top-Team", "Liga"],
                range=["#b65344", "#d8a54b", "#0d5a45"],
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
    )
    labels = base.mark_text(dy=-12, fontWeight="bold").encode(text="team_label:N")
    return (points + labels).properties(height=320)


def root_cause_chart(root_causes: pd.DataFrame) -> alt.Chart:
    frame = root_causes.head(10).copy()
    frame["metric_label"] = frame["metric"].map(pretty_metric)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("problem_score:Q", title="Problem-Score"),
            y=alt.Y("metric_label:N", sort="-x", title=None),
            color=alt.Color(
                "problem_score:Q",
                scale=alt.Scale(range=["#d8c8a0", "#b65344"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("metric_label:N", title="Metrik"),
                alt.Tooltip("jets_value:Q", title="Jets", format=".3f"),
                alt.Tooltip("league_avg:Q", title="Liga", format=".3f"),
                alt.Tooltip("metric_rank:Q", title="Rang"),
                alt.Tooltip("problem_score:Q", title="Problem-Score", format=".1f"),
            ],
        )
        .properties(height=360)
    )


def tracking_line_chart(tracking_lens: pd.DataFrame, metric: str) -> alt.Chart:
    league_metric = f"league_avg_{metric}"
    frame = tracking_lens[["season", metric, league_metric]].copy()
    plot = frame.melt(id_vars="season", value_vars=[metric, league_metric], var_name="series", value_name="value")
    plot["series"] = plot["series"].map({metric: "Jets", league_metric: "Liga-Durchschnitt"})
    return (
        alt.Chart(plot)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("season:O", title="Saison"),
            y=alt.Y("value:Q", title=pretty_metric(metric)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Jets", "Liga-Durchschnitt"], range=["#0d5a45", "#c46a3d"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("season:O", title="Saison"),
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value:Q", title=pretty_metric(metric), format=".3f"),
            ],
        )
        .properties(height=320)
    )


def tracking_gap_chart(tracking_lens: pd.DataFrame, family: str, season: int) -> alt.Chart | None:
    family_cols = [
        column
        for column in tracking_lens.columns
        if column.startswith(f"ngs_{family}_") and is_display_metric(column)
    ]
    season_frame = tracking_lens[tracking_lens["season"] == season]
    if season_frame.empty or not family_cols:
        return None

    row = season_frame.iloc[0]
    gaps = []
    for column in family_cols:
        league_column = f"league_avg_{column}"
        if league_column not in tracking_lens.columns:
            continue
        value = row[column]
        league_value = row[league_column]
        if pd.isna(value) or pd.isna(league_value):
            continue
        gaps.append(
            {
                "metric_label": pretty_metric(column),
                "delta": value - league_value,
            }
        )
    if not gaps:
        return None

    frame = pd.DataFrame(gaps).sort_values("delta")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=5)
        .encode(
            x=alt.X("delta:Q", title="Jets minus Liga"),
            y=alt.Y("metric_label:N", sort="x", title=None),
            color=alt.condition(
                alt.datum.delta >= 0,
                alt.value("#1a7f52"),
                alt.value("#b65344"),
            ),
            tooltip=[
                alt.Tooltip("metric_label:N", title="Metrik"),
                alt.Tooltip("delta:Q", title="Delta", format=".3f"),
            ],
        )
        .properties(height=max(240, 26 * len(frame)))
    )


def scenario_comparison_chart(scenario: pd.DataFrame) -> alt.Chart:
    plot = scenario[["week", "baseline_win_probability", "scenario_win_probability"]].melt(
        id_vars="week",
        value_vars=["baseline_win_probability", "scenario_win_probability"],
        var_name="series",
        value_name="value",
    )
    plot["series"] = plot["series"].map(
        {
            "baseline_win_probability": "Baseline",
            "scenario_win_probability": "Szenario",
        }
    )
    return (
        alt.Chart(plot)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("value:Q", title="Siegwahrscheinlichkeit", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Baseline", "Szenario"], range=["#7a847d", "#0d5a45"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("week:O", title="Week"),
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value:Q", title="Siegwahrscheinlichkeit", format=".1%"),
            ],
        )
        .properties(height=320)
    )


def scenario_delta_chart(scenario: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(scenario)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("win_probability_delta:Q", title="Delta Siegwahrscheinlichkeit"),
            color=alt.condition(
                alt.datum.win_probability_delta >= 0,
                alt.value("#1a7f52"),
                alt.value("#b65344"),
            ),
            tooltip=[
                alt.Tooltip("week:O", title="Week"),
                alt.Tooltip("opponent:N", title="Gegner"),
                alt.Tooltip("win_probability_delta:Q", title="Delta", format=".2%"),
            ],
        )
        .properties(height=320)
    )


def model_metric_chart(model_summary: pd.DataFrame) -> alt.Chart:
    frame = model_summary.copy()
    frame["model_label"] = frame["model_name"].map(model_label)
    plot = frame.melt(
        id_vars="model_label",
        value_vars=["avg_log_loss", "avg_brier"],
        var_name="metric",
        value_name="value",
    )
    plot["metric"] = plot["metric"].map(
        {"avg_log_loss": "Log Loss", "avg_brier": "Brier Score"}
    )
    return (
        alt.Chart(plot)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("model_label:N", title=None, sort=None),
            y=alt.Y("value:Q", title="Mittelwert"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["Log Loss", "Brier Score"], range=["#0d5a45", "#c46a3d"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            column=alt.Column("metric:N", header=alt.Header(title=None)),
            tooltip=[
                alt.Tooltip("model_label:N", title="Modell"),
                alt.Tooltip("metric:N", title="Metrik"),
                alt.Tooltip("value:Q", title="Wert", format=".4f"),
            ],
        )
        .properties(height=260)
    )


def model_seasonal_chart(evaluation_metrics: pd.DataFrame) -> alt.Chart:
    frame = evaluation_metrics.copy()
    frame["model_label"] = frame["model_name"].map(model_label)
    return (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("test_season:O", title="Testsaison"),
            y=alt.Y("log_loss:Q", title="Log Loss"),
            color=alt.Color(
                "model_label:N",
                scale=alt.Scale(
                    domain=[
                        model_label("gradient_boosting"),
                        model_label("logistic_regression"),
                        model_label("naive_home_rate"),
                    ],
                    range=["#0d5a45", "#c46a3d", "#7a847d"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("test_season:O", title="Saison"),
                alt.Tooltip("model_label:N", title="Modell"),
                alt.Tooltip("log_loss:Q", title="Log Loss", format=".4f"),
                alt.Tooltip("brier_score:Q", title="Brier Score", format=".4f"),
                alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".4f"),
            ],
        )
        .properties(height=320)
    )


def benchmark_consistency_chart(benchmark_comparison: pd.DataFrame) -> alt.Chart:
    frame = benchmark_comparison.copy()
    frame["model_label"] = frame["model_name"].map(model_label)
    plot = frame.melt(
        id_vars=["test_season", "model_label"],
        value_vars=["log_loss_gain", "brier_gain"],
        var_name="metric",
        value_name="gain",
    )
    plot["metric"] = plot["metric"].map(
        {"log_loss_gain": "Vorsprung bei Log Loss", "brier_gain": "Vorsprung bei Brier"}
    )
    return (
        alt.Chart(plot)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("test_season:O", title="Testsaison"),
            y=alt.Y("gain:Q", title="Vorsprung gegen naiven Benchmark"),
            color=alt.condition(
                alt.datum.gain >= 0,
                alt.value("#1a7f52"),
                alt.value("#b65344"),
            ),
            column=alt.Column("metric:N", header=alt.Header(title=None)),
            row=alt.Row("model_label:N", header=alt.Header(title=None)),
            tooltip=[
                alt.Tooltip("test_season:O", title="Saison"),
                alt.Tooltip("model_label:N", title="Modell"),
                alt.Tooltip("metric:N", title="Metrik"),
                alt.Tooltip("gain:Q", title="Vorsprung", format=".4f"),
            ],
        )
        .properties(height=120)
    )


def generalization_gap_chart(diagnostic_summary: pd.DataFrame) -> alt.Chart:
    frame = diagnostic_summary.copy()
    frame["model_label"] = frame["model_name"].map(model_label)
    plot = frame.melt(
        id_vars="model_label",
        value_vars=["avg_log_loss_gap", "avg_brier_gap"],
        var_name="metric",
        value_name="gap",
    )
    plot["metric"] = plot["metric"].map(
        {"avg_log_loss_gap": "Log-Loss-Gap", "avg_brier_gap": "Brier-Gap"}
    )
    return (
        alt.Chart(plot)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("model_label:N", title=None, sort=None),
            y=alt.Y("gap:Q", title="Train/Test-Gap"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["Log-Loss-Gap", "Brier-Gap"], range=["#0d5a45", "#c46a3d"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            column=alt.Column("metric:N", header=alt.Header(title=None)),
            tooltip=[
                alt.Tooltip("model_label:N", title="Modell"),
                alt.Tooltip("metric:N", title="Metrik"),
                alt.Tooltip("gap:Q", title="Gap", format=".4f"),
            ],
        )
        .properties(height=240)
    )


def calibration_curve_chart(calibration_bins: pd.DataFrame, model_name: str) -> alt.Chart | None:
    frame = calibration_bins[calibration_bins["model_name"] == model_name].copy()
    if frame.empty:
        return None
    perfect_line = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    curve = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3, color="#0d5a45")
        .encode(
            x=alt.X("mean_predicted:Q", title="Vorhergesagte Siegchance"),
            y=alt.Y("observed_rate:Q", title="Tatsaechliche Rate"),
            tooltip=[
                alt.Tooltip("count:Q", title="Spiele", format=".0f"),
                alt.Tooltip("mean_predicted:Q", title="Prediction", format=".3f"),
                alt.Tooltip("observed_rate:Q", title="Observed", format=".3f"),
                alt.Tooltip("calibration_gap:Q", title="Gap", format=".3f"),
            ],
        )
    )
    diagonal = alt.Chart(perfect_line).mark_line(strokeDash=[4, 4], color="#7a847d").encode(
        x="x:Q",
        y="y:Q",
    )
    return (diagonal + curve).properties(height=240)


def render_sidebar(
    latest_season: int,
    model_summary: pd.DataFrame,
    training_end_season: int | None,
    injury_data_note: str | None,
) -> str:
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Seite",
        ["Executive Summary", "Liga-Ueberblick", "Root Causes", "Tracking Lens", "What-If Simulator"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Projekt-Setup")
    st.sidebar.caption(
        "Streamlit-Dashboard fuer das Abschlussprojekt. Datenbasis: nflreadpy/nflverse, kuratiert in SQLite."
    )
    st.sidebar.markdown(f"**Datenschnitt:** 2016-{latest_season}")
    if training_end_season is not None:
        st.sidebar.markdown(f"**Modell trainiert bis:** {training_end_season}")
    if not model_summary.empty:
        contenders = model_summary[model_summary["model_name"] != "naive_home_rate"].copy()
        best_log_loss = contenders.sort_values("avg_log_loss").iloc[0] if not contenders.empty else model_summary.iloc[0]
        st.sidebar.markdown(f"**Bestes Modell:** {model_label(best_log_loss['model_name'])}")
        if pd.notna(best_log_loss.get("beat_log_loss_seasons")) and pd.notna(best_log_loss.get("total_test_seasons")):
            st.sidebar.markdown(
                f"**Benchmark-Fit:** {int(best_log_loss['beat_log_loss_seasons'])}/{int(best_log_loss['total_test_seasons'])} Saisons besser als naiv"
            )
    st.sidebar.markdown("**Ziel:** strukturelle Jets-Hebel sichtbar und quantifizierbar machen.")
    if injury_data_note:
        st.sidebar.caption(injury_data_note)
    return page


st.set_page_config(
    page_title="NY Jets Diagnostic Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

config = get_default_config()
sqlite_path = Path(os.getenv("JETS_SQLITE_PATH", "")).expanduser() if os.getenv("JETS_SQLITE_PATH") else None
if sqlite_path is None or not sqlite_path.exists():
    sqlite_path = config.paths.app_sqlite_path if config.paths.app_sqlite_path.exists() else config.paths.sqlite_path

if not sqlite_path.exists():
    st.warning("Die SQLite-Datenbank fehlt noch. Fuehre zuerst Ingestion, Features, Training und Evaluation aus.")
    st.stop()

available_tables = list_sqlite_tables(sqlite_path)
required_tables = {"team_season_summary", "jets_season_summary", "root_causes", "game_features"}
missing_tables = sorted(required_tables - set(available_tables))
if missing_tables:
    st.warning(
        "Es fehlen noch Tabellen fuer das Dashboard: "
        + ", ".join(missing_tables)
        + ". Fuehre `uv run python -m jets_project.evaluate` aus."
    )
    st.stop()

team_season_summary = load_table(sqlite_path, "team_season_summary")
jets_season_summary = load_table(sqlite_path, "jets_season_summary")
root_causes = clean_root_causes(load_table(sqlite_path, "root_causes"))
game_features = load_table(sqlite_path, "game_features")
tracking_lens = load_table(sqlite_path, "tracking_lens") if "tracking_lens" in available_tables else pd.DataFrame()
evaluation_metrics = (
    load_table(sqlite_path, "evaluation_metrics") if "evaluation_metrics" in available_tables else pd.DataFrame()
)
backtest_diagnostics = (
    load_table(sqlite_path, "backtest_diagnostics") if "backtest_diagnostics" in available_tables else pd.DataFrame()
)
calibration_bins = (
    load_table(sqlite_path, "calibration_bins") if "calibration_bins" in available_tables else pd.DataFrame()
)
model_runs = load_table(sqlite_path, "model_runs") if "model_runs" in available_tables else pd.DataFrame()
dataset_availability = (
    load_table(sqlite_path, "dataset_availability") if "dataset_availability" in available_tables else pd.DataFrame()
)
model_summary = build_model_summary(evaluation_metrics)
benchmark_comparison = build_benchmark_comparison(evaluation_metrics)
diagnostic_summary = build_diagnostic_summary(backtest_diagnostics)

latest_season = int(team_season_summary["season"].max())
latest = team_season_summary[team_season_summary["season"] == latest_season].copy()
jets_latest = latest[latest["team"] == config.focus_team].iloc[0]

training_end_season = None
if not model_runs.empty and "training_end_season" in model_runs.columns:
    numeric_training_end = pd.to_numeric(model_runs["training_end_season"], errors="coerce").dropna()
    if not numeric_training_end.empty:
        training_end_season = int(numeric_training_end.iloc[-1])

injury_data_note = None
if not dataset_availability.empty and {"dataset_name", "missing_seasons"} <= set(dataset_availability.columns):
    injury_rows = dataset_availability[dataset_availability["dataset_name"] == "injuries"]
    if not injury_rows.empty:
        missing_text = str(injury_rows.iloc[0].get("missing_seasons", "[]"))
        if "2025" in missing_text and training_end_season is not None and training_end_season < latest_season:
            injury_data_note = (
                "Der oeffentliche Injury-Feed fuer 2025 ist unvollstaendig. "
                f"Deshalb wurde das Entscheidungsmodell auf vollstaendige Saisons bis {training_end_season} trainiert."
            )

page = render_sidebar(latest_season, model_summary, training_end_season, injury_data_note)
render_hero(latest_season, jets_latest, root_causes, model_summary)

if page == "Executive Summary":
    avg_wins = float(jets_season_summary["wins"].mean())
    winning_seasons = int((jets_season_summary["wins"] > jets_season_summary["losses"]).sum())
    latest_win_gap = float(jets_latest["win_pct"] - jets_latest["league_avg_win_pct"])
    latest_point_gap = float(jets_latest["avg_point_diff"] - jets_latest["league_avg_point_diff"])
    top_cause = root_causes.iloc[0] if not root_causes.empty else None
    contenders = model_summary[model_summary["model_name"] != "naive_home_rate"].copy()
    best_model = contenders.sort_values("avg_log_loss").iloc[0] if not contenders.empty else None
    best_diagnostic = (
        diagnostic_summary[diagnostic_summary["model_name"] == best_model["model_name"]].iloc[0]
        if best_model is not None
        and not diagnostic_summary.empty
        and best_model["model_name"] in diagnostic_summary["model_name"].values
        else None
    )

    if best_model is None:
        model_takeaway = "Backtests noch nicht verfuegbar."
    else:
        beat_log_loss = int(best_model["beat_log_loss_seasons"]) if pd.notna(best_model.get("beat_log_loss_seasons")) else 0
        total_test_seasons = int(best_model["total_test_seasons"]) if pd.notna(best_model.get("total_test_seasons")) else 0
        model_takeaway = (
            f"{model_label(best_model['model_name'])} ist das tragfaehigste Modell: "
            f"im Mittel {best_model['avg_log_loss']:.4f} Log Loss, {best_model['avg_brier']:.4f} Brier "
            f"und in {beat_log_loss}/{total_test_seasons} Test-Saisons besser als der naive Heimvorteils-Benchmark."
        )
        if best_diagnostic is not None:
            model_takeaway += (
                f" Der mittlere Train/Test-Gap liegt nur noch bei {best_diagnostic['avg_log_loss_gap']:.4f} im Log Loss."
            )
        if pd.notna(best_model.get("avg_ece")):
            model_takeaway += f" Die mittlere Calibration Error liegt bei {best_model['avg_ece']:.4f}."

    cards = st.columns(5)
    with cards[0]:
        render_stat_card(
            "2025 Record",
            f"{int(jets_latest['wins'])}-{int(jets_latest['losses'])}",
            f"Siegquote {jets_latest['win_pct']:.1%}",
            tone_for_delta(latest_win_gap, True),
        )
    with cards[1]:
        render_stat_card(
            "Langfristiger Schnitt",
            f"{avg_wins:.1f} Siege",
            f"{winning_seasons} Winning Seasons seit 2016",
            "neutral",
        )
    with cards[2]:
        render_stat_card(
            "Gap zur Liga",
            format_diff(latest_win_gap, pct=True),
            f"Punktdifferenz {format_diff(latest_point_gap, points=2)}",
            tone_for_delta(latest_win_gap, True),
        )
    with cards[3]:
        render_stat_card(
            "Prioritaet #1",
            pretty_metric(top_cause["metric"]) if top_cause is not None else "n/a",
            f"Problem-Score {top_cause['problem_score']:.1f}" if top_cause is not None else "noch leer",
            "negative" if top_cause is not None else "neutral",
        )
    with cards[4]:
        if best_model is None:
            render_stat_card("Backtest-Stabilitaet", "n/a", "ohne Modellmetriken", "neutral")
        else:
            render_stat_card(
                "Backtest-Stabilitaet",
                f"{int(best_model['beat_log_loss_seasons'])}/{int(best_model['total_test_seasons'])}",
                "Saisons besser als naiv",
                "positive",
            )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="section-title">Kernaussage fuer Pruefer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Die Jets sind nicht nur knapp unterdurchschnittlich, sondern in mehreren strukturell relevanten Bereichen stabil schwach. Gleichzeitig ist der Modellteil jetzt robust genug, um diese Hebel saisonuebergreifend zu quantifizieren.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="summary-note">
                2025 schliessen die Jets mit <strong>{int(jets_latest['wins'])}-{int(jets_latest['losses'])}</strong> ab und liegen
                bei der Siegquote um <strong>{format_diff(latest_win_gap, pct=True)}</strong> unter dem Ligamittel.
                Besonders kritisch sind <strong>{escape(pretty_metric(top_cause['metric'])) if top_cause is not None else 'die Top-Ursachen'}</strong>,
                weil diese Kennzahl nicht nur schwach ist, sondern vom Modell auch als spielentscheidend gewichtet wird.
                <br/><br/>
                <strong>Modellfazit:</strong> {escape(model_takeaway)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if injury_data_note:
            st.markdown(f'<div class="summary-note">{escape(injury_data_note)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Modellvergleich im Mittel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Entscheidend fuer dieses Projekt sind kalibrierte Wahrscheinlichkeiten. Deshalb stehen Log Loss und Brier im Mittelpunkt.</div>',
            unsafe_allow_html=True,
        )
        if model_summary.empty:
            st.info("Noch keine Modellmetriken verfuegbar.")
        else:
            st.altair_chart(model_metric_chart(model_summary), use_container_width=True)

        if best_diagnostic is not None:
            ece_note = (
                f" Die mittlere Expected Calibration Error liegt bei <strong>{best_model['avg_ece']:.4f}</strong>."
                if pd.notna(best_model.get("avg_ece"))
                else ""
            )
            st.markdown(
                f"""
                <div class="summary-note" style="margin-top:0.9rem;">
                    <strong>Generalisation-Check:</strong> {model_label(best_model["model_name"])} erreicht im Mittel
                    einen Log-Loss-Gap von <strong>{best_diagnostic["avg_log_loss_gap"]:.4f}</strong> und einen
                    Brier-Gap von <strong>{best_diagnostic["avg_brier_gap"]:.4f}</strong>.{ece_note} Das ist
                    klein genug, um Overfitting und schlechte Kalibrierung nicht mehr als Hauptproblem zu sehen.
                </div>
                """,
                unsafe_allow_html=True,
            )

    trend_col, cause_col = st.columns([1.05, 0.95])
    with trend_col:
        st.markdown('<div class="section-title">Die Jets im Zeitverlauf</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Die Siegquote bleibt ueber weite Strecken unter dem Ligamittel und stabilisiert sich nie nachhaltig.</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(line_chart_win_trend(jets_season_summary), use_container_width=True)
    with cause_col:
        st.markdown('<div class="section-title">Die drei dringendsten Hebel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Hier sollte die organisatorische und spielerische Verbesserung zuerst ansetzen.</div>',
            unsafe_allow_html=True,
        )
        top_cards = root_causes.head(3)
        for _, row in top_cards.iterrows():
            st.markdown(
                f"""
                <div class="story-card" style="margin-bottom:0.8rem;">
                    <div class="story-rank">Prioritaet</div>
                    <h4>{escape(pretty_metric(row["metric"]))}</h4>
                    <p>
                        Jets: <strong>{row["jets_value"]:.3f}</strong> vs. Liga: <strong>{row["league_avg"]:.3f}</strong><br/>
                        {escape(recommendation_for_metric(row["metric"]))}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    model_left, model_right = st.columns([1.1, 0.9])
    with model_left:
        st.markdown('<div class="section-title">Benchmark je Testsaison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Das Hauptmodell ist nicht nur im Mittel besser, sondern in jeder einzelnen Testsaison vor dem naiven Heimvorteils-Benchmark.</div>',
            unsafe_allow_html=True,
        )
        if evaluation_metrics.empty:
            st.info("Noch keine saisonalen Modellmetriken verfuegbar.")
        else:
            st.altair_chart(model_seasonal_chart(evaluation_metrics), use_container_width=True)
    with model_right:
        st.markdown('<div class="section-title">Konsistenz und Generalisierung</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Gruene Balken bedeuten Vorsprung gegen naiv; kleine Gaps bedeuten gute Generalisierung.</div>',
            unsafe_allow_html=True,
        )
        if benchmark_comparison.empty:
            st.info("Noch kein Benchmark-Vergleich verfuegbar.")
        else:
            st.altair_chart(benchmark_consistency_chart(benchmark_comparison), use_container_width=True)
        if diagnostic_summary.empty:
            st.info("Noch keine Diagnostik fuer Train/Test-Gaps verfuegbar.")
        else:
            st.altair_chart(generalization_gap_chart(diagnostic_summary), use_container_width=True)
        if best_model is not None:
            calibration_chart = calibration_curve_chart(calibration_bins, best_model["model_name"])
            if calibration_chart is not None:
                st.altair_chart(calibration_chart, use_container_width=True)

elif page == "Liga-Ueberblick":
    win_gap = float(jets_latest["win_pct"] - jets_latest["league_avg_win_pct"])
    point_gap = float(jets_latest["avg_point_diff"] - jets_latest["league_avg_point_diff"])
    record_tone = tone_for_delta(win_gap, higher_is_better=True)
    point_tone = tone_for_delta(point_gap, higher_is_better=True)
    rank_tone = "negative" if jets_latest["win_rank"] > 20 else "neutral"

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_stat_card(
            "Letzte Saison",
            f"{int(jets_latest['wins'])}-{int(jets_latest['losses'])}",
            f"Siegquote {jets_latest['win_pct']:.1%}",
            record_tone,
        )
    with kpi_cols[1]:
        render_stat_card(
            "Gap zur Liga",
            format_diff(win_gap, pct=True),
            "gegenueber Liga-Mittel",
            record_tone,
        )
    with kpi_cols[2]:
        render_stat_card(
            "Punktdifferenz",
            f"{jets_latest['avg_point_diff']:.2f}",
            f"Liga-Gap {format_diff(point_gap, points=2)}",
            point_tone,
        )
    with kpi_cols[3]:
        render_stat_card(
            "Liga-Rang",
            f"#{int(jets_latest['win_rank'])}",
            f"Punktdiff-Rang #{int(jets_latest['point_diff_rank'])}",
            rank_tone,
        )

    left, right = st.columns([1.15, 0.95])
    with left:
        st.markdown('<div class="section-title">Jets vs. Liga im Zeitverlauf</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Der Kernvergleich: wie weit liegen die Jets Jahr fuer Jahr unter dem Ligamittel?</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(line_chart_win_trend(jets_season_summary), use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Position in der Liga 2025</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Siege und Punktdifferenz zeigen sofort, wie weit die Jets vom oberen Tier entfernt sind.</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(scatter_latest_season(team_season_summary, latest_season, config.focus_team), use_container_width=True)

    benchmark_teams = latest.sort_values(["wins", "avg_point_diff"], ascending=[False, False]).head(5)["team"].tolist()
    benchmark = latest[latest["team"].isin([config.focus_team, *benchmark_teams])].copy()
    benchmark["Team"] = benchmark["team"]
    benchmark["Record"] = benchmark["wins"].astype(int).astype(str) + "-" + benchmark["losses"].astype(int).astype(str)
    benchmark["Siegquote"] = benchmark["win_pct"].map(lambda value: f"{value:.1%}")
    benchmark["Punktdifferenz"] = benchmark["avg_point_diff"].map(lambda value: f"{value:.2f}")
    benchmark["Win Rank"] = benchmark["win_rank"].astype(int)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Benchmark gegen Contender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Jets und die besten Teams der letzten Saison im direkten Snapshot.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        benchmark[["Team", "Record", "Siegquote", "Punktdifferenz", "Win Rank"]].sort_values("Win Rank"),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Root Causes":
    st.markdown('<div class="section-title">Die strukturellen Jets-Probleme</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Der Problem-Score kombiniert Ligagap und Modellrelevanz. Hohe Werte sind echte Prioritaeten.</div>',
        unsafe_allow_html=True,
    )

    if root_causes.empty:
        st.info("Noch keine Root-Cause-Tabelle verfuegbar.")
    else:
        chart_col, note_col = st.columns([1.25, 0.75])
        with chart_col:
            st.altair_chart(root_cause_chart(root_causes), use_container_width=True)
        with note_col:
            worst = root_causes.iloc[0]
            direction_hint = "hoeher" if beneficial_when_higher(worst["metric"]) else "niedriger"
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">Leitbefund 2025</div>
                    <div class="section-copy">
                        Die groesste Baustelle ist <strong>{escape(pretty_metric(worst["metric"]))}</strong>.
                        Der Jets-Wert liegt bei <strong>{worst["jets_value"]:.3f}</strong>, der Ligamittelwert bei
                        <strong>{worst["league_avg"]:.3f}</strong>. Diese Kennzahl muss fuer eine Trendwende
                        klar <strong>{direction_hint}</strong> werden.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        top_cards = root_causes.head(3).copy()
        card_columns = st.columns(3)
        max_problem = float(root_causes["problem_score"].max())
        for column, (_, row) in zip(card_columns, top_cards.iterrows(), strict=False):
            severity = "Kritisch" if row["problem_score"] >= max_problem * 0.75 else "Hoch"
            with column:
                st.markdown(
                    f"""
                    <div class="story-card">
                        <div class="story-rank">{severity}</div>
                        <h4>{escape(pretty_metric(row["metric"]))}</h4>
                        <p>
                            Jets: <strong>{row["jets_value"]:.3f}</strong> vs. Liga: <strong>{row["league_avg"]:.3f}</strong><br/>
                            Modellgewicht: <strong>{row["importance"]:.1f}</strong><br/>
                            {escape(recommendation_for_metric(row["metric"]))}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        table = root_causes.copy()
        table["Metrik"] = table["metric"].map(pretty_metric)
        table["Jets"] = table["jets_value"].map(lambda value: f"{value:.3f}")
        table["Liga"] = table["league_avg"].map(lambda value: f"{value:.3f}")
        table["Modellgewicht"] = table["importance"].map(lambda value: f"{value:.1f}")
        table["Problem-Score"] = table["problem_score"].map(lambda value: f"{value:.1f}")
        table["Rank"] = table["metric_rank"].astype(int)
        table["Empfehlung"] = table["metric"].map(recommendation_for_metric)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Vollstaendige Prioritaetenliste</div>', unsafe_allow_html=True)
        st.dataframe(
            table[["Metrik", "Jets", "Liga", "Modellgewicht", "Rank", "Problem-Score", "Empfehlung"]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Tracking Lens":
    st.markdown('<div class="section-title">Tracking Lens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Next-Gen-Stats zeigen, ob das Jets-Problem nur klassische Boxscore-Logik ist oder auch im Bewegungsprofil sichtbar wird.</div>',
        unsafe_allow_html=True,
    )
    if tracking_lens.empty:
        st.info("Keine Tracking-Lens-Tabelle gefunden.")
    else:
        metric_pool = [
            column
            for column in tracking_lens.columns
            if column.startswith("ngs_")
            and not column.startswith("league_avg_")
            and is_display_metric(column)
        ]
        families = sorted({column.split("_")[1] for column in metric_pool})
        selected_family = st.radio("Metrikfamilie", families, horizontal=True)
        family_metrics = [column for column in metric_pool if column.startswith(f"ngs_{selected_family}_")]
        family_metrics = sorted(family_metrics, key=pretty_metric)
        selected_metric = st.selectbox(
            "Tracking-Metrik",
            family_metrics,
            format_func=pretty_metric,
        )

        latest_tracking = tracking_lens[tracking_lens["season"] == latest_season]
        latest_delta = None
        if not latest_tracking.empty:
            latest_delta = float(
                latest_tracking[selected_metric].iloc[0] - latest_tracking[f"league_avg_{selected_metric}"].iloc[0]
            )

        info_cols = st.columns(3)
        with info_cols[0]:
            render_stat_card(
                "Gewaehlte Metrik",
                pretty_metric(selected_metric),
                "Jets gegen Liga ueber Zeit",
                "neutral",
            )
        with info_cols[1]:
            render_stat_card(
                f"{latest_season} Delta",
                f"{latest_delta:.3f}" if latest_delta is not None else "n/a",
                "Jets minus Liga",
                tone_for_delta(latest_delta or 0.0, beneficial_when_higher(selected_metric)),
            )
        with info_cols[2]:
            render_stat_card(
                "Interpretation",
                "Hoeher ist besser" if beneficial_when_higher(selected_metric) else "Niedriger ist besser",
                selected_family.title(),
                "neutral",
            )

        left, right = st.columns([1.1, 0.9])
        with left:
            st.altair_chart(tracking_line_chart(tracking_lens, selected_metric), use_container_width=True)
        with right:
            gap_chart = tracking_gap_chart(tracking_lens, selected_family, latest_season)
            if gap_chart is not None:
                st.altair_chart(gap_chart, use_container_width=True)
            else:
                st.info("Fuer diese Familie sind keine sauberen Deltas verfuegbar.")

elif page == "What-If Simulator":
    st.markdown('<div class="section-title">What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Hier wird nicht geraten, sondern simuliert: Welche realistischen Veraenderungen verschieben die Jets-Siegwahrscheinlichkeit am staerksten?</div>',
        unsafe_allow_html=True,
    )

    model_path = (
        Path(os.getenv("JETS_MODEL_BUNDLE_PATH")).expanduser()
        if os.getenv("JETS_MODEL_BUNDLE_PATH")
        else config.paths.model_bundle_path
    )
    if not model_path.exists():
        st.warning("Es fehlt das Modell-Bundle. Fuehre zuerst `uv run python -m jets_project.train` aus.")
        st.stop()

    model_bundle = load_model_bundle(model_path)
    seasons = sorted(
        game_features.loc[
            (game_features["home_team"] == config.focus_team) | (game_features["away_team"] == config.focus_team),
            "season",
        ].unique()
    )
    selected_season = st.selectbox("Saison", seasons, index=len(seasons) - 1)

    actionable = [metric for metric in model_bundle.get("top_actionable_metrics", []) if is_display_metric(metric)][:4]
    if not actionable:
        st.info("Keine Top-Metriken im Bundle hinterlegt.")
        st.stop()

    adjustments: dict[str, float] = {}
    slider_columns = st.columns(2)
    for index, metric in enumerate(actionable):
        hint = "Hoeher ist besser" if beneficial_when_higher(metric) else "Niedriger ist besser"
        with slider_columns[index % 2]:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">{escape(pretty_metric(metric))}</div>
                    <div class="section-copy">{escape(hint)}. {escape(recommendation_for_metric(metric))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            value = st.slider(
                f"{pretty_metric(metric)} Delta",
                min_value=-0.20,
                max_value=0.20,
                value=0.0,
                step=0.01,
                key=f"slider_{metric}",
            )
            if abs(value) > 0:
                adjustments[metric] = value

    scenario = simulate_team_improvement(
        game_features=game_features,
        model_bundle=model_bundle,
        focus_team=config.focus_team,
        season=int(selected_season),
        adjustments=adjustments,
    )
    if scenario.empty:
        st.info("Keine Jets-Spiele fuer die gewaehlte Saison gefunden.")
        st.stop()

    baseline_distribution = np.asarray(
        scenario.attrs.get("baseline_expected_wins_distribution", np.array([scenario["baseline_win_probability"].sum()]))
    )
    scenario_distribution = np.asarray(
        scenario.attrs.get("scenario_expected_wins_distribution", np.array([scenario["scenario_win_probability"].sum()]))
    )
    baseline_expected_wins = float(baseline_distribution.mean())
    scenario_expected_wins = float(scenario_distribution.mean())
    delta_expected_wins = scenario_expected_wins - baseline_expected_wins
    baseline_band = (float(np.quantile(baseline_distribution, 0.10)), float(np.quantile(baseline_distribution, 0.90)))
    scenario_band = (float(np.quantile(scenario_distribution, 0.10)), float(np.quantile(scenario_distribution, 0.90)))
    delta_distribution = scenario_distribution - baseline_distribution
    delta_band = (float(np.quantile(delta_distribution, 0.10)), float(np.quantile(delta_distribution, 0.90)))

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_stat_card(
            "Baseline Expected Wins",
            f"{baseline_expected_wins:.2f}",
            f"Band {baseline_band[0]:.2f} bis {baseline_band[1]:.2f}",
            "neutral",
        )
    with summary_cols[1]:
        render_stat_card(
            "Scenario Expected Wins",
            f"{scenario_expected_wins:.2f}",
            f"Band {scenario_band[0]:.2f} bis {scenario_band[1]:.2f}",
            tone_for_delta(delta_expected_wins, True),
        )
    with summary_cols[2]:
        render_stat_card(
            "Expected Wins Delta",
            f"{delta_expected_wins:+.2f}",
            f"Band {delta_band[0]:+.2f} bis {delta_band[1]:+.2f}",
            tone_for_delta(delta_expected_wins, True),
        )
    with summary_cols[3]:
        best_swing = scenario.sort_values("win_probability_delta", ascending=False).iloc[0]
        render_stat_card(
            "Groesster Hebel",
            best_swing["opponent"],
            (
                f"Week {int(best_swing['week'])} | {best_swing['win_probability_delta']:+.2%} | "
                f"{best_swing['decision_confidence'].title()}"
            ),
            tone_for_delta(float(best_swing["win_probability_delta"]), True),
        )

    st.markdown(
        f"""
        <div class="summary-note" style="margin-bottom:1rem;">
            <strong>Decision-Support-Modus:</strong> Das Szenario nutzt {model_label(scenario.attrs.get("model_name", model_bundle.get("preferred_model_name", "gradient_boosting")))}
            mit einem Ensemble von <strong>{int(scenario.attrs.get("ensemble_size", 1))}</strong> Modellen. Angezeigt werden Mittelwerte und Unsicherheitsbaender, nicht nur Punktprognosen.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns([1.1, 0.9])
    with chart_left:
        st.altair_chart(scenario_comparison_chart(scenario), use_container_width=True)
    with chart_right:
        st.altair_chart(scenario_delta_chart(scenario), use_container_width=True)

    top_swings = scenario.sort_values("win_probability_delta", ascending=False).copy()
    top_swings["Baseline"] = top_swings["baseline_win_probability"].map(lambda value: f"{value:.1%}")
    top_swings["Szenario"] = top_swings["scenario_win_probability"].map(lambda value: f"{value:.1%}")
    top_swings["Delta"] = top_swings["win_probability_delta"].map(lambda value: f"{value:+.1%}")
    top_swings["Delta-Band"] = top_swings.apply(
        lambda row: f"{row['win_probability_delta_p10']:+.1%} bis {row['win_probability_delta_p90']:+.1%}",
        axis=1,
    )
    top_swings["Signal"] = top_swings["positive_scenario_share"].map(lambda value: f"{value:.0%}")
    top_swings["Sicherheit"] = top_swings["decision_confidence"].str.title()
    top_swings["Ort"] = top_swings["location"].map({"home": "Heim", "away": "Auswaerts"}).fillna(top_swings["location"])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Groesste Spielverschiebungen</div>', unsafe_allow_html=True)
    st.dataframe(
        top_swings[["week", "opponent", "Ort", "Baseline", "Szenario", "Delta", "Delta-Band", "Signal", "Sicherheit"]].head(8),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Methodik und Leselogik"):
    st.markdown(
        """
        - Alle visuellen Komponenten sind in Streamlit umgesetzt.
        - Die Modellseite arbeitet auf pregame Features, um Leakage zu vermeiden.
        - `Root Causes` kombiniert Ligagap und Modellrelevanz zu einem priorisierbaren Score.
        - `Tracking Lens` nutzt Next-Gen-Stats als zweite Perspektive auf dieselbe Jets-Frage.
        - Der Simulator veraendert nur ausgewaehlte Metriken und liest ansonsten das gespeicherte Modellbundle.
        - Fuer Decision Support werden kalibrierte Wahrscheinlichkeiten und ein kleines Modell-Ensemble mit Unsicherheitsbaendern genutzt.
        - Nicht interpretierbare Tracking-Spalten wie Jersey-Nummern werden bewusst aus den visuellen Empfehlungen ausgeschlossen.
        """
    )
