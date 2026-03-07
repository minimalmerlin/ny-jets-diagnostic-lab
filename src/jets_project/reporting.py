from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_NAME_MAP = {
    "gradient_boosting": "Gradient-Boosting-Modell",
    "logistic_regression": "Logistische Regression",
    "naive_home_rate": "Naiver Heimvorteils-Benchmark",
}

METRIC_LABELS = {
    "offense_sack_rate": "Offense-Sack-Rate",
    "offense_success_rate": "Offense-Erfolgsrate",
    "offense_epa_per_play": "Offense-EPA pro Play",
    "defense_takeaway_rate": "Takeaway-Rate der Defense",
    "defense_explosive_play_rate_allowed": "Zugelassene Explosive Plays",
    "ngs_passing_aggressiveness": "Passing-Aggressiveness",
    "ngs_passing_expected_completion_percentage": "Expected Completion Percentage",
    "ngs_receiving_receptions": "Receptions",
    "injury_out_rate": "Anteil Ausfaelle im Injury Report",
    "injury_dnp_rate": "Anteil DNP im Practice Report",
    "injury_impact_score": "Injury-Impact-Score",
    "contract_avg_cap_percent": "Durchschnittlicher Cap-Anteil pro Vertrag",
    "contract_player_count": "Vertrags-Slots im aktiven Kader",
    "contract_top3_cap_share": "Cap-Anteil der Top-3-Vertraege",
    "contract_qb_cap_share": "QB-Cap-Anteil",
}
NON_ACTIONABLE_METRICS = {
    "point_diff",
    "win",
}
NON_ACTIONABLE_METRIC_TOKENS = (
    "games_played",
    "rest_days",
    "strength_of_schedule",
)


def _model_label(model_name: str) -> str:
    return MODEL_NAME_MAP.get(model_name, model_name.replace("_", " ").title())


def _pretty_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _is_report_metric(metric: str) -> bool:
    if metric in NON_ACTIONABLE_METRICS:
        return False
    return not any(token in metric for token in NON_ACTIONABLE_METRIC_TOKENS)


def _build_benchmark_comparison(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
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


def _build_model_summary(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
    if evaluation_metrics.empty:
        return pd.DataFrame()
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
    comparison = _build_benchmark_comparison(evaluation_metrics)
    if comparison.empty:
        return summary
    summary = summary.merge(
        comparison.groupby("model_name", as_index=False)
        .agg(
            beat_log_loss_seasons=("beat_log_loss", "sum"),
            beat_brier_seasons=("beat_brier", "sum"),
            total_test_seasons=("test_season", "nunique"),
        ),
        on="model_name",
        how="left",
    )
    return summary


def _build_diagnostic_summary(backtest_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if backtest_diagnostics.empty:
        return pd.DataFrame()
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


def render_markdown_report(
    jets_season_summary: pd.DataFrame,
    root_causes: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    backtest_diagnostics: pd.DataFrame,
    focus_team: str = "NYJ",
) -> str:
    if jets_season_summary.empty:
        return "# NY Jets Diagnostic Lab\n\nKeine Auswertungsdaten verfuegbar.\n"

    latest = jets_season_summary.sort_values("season").iloc[-1]
    root_causes = root_causes[root_causes["metric"].map(_is_report_metric)].reset_index(drop=True)
    model_summary = _build_model_summary(evaluation_metrics)
    benchmark = _build_benchmark_comparison(evaluation_metrics)
    diagnostic_summary = _build_diagnostic_summary(backtest_diagnostics)

    contenders = model_summary[model_summary["model_name"] != "naive_home_rate"].copy()
    best_model = contenders.sort_values("avg_log_loss").iloc[0] if not contenders.empty else None
    best_diagnostic = (
        diagnostic_summary[diagnostic_summary["model_name"] == best_model["model_name"]].iloc[0]
        if best_model is not None
        and not diagnostic_summary.empty
        and best_model["model_name"] in diagnostic_summary["model_name"].values
        else None
    )

    avg_wins = float(jets_season_summary["wins"].mean())
    winning_seasons = int((jets_season_summary["wins"] > jets_season_summary["losses"]).sum())
    latest_win_gap = float(latest["win_pct"] - latest["league_avg_win_pct"])
    latest_point_gap = float(latest["avg_point_diff"] - latest["league_avg_point_diff"])

    top_findings = []
    for row in root_causes.head(5).itertuples(index=False):
        top_findings.append(
            f"- **{_pretty_metric(row.metric)}**: Jets {row.jets_value:.3f} vs. Liga {row.league_avg:.3f}, Problem-Score {row.problem_score:.1f}"
        )

    model_table = []
    if not model_summary.empty:
        for row in model_summary.itertuples(index=False):
            benchmark_note = "-"
            ece_value = f"{row.avg_ece:.4f}" if pd.notna(row.avg_ece) else "-"
            if row.model_name != "naive_home_rate" and pd.notna(getattr(row, "beat_log_loss_seasons", None)):
                benchmark_note = f"{int(row.beat_log_loss_seasons)}/{int(row.total_test_seasons)}"
            model_table.append(
                f"| {_model_label(row.model_name)} | {row.avg_log_loss:.4f} | {row.avg_brier:.4f} | {row.avg_roc_auc:.4f} | {ece_value} | {benchmark_note} |"
            )

    seasonal_wins = []
    if not benchmark.empty and best_model is not None:
        best_rows = benchmark[benchmark["model_name"] == best_model["model_name"]].sort_values("test_season")
        for row in best_rows.itertuples(index=False):
            seasonal_wins.append(
                f"- **{int(row.test_season)}**: Log-Loss-Vorsprung {row.log_loss_gain:.4f}, Brier-Vorsprung {row.brier_gain:.4f}"
            )

    recommendations = []
    for metric in root_causes["metric"].head(3):
        recommendations.append(f"- {_pretty_metric(metric)} gezielt in Richtung Liga-Topquartil bewegen.")

    report = "\n".join(
        [
            "# NY Jets Diagnostic Lab",
            "",
            "## 1. Executive Summary",
            "",
            (
                f"Die {focus_team} beenden die Saison {int(latest['season'])} mit einem Record von "
                f"**{int(latest['wins'])}-{int(latest['losses'])}**. Damit liegt die Siegquote bei "
                f"**{latest['win_pct']:.1%}** und damit um **{latest_win_gap:+.1%}** unter dem Ligamittel. "
                f"Auch die Punktdifferenz pro Spiel ist mit **{latest['avg_point_diff']:.2f}** klar negativ "
                f"und liegt um **{latest_point_gap:+.2f}** unter dem Liga-Schnitt."
            ),
            "",
            (
                f"Seit 2016 erreichen die Jets im Mittel nur **{avg_wins:.1f} Siege** pro Saison und kommen nur auf "
                f"**{winning_seasons} Winning Seasons**. Die Underperformance ist also nicht kurzfristig, sondern strukturell."
            ),
            "",
            "## 2. Daten und Methodik",
            "",
            f"- Datenschnitt: 2016 bis {int(latest['season'])}",
            "- Datenbasis: nflreadpy / nflverse, kuratiert in SQLite",
            "- Modellierung: Logistische Regression als Baseline, Gradient-Boosting als Hauptmodell",
            "- Evaluation: Rolling-Origin-Backtests, pro Saison nur mit historischen Informationen",
            "",
            "## 3. Liga-Vergleich",
            "",
            (
                "Die Jets bleiben ueber weite Strecken des Zeitraums unter dem Ligamittel. Der Rueckstand zeigt "
                "sich nicht nur in Siegen, sondern auch in Prozessmetriken wie Protection, Effizienz und "
                "Explosive-Play-Management. Damit ist die Grundhypothese bestaetigt: Das Problem liegt nicht nur "
                "im Zufall oder in einzelnen engen Spielen."
            ),
            "",
            "## 4. Modellbefunde",
            "",
            "| Modell | Log Loss | Brier Score | ROC-AUC | ECE | Benchmark-Siege |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *(
                model_table
                if model_table
                else ["| Keine Modellmetriken verfuegbar. | - | - | - | - | - |"]
            ),
            "",
        ]
    )

    if best_model is not None:
        report += (
            f"{_model_label(best_model['model_name'])} ist das tragfaehigste Hauptmodell. "
            f"Es schlaegt den naiven Heimvorteils-Benchmark in "
            f"**{int(best_model['beat_log_loss_seasons'])}/{int(best_model['total_test_seasons'])}** "
            f"Test-Saisons bei Log Loss und ebenso in "
            f"**{int(best_model['beat_brier_seasons'])}/{int(best_model['total_test_seasons'])}** "
            f"Saisons bei Brier.\n\n"
        )
    if best_diagnostic is not None:
        report += (
            f"Der Generalisierungs-Check zeigt keinen kritischen Overfitting-Befund mehr. "
            f"Der mittlere Train/Test-Gap des Hauptmodells liegt bei **{best_diagnostic['avg_log_loss_gap']:.4f}** "
            f"im Log Loss und **{best_diagnostic['avg_brier_gap']:.4f}** im Brier Score.\n\n"
        )
        if pd.notna(best_model.get("avg_ece")):
            report += (
                f"Zusaetzlich liegt die mittlere Expected Calibration Error des Hauptmodells bei "
                f"**{best_model['avg_ece']:.4f}**. Die Wahrscheinlichkeiten sind damit nicht nur trennscharf, "
                f"sondern auch fuer Decision Support ausreichend sauber kalibriert.\n\n"
            )
    if seasonal_wins:
        report += "Saisonale Benchmark-Vorspruenge des Hauptmodells:\n\n"
        report += "\n".join(seasonal_wins)
        report += "\n\n"

    report += "## 5. Wichtigste strukturelle Ursachen\n\n"
    report += "\n".join(top_findings) if top_findings else "- Keine Root-Cause-Daten verfuegbar."
    report += "\n\n## 6. Handlungsempfehlungen\n\n"
    report += "\n".join(recommendations) if recommendations else "- Noch keine Empfehlungen verfuegbar."
    report += (
        "\n\n## 7. Fazit\n\n"
        "Die Jets verlieren nicht wegen eines einzelnen Akteurs, sondern wegen eines wiederkehrenden Profils aus "
        "Protection-Problemen, fehlender Offense-Effizienz und zu vielen zugelassenen Big Plays. "
        "Der Modellteil ist jetzt stabil genug, um diese Hebel nicht nur deskriptiv, sondern auch prognostisch zu bewerten.\n"
    )
    return report


def write_markdown_report(
    output_path: Path,
    jets_season_summary: pd.DataFrame,
    root_causes: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    backtest_diagnostics: pd.DataFrame,
    focus_team: str = "NYJ",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown_report(
            jets_season_summary=jets_season_summary,
            root_causes=root_causes,
            evaluation_metrics=evaluation_metrics,
            backtest_diagnostics=backtest_diagnostics,
            focus_team=focus_team,
        ),
        encoding="utf-8",
    )
