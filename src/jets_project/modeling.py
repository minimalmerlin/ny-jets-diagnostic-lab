from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .optional import import_optional_dependency

EXCLUDED_METRIC_TOKENS = (
    "player_jersey_number",
    "weight_sum",
)
EXCLUDED_SELECTION_TOKENS = EXCLUDED_METRIC_TOKENS + (
    "ngs_rushing_expected_rush_yards",
    "ngs_rushing_rush_yards_over_expected",
    "ngs_rushing_rush_pct_over_expected",
    "qb_stability_score",
)
ROLLING4_PRIORITY_METRICS = {
    "offense_epa_per_play",
    "offense_success_rate",
    "offense_turnover_rate",
    "offense_sack_rate",
    "offense_explosive_play_rate",
    "defense_epa_per_play_allowed",
    "defense_success_rate_allowed",
    "defense_takeaway_rate",
    "defense_sack_rate",
    "defense_explosive_play_rate_allowed",
    "point_diff",
    "qb_primary_changed",
}
NON_ACTIONABLE_METRICS = {
    "defense_plays_faced",
    "games_played",
    "offensive_plays",
    "point_diff",
    "rest_days",
    "strength_of_schedule",
    "win",
}
NON_ACTIONABLE_METRIC_TOKENS = (
    "games_played",
    "instability",
    "opponent_",
    "rest_days",
    "strength_of_schedule",
    "tenure",
)
MIN_FEATURE_COVERAGE = 0.93
DECISION_SUPPORT_ENSEMBLE_SEEDS = (7, 17, 27, 37, 47)


class SupportsPredictProba(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(slots=True)
class IdentityCalibrator:
    calibrator_type: str = "identity"

    def fit(self, probabilities: np.ndarray, target: pd.Series) -> IdentityCalibrator:
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


@dataclass(slots=True)
class PlattCalibrator:
    calibrator_type: str = "platt"
    model: LogisticRegression | None = None
    fitted: bool = False

    def fit(self, probabilities: np.ndarray, target: pd.Series) -> PlattCalibrator:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        target = pd.Series(target).astype(int)
        if (
            len(probabilities) < 80
            or target.nunique() < 2
            or pd.Series(probabilities).round(6).nunique() < 5
        ):
            self.model = None
            self.fitted = False
            return self
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        self.model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        self.model.fit(logits, target)
        self.fitted = True
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if not self.fitted or self.model is None:
            return probabilities
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


@dataclass(slots=True)
class CalibratedModel:
    base_model: Pipeline
    calibrator: IdentityCalibrator | PlattCalibrator
    model_name: str

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw_probabilities = self.base_model.predict_proba(features)[:, 1]
        calibrated_probabilities = self.calibrator.predict(raw_probabilities)
        return np.column_stack([1 - calibrated_probabilities, calibrated_probabilities])


@dataclass(slots=True)
class EnsembleModel:
    members: list[CalibratedModel]
    model_name: str

    def predict_member_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if not self.members:
            return np.empty((0, len(features)))
        return np.vstack([member.predict_proba(features)[:, 1] for member in self.members])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        member_probabilities = self.predict_member_probabilities(features)
        if member_probabilities.size == 0:
            return np.zeros((len(features), 2))
        mean_probabilities = member_probabilities.mean(axis=0)
        return np.column_stack([1 - mean_probabilities, mean_probabilities])


@dataclass(slots=True)
class TrainingResults:
    backtest_predictions: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    feature_importance: pd.DataFrame
    final_models: dict[str, SupportsPredictProba]
    calibration_summary: pd.DataFrame
    feature_ranges: dict[str, dict[str, float]]


def _make_logistic_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, C=0.1, solver="lbfgs")),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def _make_gradient_pipeline(random_state: int) -> Pipeline:
    try:
        lightgbm = import_optional_dependency("lightgbm", "uv sync")
        model = lightgbm.LGBMClassifier(
            force_col_wise=True,
            colsample_bytree=0.5,
            learning_rate=0.02,
            max_depth=2,
            min_child_samples=100,
            min_split_gain=0.1,
            n_estimators=160,
            num_leaves=5,
            random_state=random_state,
            reg_alpha=1.0,
            reg_lambda=5.0,
            subsample=0.7,
            verbosity=-1,
        )
    except RuntimeError:
        model = HistGradientBoostingClassifier(
            l2_regularization=2.0,
            learning_rate=0.02,
            max_depth=3,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=60,
            random_state=random_state,
        )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def build_model_factories(random_state: int = 7):
    return {
        "logistic_regression": _make_logistic_pipeline,
        "gradient_boosting": lambda: _make_gradient_pipeline(random_state=random_state),
    }


def build_model_factory(model_name: str, random_state: int = 7):
    factories = build_model_factories(random_state=random_state)
    if model_name not in factories:
        raise KeyError(f"Unbekanntes Modell: {model_name}")
    return factories[model_name]


def expected_calibration_error(
    y_true: pd.Series,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    if len(probabilities) == 0:
        return np.nan
    frame = pd.DataFrame({"target": pd.Series(y_true).astype(float), "probability": probabilities})
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    total = len(frame)
    ece = 0.0
    for _, group in frame.groupby("bin", observed=False):
        if group.empty:
            continue
        ece += abs(group["target"].mean() - group["probability"].mean()) * (len(group) / total)
    return float(ece)


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    if len(probabilities) == 0:
        return {"log_loss": np.nan, "brier_score": np.nan, "roc_auc": np.nan, "ece": np.nan}
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    metrics = {
        "log_loss": log_loss(y_true, clipped, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, clipped),
        "roc_auc": roc_auc_score(y_true, clipped) if y_true.nunique() > 1 else np.nan,
        "ece": expected_calibration_error(y_true, clipped),
    }
    return metrics


def _predict_proba(model: SupportsPredictProba, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def _fit_probability_calibrator(
    probabilities: np.ndarray,
    target: pd.Series,
) -> IdentityCalibrator | PlattCalibrator:
    calibrator = PlattCalibrator().fit(probabilities, target)
    if getattr(calibrator, "fitted", False):
        return calibrator
    return IdentityCalibrator()


def _build_calibration_frame(
    model_frame: pd.DataFrame,
    feature_cols: list[str],
    factory,
    min_train_seasons: int,
) -> pd.DataFrame:
    seasons = sorted(model_frame["season"].dropna().unique())
    rows = []
    for season_index, calibration_season in enumerate(seasons):
        if season_index < min_train_seasons:
            continue
        inner_train = model_frame[model_frame["season"].isin(seasons[:season_index])].copy()
        inner_test = model_frame[model_frame["season"] == calibration_season].copy()
        if inner_train.empty or inner_test.empty or inner_train["home_win"].nunique() < 2:
            continue
        model = factory()
        model.fit(inner_train[feature_cols], inner_train["home_win"])
        raw_probabilities = _predict_proba(model, inner_test[feature_cols])
        rows.append(
            pd.DataFrame(
                {
                    "season": int(calibration_season),
                    "raw_probability": raw_probabilities,
                    "actual": inner_test["home_win"].astype(int).to_numpy(),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["season", "raw_probability", "actual"])
    return pd.concat(rows, ignore_index=True)


def _feature_ranges(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    return {
        feature: {
            "q05": float(frame[feature].quantile(0.05)),
            "q95": float(frame[feature].quantile(0.95)),
        }
        for feature in feature_cols
    }


def _feature_importance_for_model(
    model_name: str,
    model: SupportsPredictProba,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    if isinstance(model, EnsembleModel):
        estimator = model.members[0].base_model.named_steps["model"]
        base_model = model.members[0].base_model
    elif isinstance(model, CalibratedModel):
        estimator = model.base_model.named_steps["model"]
        base_model = model.base_model
    else:
        estimator = model.named_steps["model"]
        base_model = model
    if hasattr(estimator, "coef_"):
        return pd.DataFrame(
            {
                "model_name": model_name,
                "feature": features.columns,
                "importance": np.abs(estimator.coef_[0]),
                "direction": np.sign(estimator.coef_[0]),
            }
        )
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "model_name": model_name,
                "feature": features.columns,
                "importance": estimator.feature_importances_,
                "direction": np.nan,
            }
        )
    sample_size = min(len(features), 400)
    sample = features.sample(sample_size, random_state=7)
    sample_target = target.loc[sample.index]
    result = permutation_importance(
        base_model,
        sample,
        sample_target,
        n_repeats=5,
        random_state=7,
        scoring="neg_log_loss",
    )
    return pd.DataFrame(
        {
            "model_name": model_name,
            "feature": features.columns,
            "importance": result.importances_mean,
            "direction": np.nan,
        }
    )


def _base_metric_name(feature_name: str) -> str:
    base = re.sub(r"^pregame_", "", feature_name)
    if re.search(r"_(rolling4|season_avg)_diff$", base):
        return re.sub(r"_(rolling4|season_avg)_diff$", "", base)
    return re.sub(r"_diff$", "", base)


def _is_actionable_base_metric(metric_name: str) -> bool:
    if metric_name in NON_ACTIONABLE_METRICS:
        return False
    if any(token in metric_name for token in [*EXCLUDED_METRIC_TOKENS, *NON_ACTIONABLE_METRIC_TOKENS]):
        return False
    return True


def select_training_features(
    model_frame: pd.DataFrame,
    min_coverage: float = MIN_FEATURE_COVERAGE,
) -> list[str]:
    feature_cols = [column for column in model_frame.columns if column.endswith("_diff")]
    if not feature_cols:
        return []

    coverage = model_frame[feature_cols].notna().mean()
    selected = []
    for column in feature_cols:
        metric = _base_metric_name(column)
        if any(token in metric for token in EXCLUDED_SELECTION_TOKENS):
            continue
        if coverage[column] < min_coverage:
            continue
        if column.endswith("_rolling4_diff") and metric not in ROLLING4_PRIORITY_METRICS:
            continue
        if model_frame[column].dropna().nunique() <= 1:
            continue
        selected.append(column)
    return selected


def top_actionable_metrics(importance: pd.DataFrame, top_n: int = 6) -> list[str]:
    if importance.empty:
        return []
    actionable = importance.copy()
    actionable = actionable[actionable["feature"].str.contains(r"_(?:rolling4|season_avg)_diff$", na=False)]
    actionable["base_metric"] = actionable["feature"].map(_base_metric_name)
    actionable = actionable[actionable["base_metric"].map(_is_actionable_base_metric)]
    top = (
        actionable.groupby("base_metric", as_index=False)["importance"]
        .max()
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return top["base_metric"].tolist()


def build_decision_support_ensemble(
    model_name: str,
    model_frame: pd.DataFrame,
    feature_cols: list[str],
    random_states: tuple[int, ...] = DECISION_SUPPORT_ENSEMBLE_SEEDS,
    min_train_seasons: int = 2,
) -> EnsembleModel:
    if model_name == "logistic_regression":
        factory = build_model_factory(model_name, random_state=random_states[0])
        calibration_frame = _build_calibration_frame(model_frame, feature_cols, factory, min_train_seasons)
        calibrator = _fit_probability_calibrator(
            calibration_frame["raw_probability"].to_numpy(),
            calibration_frame["actual"],
        )
        member = CalibratedModel(
            base_model=factory().fit(model_frame[feature_cols], model_frame["home_win"]),
            calibrator=calibrator,
            model_name=model_name,
        )
        return EnsembleModel(members=[member], model_name=model_name)

    members: list[CalibratedModel] = []
    for random_state in random_states:
        factory = build_model_factory(model_name, random_state=random_state)
        calibration_frame = _build_calibration_frame(model_frame, feature_cols, factory, min_train_seasons)
        calibrator = _fit_probability_calibrator(
            calibration_frame["raw_probability"].to_numpy(),
            calibration_frame["actual"],
        )
        member = CalibratedModel(
            base_model=factory().fit(model_frame[feature_cols], model_frame["home_win"]),
            calibrator=calibrator,
            model_name=model_name,
        )
        members.append(member)
    return EnsembleModel(members=members, model_name=model_name)


def run_training_pipeline(
    model_frame: pd.DataFrame,
    feature_cols: list[str],
    min_train_seasons: int = 2,
    random_state: int = 7,
) -> TrainingResults:
    seasons = sorted(model_frame["season"].dropna().unique())
    factories = build_model_factories(random_state=random_state)
    prediction_rows = []
    metric_rows = []
    prediction_columns = [
        "game_id",
        "season",
        "week",
        "model_name",
        "raw_predicted_home_win_probability",
        "predicted_home_win_probability",
    ]
    metric_columns = [
        "model_name",
        "test_season",
        "train_end_season",
        "log_loss",
        "brier_score",
        "roc_auc",
        "ece",
    ]
    diagnostic_columns = [
        "model_name",
        "test_season",
        "train_end_season",
        "train_log_loss",
        "test_log_loss",
        "train_brier_score",
        "test_brier_score",
        "log_loss_gap",
        "brier_score_gap",
        "calibrator_type",
        "calibration_rows",
    ]
    diagnostic_rows = []
    calibration_rows = []

    for season_index, test_season in enumerate(seasons):
        if season_index < min_train_seasons:
            continue
        train_seasons = seasons[:season_index]
        train_frame = model_frame[model_frame["season"].isin(train_seasons)].copy()
        test_frame = model_frame[model_frame["season"] == test_season].copy()
        if train_frame.empty or test_frame.empty or train_frame["home_win"].nunique() < 2:
            continue

        X_train = train_frame[feature_cols]
        y_train = train_frame["home_win"]
        X_test = test_frame[feature_cols]
        y_test = test_frame["home_win"]

        naive_prob = float(y_train.mean())
        naive_preds = np.repeat(naive_prob, len(test_frame))
        naive_metrics = evaluate_probabilities(y_test, naive_preds)
        metric_rows.append(
            {
                "model_name": "naive_home_rate",
                "test_season": int(test_season),
                "train_end_season": int(max(train_seasons)),
                **naive_metrics,
            }
        )
        for game_id, probability in zip(test_frame["game_id"], naive_preds, strict=True):
            prediction_rows.append(
                {
                    "game_id": game_id,
                    "season": int(test_season),
                    "week": int(test_frame.loc[test_frame["game_id"] == game_id, "week"].iloc[0]),
                    "model_name": "naive_home_rate",
                    "raw_predicted_home_win_probability": probability,
                    "predicted_home_win_probability": probability,
                }
            )

        for model_name, factory in factories.items():
            calibration_frame = _build_calibration_frame(
                train_frame,
                feature_cols=feature_cols,
                factory=factory,
                min_train_seasons=min_train_seasons,
            )
            calibrator = _fit_probability_calibrator(
                calibration_frame["raw_probability"].to_numpy(),
                calibration_frame["actual"],
            )
            model = factory()
            model.fit(X_train, y_train)
            calibrated_model = CalibratedModel(base_model=model, calibrator=calibrator, model_name=model_name)
            raw_test_probabilities = model.predict_proba(X_test)[:, 1]
            train_probabilities = _predict_proba(calibrated_model, X_train)
            probabilities = _predict_proba(calibrated_model, X_test)
            train_metrics = evaluate_probabilities(y_train, train_probabilities)
            metrics = evaluate_probabilities(y_test, probabilities)
            metric_rows.append(
                {
                    "model_name": model_name,
                    "test_season": int(test_season),
                    "train_end_season": int(max(train_seasons)),
                    **metrics,
                }
            )
            diagnostic_rows.append(
                {
                    "model_name": model_name,
                    "test_season": int(test_season),
                    "train_end_season": int(max(train_seasons)),
                    "train_log_loss": train_metrics["log_loss"],
                    "test_log_loss": metrics["log_loss"],
                    "train_brier_score": train_metrics["brier_score"],
                    "test_brier_score": metrics["brier_score"],
                    "log_loss_gap": metrics["log_loss"] - train_metrics["log_loss"],
                    "brier_score_gap": metrics["brier_score"] - train_metrics["brier_score"],
                    "calibrator_type": getattr(calibrator, "calibrator_type", "identity"),
                    "calibration_rows": int(len(calibration_frame)),
                }
            )
            calibration_rows.append(
                {
                    "model_name": model_name,
                    "test_season": int(test_season),
                    "train_end_season": int(max(train_seasons)),
                    "calibrator_type": getattr(calibrator, "calibrator_type", "identity"),
                    "calibration_rows": int(len(calibration_frame)),
                    "calibration_seasons": int(calibration_frame["season"].nunique()) if not calibration_frame.empty else 0,
                }
            )
            for row, raw_probability, probability in zip(
                test_frame.itertuples(index=False),
                raw_test_probabilities,
                probabilities,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "game_id": row.game_id,
                        "season": int(row.season),
                        "week": int(row.week),
                        "model_name": model_name,
                        "raw_predicted_home_win_probability": raw_probability,
                        "predicted_home_win_probability": probability,
                    }
                )

    full_X = model_frame[feature_cols]
    full_y = model_frame["home_win"]
    final_models: dict[str, SupportsPredictProba] = {}
    for model_name, factory in factories.items():
        calibration_frame = _build_calibration_frame(
            model_frame,
            feature_cols=feature_cols,
            factory=factory,
            min_train_seasons=min_train_seasons,
        )
        calibrator = _fit_probability_calibrator(
            calibration_frame["raw_probability"].to_numpy(),
            calibration_frame["actual"],
        )
        fitted = factory().fit(full_X, full_y)
        final_models[model_name] = CalibratedModel(
            base_model=fitted,
            calibrator=calibrator,
            model_name=model_name,
        )
    importance_frames = [
        _feature_importance_for_model(name, model, full_X, full_y)
        for name, model in final_models.items()
    ]

    return TrainingResults(
        backtest_predictions=pd.DataFrame(prediction_rows, columns=prediction_columns),
        metrics=pd.DataFrame(metric_rows, columns=metric_columns),
        diagnostics=pd.DataFrame(diagnostic_rows, columns=diagnostic_columns),
        calibration_summary=pd.DataFrame(
            calibration_rows,
            columns=[
                "model_name",
                "test_season",
                "train_end_season",
                "calibrator_type",
                "calibration_rows",
                "calibration_seasons",
            ],
        ),
        feature_importance=pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["model_name", "feature", "importance", "direction"]),
        final_models=final_models,
        feature_ranges=_feature_ranges(model_frame, feature_cols),
    )
