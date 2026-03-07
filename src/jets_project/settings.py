from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ProjectPaths:
    root: Path = field(default_factory=_project_root)
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    data_reference: Path = field(init=False)
    db_dir: Path = field(init=False)
    artifacts_models: Path = field(init=False)
    artifacts_app: Path = field(init=False)
    artifacts_reports: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.data_reference = self.root / "data" / "reference"
        self.db_dir = self.root / "db"
        self.artifacts_models = self.root / "artifacts" / "models"
        self.artifacts_app = self.root / "artifacts" / "app"
        self.artifacts_reports = self.root / "artifacts" / "reports"
        self.reports_dir = self.root / "reports"

    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "jets_nfl.sqlite"

    @property
    def staff_tenure_path(self) -> Path:
        return self.data_reference / "staff_tenure.csv"

    @property
    def model_bundle_path(self) -> Path:
        return self.artifacts_models / "model_bundle.joblib"

    @property
    def model_matrix_path(self) -> Path:
        return self.data_processed / "game_features.parquet"

    @property
    def app_sqlite_path(self) -> Path:
        return self.artifacts_app / "jets_app.sqlite"


@dataclass(slots=True)
class ProjectConfig:
    start_season: int = 2016
    end_season: int = 2025
    focus_team: str = "NYJ"
    report_language: str = "de"
    top_teams: tuple[str, ...] = ("BAL", "BUF", "KC", "PHI", "SF")
    latest_complete_season_fallback: int = 2024
    regular_season_only: bool = True
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def seasons(self) -> list[int]:
        return list(range(self.start_season, self.end_season + 1))

    def ensure_directories(self) -> None:
        for path in (
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.db_dir,
            self.paths.artifacts_models,
            self.paths.artifacts_app,
            self.paths.artifacts_reports,
            self.paths.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_default_config() -> ProjectConfig:
    config = ProjectConfig()
    config.ensure_directories()
    return config
