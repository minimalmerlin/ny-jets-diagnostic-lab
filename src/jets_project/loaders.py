from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

import pandas as pd

from .normalization import clean_columns
from .optional import import_optional_dependency

LOGGER = logging.getLogger(__name__)


def _to_pandas(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _is_missing_resource_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message or "not found" in message or "failed to download" in message


class NFLDataClient:
    """Thin wrapper around nflreadpy with resilient loader resolution."""

    def __init__(self) -> None:
        self._module = None

    @property
    def module(self):
        if self._module is None:
            self._module = import_optional_dependency(
                "nflreadpy", "uv sync"
            )
        return self._module

    def _resolve_loader(self, *candidates: str) -> Callable:
        for candidate in candidates:
            loader = getattr(self.module, candidate, None)
            if callable(loader):
                return loader
        available = [
            name
            for name, value in inspect.getmembers(self.module)
            if callable(value) and name.startswith("load_")
        ]
        raise RuntimeError(
            "Kein passender nflreadpy-Loader gefunden. "
            f"Erwartet wurde einer von {candidates}, vorhanden sind u. a. {available[:12]}."
        )

    def _attempt_loader_call(self, loader: Callable, kwargs_candidates: list[dict]) -> pd.DataFrame:
        last_error: Exception | None = None
        for kwargs in kwargs_candidates:
            try:
                return clean_columns(_to_pandas(loader(**kwargs)))
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Loader-Aufruf fehlgeschlagen.")

    def _loader_kwargs(self, seasons: list[int], **extra_kwargs) -> list[dict]:
        attempts = []
        if extra_kwargs:
            attempts.extend(
                [
                    {"seasons": seasons, **extra_kwargs},
                    {"season": seasons, **extra_kwargs},
                    {"years": seasons, **extra_kwargs},
                ]
            )
        attempts.extend(
            [
                {"seasons": seasons},
                {"season": seasons},
                {"years": seasons},
                {},
            ]
        )
        return attempts

    def _call_loader_partial(self, loader: Callable, seasons: list[int], **extra_kwargs) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for season in seasons:
            season_attempts = []
            if extra_kwargs:
                season_attempts.extend(
                    [
                        {"seasons": [season], **extra_kwargs},
                        {"season": [season], **extra_kwargs},
                        {"season": season, **extra_kwargs},
                        {"years": [season], **extra_kwargs},
                        {"years": season, **extra_kwargs},
                    ]
                )
            season_attempts.extend(
                [
                    {"seasons": [season]},
                    {"season": [season]},
                    {"season": season},
                    {"years": [season]},
                    {"years": season},
                ]
            )
            try:
                frame = self._attempt_loader_call(loader, season_attempts)
            except Exception as exc:
                if _is_missing_resource_error(exc):
                    LOGGER.warning("Quelle fuer Saison %s nicht verfuegbar; Saison wird uebersprungen.", season)
                    continue
                raise
            frames.append(frame)

        if not frames:
            return pd.DataFrame()
        return clean_columns(pd.concat(frames, ignore_index=True))

    def _call_loader(self, loader: Callable, seasons: list[int], allow_partial: bool = False, **extra_kwargs) -> pd.DataFrame:
        attempts = self._loader_kwargs(seasons, **extra_kwargs)

        for kwargs in attempts:
            try:
                return clean_columns(_to_pandas(loader(**kwargs)))
            except TypeError:
                continue
            except Exception as exc:
                if allow_partial and _is_missing_resource_error(exc):
                    return self._call_loader_partial(loader, seasons, **extra_kwargs)
                raise

        return self._attempt_loader_call(loader, attempts)

    def load_schedules(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_schedules")
        return self._call_loader(loader, seasons)

    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_pbp", "load_play_by_play")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_team_stats(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_team_stats", "load_weekly_team_stats")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_player_stats(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_player_stats", "load_weekly_player_stats")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_snap_counts(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_snap_counts")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_rosters(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_rosters", "load_weekly_rosters")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_injuries(self, seasons: list[int]) -> pd.DataFrame:
        loader = self._resolve_loader("load_injuries")
        return self._call_loader(loader, seasons, allow_partial=True)

    def load_contracts(self) -> pd.DataFrame:
        loader = self._resolve_loader("load_contracts")
        return clean_columns(_to_pandas(loader()))

    def load_nextgen(self, kind: str, seasons: list[int]) -> pd.DataFrame:
        specialized_names = {
            "passing": ("load_ngs_passing", "load_player_ngs_passing"),
            "rushing": ("load_ngs_rushing", "load_player_ngs_rushing"),
            "receiving": ("load_ngs_receiving", "load_player_ngs_receiving"),
        }
        generic = self._resolve_loader(
            *specialized_names.get(kind, ()),
            "load_nextgen_stats",
        )
        for kwargs in (
            {"stat_type": kind},
            {"kind": kind},
            {"dataset": kind},
            {"stat_group": kind},
        ):
            try:
                return self._call_loader(generic, seasons, allow_partial=True, **kwargs)
            except TypeError:
                continue
        return self._call_loader(generic, seasons, allow_partial=True)
