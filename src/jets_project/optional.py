from __future__ import annotations

from importlib import import_module


def import_optional_dependency(module_name: str, install_hint: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Optionale Abhaengigkeit '{module_name}' fehlt. Installiere sie mit `{install_hint}`."
        ) from exc
