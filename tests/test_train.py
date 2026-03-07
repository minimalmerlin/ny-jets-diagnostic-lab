import pandas as pd

from jets_project.train import _select_complete_training_end_season
from jets_project.storage import write_sqlite_table


def test_select_complete_training_end_season_caps_to_last_complete_source(tmp_path) -> None:
    sqlite_path = tmp_path / "test.sqlite"
    availability = pd.DataFrame(
        [
            {
                "dataset_name": "pbp",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "team_stats",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "player_stats",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "snap_counts",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "rosters",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "ngs_passing",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "ngs_receiving",
                "loaded_seasons": "[2023, 2024, 2025]",
            },
            {
                "dataset_name": "injuries",
                "loaded_seasons": "[2023, 2024]",
            },
        ]
    )
    model_frame = pd.DataFrame(
        [
            {"season": 2023, "game_id": "g1"},
            {"season": 2024, "game_id": "g2"},
            {"season": 2025, "game_id": "g3"},
        ]
    )

    write_sqlite_table(availability, "dataset_availability", sqlite_path)

    selected = _select_complete_training_end_season(sqlite_path, model_frame, preferred_end_season=2025)

    assert selected == 2024
