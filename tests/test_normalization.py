from jets_project.normalization import normalize_team_abbr


def test_normalize_team_abbr_handles_nicknames_and_slash_values() -> None:
    assert normalize_team_abbr("Jets") == "NYJ"
    assert normalize_team_abbr("Packers") == "GB"
    assert normalize_team_abbr("49ers") == "SF"
    assert normalize_team_abbr("NYJ/GB") == "NYJ"
    assert normalize_team_abbr("Washington") == "WAS"
