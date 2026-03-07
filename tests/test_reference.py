from pathlib import Path

from jets_project.reference import load_staff_tenure


def test_load_staff_tenure_computes_changes_and_tenure(tmp_path: Path) -> None:
    path = tmp_path / "staff.csv"
    path.write_text(
        "season,team,head_coach,offensive_coordinator,defensive_coordinator,general_manager\n"
        "2021,NYJ,Coach A,OC A,DC A,GM A\n"
        "2022,NYJ,Coach A,OC A,DC A,GM A\n"
        "2023,NYJ,Coach B,OC B,DC A,GM A\n"
    )

    staff = load_staff_tenure(path)

    assert staff.loc[1, "hc_tenure_years"] == 2
    assert staff.loc[2, "hc_changed"] == 1
    assert staff.loc[2, "oc_changed"] == 1
    assert staff.loc[2, "gm_changed"] == 0
