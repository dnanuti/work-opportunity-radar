from src.profile import build_profile


def test_profile_lists_are_clean_and_candidate_specific():
    profile = build_profile(
        "Sam", "Accra, Ghana", "Python, SQL, python", "Data Analyst, QA Engineer",
        "Accra, Remote", "junior, entry_level",
    )
    assert profile["skills"] == ["Python", "SQL"]
    assert profile["target_roles"] == ["Data Analyst", "QA Engineer"]
    assert profile["preferred_levels"] == ["junior", "entry_level"]
