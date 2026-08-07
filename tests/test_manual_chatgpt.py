import json

from src.manual_chatgpt import validate_response_file


def test_manual_chatgpt_response_is_validated(tmp_path):
    path = tmp_path / "response.json"
    path.write_text(json.dumps({
        "is_opportunity": True,
        "role": "Junior Developer",
        "level": "junior",
        "skills": ["Python"],
        "location": "Remote",
        "salary": None,
        "deadline": None,
        "application_link": "https://example.org/apply",
        "confidence": "medium",
        "missing_information": ["closing_date"],
        "evidence": {"role": "Junior Developer", "level": "Junior Developer",
                     "skills": "Python", "location": "Remote", "salary": None,
                     "deadline": None, "application_link": "https://example.org/apply"},
    }), encoding="utf-8")
    result = validate_response_file(path)
    assert result.role == "Junior Developer"
