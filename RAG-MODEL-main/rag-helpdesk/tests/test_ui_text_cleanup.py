from pathlib import Path


def test_no_mojibake_in_app_strings():
    app_text = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")

    assert "ðŸ" not in app_text
    assert "â" not in app_text
    assert "€" not in app_text
    assert "“" not in app_text
    assert "”" not in app_text
