from pathlib import Path

from src.propensity.annotation.rubrics import rubric_path_for, load_rubric_text


def test_rubric_path_for_resolves_canonical_location():
    assert rubric_path_for("RA") == Path("rubrics") / "RA" / "RA_v1.md"


def test_rubric_path_for_accepts_custom_rubrics_dir(tmp_path):
    path = rubric_path_for("RA", rubrics_dir=tmp_path)
    assert path == tmp_path / "RA" / "RA_v1.md"


def test_load_rubric_text_reads_file_content(tmp_path):
    rubric_file = tmp_path / "rubric.md"
    rubric_file.write_text("Level -3 to 3 rubric text.", encoding="utf-8")

    assert load_rubric_text(rubric_file) == "Level -3 to 3 rubric text."


def test_load_rubric_text_reads_real_ra_rubric():
    text = load_rubric_text(rubric_path_for("RA"))
    assert "Level" in text or "level" in text
    assert len(text) > 0
