import pytest

from src.propensity.generation.spec import BenchmarkSpec, load_spec


def test_load_spec_reads_name_kind_and_params(tmp_path):
    path = tmp_path / "ra.yaml"
    path.write_text("name: RA\nkind: risk_levels\nquestions_per_level: 10\n")

    spec = load_spec(path)

    assert spec.name == "RA"
    assert spec.kind == "risk_levels"
    assert spec.params == {"questions_per_level": 10}


def test_benchmark_spec_rejects_unknown_kind():
    with pytest.raises(ValueError):
        BenchmarkSpec(name="X", kind="not_a_real_kind")


@pytest.mark.parametrize("kind", ["risk_levels", "color_permutation", "context_grounded", "external"])
def test_benchmark_spec_accepts_valid_kinds(kind):
    spec = BenchmarkSpec(name="X", kind=kind)
    assert spec.kind == kind
