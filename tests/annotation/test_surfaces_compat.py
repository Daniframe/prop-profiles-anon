"""Confirms Sector 2's output schema is directly loadable by Sector 4's
existing data loaders, with no glue code -- this is a hard downstream
constraint, not just a nice-to-have (see docs/sector1_generation.md /
sector4_surfaces.md and the checked-in real data/benchmarks/annotated/*.jsonl)."""

from src.propensity.annotation.sequential import annotate_sequential
from src.propensity.common.io import write_jsonl
from src.propensity.surfaces.data import load_demands_single_dim


class _FakeClient:
    def __init__(self, response_text):
        from types import SimpleNamespace
        self._response = response_text
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda model, messages, temperature: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._response))]
            )
        ))


def test_annotate_sequential_output_loads_via_load_demands_single_dim(tmp_path):
    questions = [
        {"question_id": "RA_0", "question_text": "Q0"},
        {"question_id": "RA_1", "question_text": "Q1"},
    ]
    client = _FakeClient("The propensity range is [-1, +2]")

    annotated = annotate_sequential(questions, rubric="<rubric>", propensity_name="RA", client=client, model="gpt-4.1")
    output_path = tmp_path / "RA_RA_annotations_test.jsonl"
    write_jsonl(annotated, output_path)

    demands = load_demands_single_dim(output_path)

    assert list(demands["lower"]) == [-1, -1]
    assert list(demands["upper"]) == [2, 2]
    assert list(demands["question_id"]) == ["RA_0", "RA_1"]
