import json
from types import SimpleNamespace

from src.propensity.generation.spec import BenchmarkSpec
from src.propensity.generation.context_generation import generate_context_seeds


class FakeAzureClient:
    """Mocks the AzureOpenAI structured-output interface: constructs a valid
    instance of whatever dynamic Pydantic schema was requested, without a
    real network call."""

    def __init__(self):
        self.calls = []

        class _Beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(model, messages, temperature, response_format):
                        options_type = response_format.model_fields["options"].annotation
                        option_data = {name: f"{name} text" for name in options_type.model_fields}
                        correct_option = next(iter(options_type.model_fields))
                        parsed = response_format(
                            question_text="Generated question text",
                            options=options_type(**option_data),
                            correct_option=correct_option,
                            reasoning="because reasons",
                        )
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

        self.beta = _Beta()


def _spec(tmp_path):
    contexts_file = tmp_path / "contexts.json"
    contexts_file.write_text(json.dumps({
        "contexts": [{
            "context_name": "ctx1",
            "context_opening": "You are an AI.",
            "context_description": "A situation happens.",
            "options": {"interaction": "talk", "reflection": "think", "distractor": "ignore"},
        }]
    }))
    rubric_file = tmp_path / "rubric.md"
    rubric_file.write_text("Level -3 to 3 rubric text.")
    template_neg = tmp_path / "neg.txt"
    template_neg.write_text(
        "{bias_name} {rubric} {less_biased_level} {less_biased_level_name} "
        "{more_biased_level} {more_biased_level_name} {context_description} "
        "{context_opening} {interaction_option} {reflection_option} {distractor_option}"
    )
    template_pos = tmp_path / "pos.txt"
    template_pos.write_text(template_neg.read_text())

    return BenchmarkSpec(
        name="Ex",
        kind="context_grounded",
        params={
            "bias_name": "extraversion or introversion",
            "option_fields": ["interaction", "reflection", "distractor"],
            "contexts_file": str(contexts_file),
            "rubric_path": str(rubric_file),
            "prompt_template_negative": str(template_neg),
            "prompt_template_positive": str(template_pos),
            "level_names": {-1: "Mild", 0: "Neutral", 1: "Mild+"},
            "level_pairs": [[0, -1], [0, 1]],
        },
    )


def test_generate_context_seeds_produces_one_record_per_context_and_pair(tmp_path):
    spec = _spec(tmp_path)
    results = generate_context_seeds(spec, FakeAzureClient(), model="gpt-4o")
    assert len(results) == 2  # 1 context * 2 level pairs * 1 version


def test_generate_context_seeds_computes_propensity_band_for_negative_pair(tmp_path):
    spec = _spec(tmp_path)
    results = generate_context_seeds(spec, FakeAzureClient())
    negative_pair = next(r for r in results if r["more_biased_level"] == -1)
    assert (negative_pair["intended_propensity_lower"], negative_pair["intended_propensity_higher"]) == (0, 3)


def test_generate_context_seeds_computes_propensity_band_for_positive_pair(tmp_path):
    spec = _spec(tmp_path)
    results = generate_context_seeds(spec, FakeAzureClient())
    positive_pair = next(r for r in results if r["more_biased_level"] == 1)
    assert (positive_pair["intended_propensity_lower"], positive_pair["intended_propensity_higher"]) == (-3, 0)


def test_generate_context_seeds_includes_parsed_question_fields(tmp_path):
    spec = _spec(tmp_path)
    results = generate_context_seeds(spec, FakeAzureClient())
    for r in results:
        assert r["question_text"] == "Generated question text"
        assert set(r["options"]) == {"interaction", "reflection", "distractor"}
        assert r["correct_option"] in {"interaction", "reflection"}


def test_generate_context_seeds_respects_versions_param(tmp_path):
    spec = _spec(tmp_path)
    results = generate_context_seeds(spec, FakeAzureClient(), versions=2)
    assert len(results) == 4  # 1 context * 2 pairs * 2 versions
