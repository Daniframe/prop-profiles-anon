"""Context-grounded benchmark seed generation via Azure OpenAI structured
outputs. Generalizes the original generate_Introv_context_levels.py
(one question per context x adjacent-level-pair): for each context and each
level pair in spec.params['level_pairs'], asks the model to write a multiple
choice question distinguishing the two levels, validated against a dynamically
built Pydantic schema keyed by spec.params['option_fields'].

Only the simpler "one call per level pair" generation pattern is ported here.
generate_Ultracrep_context_levels.py's batched "three questions per call"
variant is structurally different (a single prompt spanning 3 level pairs at
once) and is not reproduced -- Ul's pre-generated seed file is checked into
data/benchmarks/raw/Ul/ and consumed directly by render.py instead.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .spec import BenchmarkSpec

SYSTEM_PROMPT = "You are an expert psychometrician designing behavioral assessment questions."


def _build_question_schema(option_fields: list[str]) -> type[BaseModel]:
    """Dynamically builds a Pydantic schema with one str field per option
    (e.g. interaction/reflection/distractor), matching the seed-file shape
    already used across benchmarks (see data/benchmarks/raw/{Ex,Ul}/*_generated_context_levels.json)."""
    options_model = create_model(
        "QuestionOptions",
        **{field_name: (str, Field(description=f"The {field_name} option, with added situational detail")) for field_name in option_fields},
    )
    biased_fields = tuple(option_fields[:2])
    return create_model(
        "ContextGeneratedQuestion",
        question_text=(str, Field(description="The question text, ending to invite (but not include) the options")),
        options=(options_model, ...),
        correct_option=(Literal[biased_fields], Field(description="Which option is the unbiased choice for this question")),
        reasoning=(str, Field(description="Brief explanation of why this question distinguishes between the two target levels")),
    )


def _build_prompt(template: str, rubric: str, context: dict, less_biased: int, more_biased: int,
                   level_names: dict[int, str], option_fields: list[str], bias_name: str) -> str:
    format_args = {
        "bias_name": bias_name,
        "rubric": rubric,
        "less_biased_level": less_biased,
        "less_biased_level_name": level_names[less_biased],
        "more_biased_level": more_biased,
        "more_biased_level_name": level_names[more_biased],
        "context_name": context["context_name"],
        "context_description": context["context_description"],
        "context_opening": context.get("context_opening", ""),
    }
    for field_name in option_fields:
        format_args[f"{field_name}_option"] = context["options"][field_name]
    return template.format(**format_args)


def generate_context_seeds(
    spec: BenchmarkSpec,
    client,
    model: str = "gpt-4o",
    temperature: float = 0.5,
    versions: int = 1,
) -> list[dict]:
    """Generates one seed question per (context, level pair, version) via the
    Azure client's structured-output parsing. `client` is expected to expose
    the same `beta.chat.completions.parse(...)` interface as AzureOpenAI/OpenAI
    (see src/propensity/common/llm_clients.get_azure_client)."""
    p = spec.params
    with open(p["contexts_file"], encoding="utf-8") as f:
        contexts = json.load(f)["contexts"]
    rubric = Path(p["rubric_path"]).read_text(encoding="utf-8")
    template_positive = Path(p["prompt_template_positive"]).read_text(encoding="utf-8")
    template_negative = Path(p["prompt_template_negative"]).read_text(encoding="utf-8")
    level_names = {int(k): v for k, v in p["level_names"].items()}
    level_pairs = [tuple(pair) for pair in p["level_pairs"]]
    option_fields = p["option_fields"]
    bias_name = p["bias_name"]

    schema = _build_question_schema(option_fields)

    results = []
    for context in contexts:
        for less_biased, more_biased in level_pairs:
            template = template_negative if more_biased < less_biased else template_positive
            prompt = _build_prompt(template, rubric, context, less_biased, more_biased, level_names, option_fields, bias_name)

            for version in range(versions):
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format=schema,
                )
                question = response.choices[0].message.parsed

                if more_biased < less_biased:
                    prop_lower, prop_higher = less_biased, 3
                else:
                    prop_lower, prop_higher = -3, less_biased

                results.append({
                    "context_name": context["context_name"],
                    "less_biased_level": less_biased,
                    "more_biased_level": more_biased,
                    "intended_propensity_lower": prop_lower,
                    "intended_propensity_higher": prop_higher,
                    "version": version + 1,
                    **question.model_dump(),
                })
    return results
