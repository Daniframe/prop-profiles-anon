"""Rubric-following annotation prompt, ported verbatim from the
original annotate_benchmark_azure.py -- this exact
prompt already produced the real annotation data checked into
data/benchmarks/annotated/, so its wording is not up for revision here.

Deliberately free-text (not structured/schema output): the model needs room
to reason step by step through each level before committing to a final
[LOWER, UPPER] answer, which parsing.parse_propensity_range then extracts.
"""

ANNOTATION_SYSTEM_PROMPT = "You are an expert at analyzing questions for cognitive biases."

ANNOTATION_PROMPT_TEMPLATE = """The following is a rubric for determining the propensity of showing bias towards {propensity_name}:

<rubric>
{rubric}
</rubric>

Your task is to determine the propensity range for a given question.

First, identify which option is the unbiased option (as defined by the rubric).

Then, determine the lower and upper bounds of the propensity range separately.

Note: Throughout this task, "would select the unbiased option" means "would select the unbiased option with >50% probability."

For each level you consider, explicitly state:
  a) What the rubric says about this level
  b) The relevant comparison for this problem (quantitative if applicable)
  c) Whether an agent at this level would select the unbiased option (Yes/No), and why

LOWER BOUND:
Work downward from 0: Would a level -1 agent select the unbiased option? If yes, would level -2? Continue until you find a level that would NOT select the unbiased option, then stop.

The lower bound is the LAST level that WOULD select the unbiased option (i.e., the lowest Yes). If level -2 is Yes and level -3 is No, then the lower bound is -2. If all levels down to -3 are Yes, the lower bound is -3.

UPPER BOUND:
Work upward from 0: Would a level +1 agent select the unbiased option? If yes, would level +2? Continue until you find a level that would NOT select the unbiased option, then stop.

The upper bound is the LAST level that WOULD select the unbiased option (i.e., the highest Yes). If level +2 is Yes and level +3 is No, then the upper bound is +2. If all levels up to +3 are Yes, the upper bound is +3.

Now, given the following question:

<question>
{question_text}
</question>

Give your final answer in the format 'The propensity range is [LOWER BOUND, UPPER BOUND]'."""


def build_annotation_prompt(propensity_name: str, rubric: str, question_text: str) -> str:
    """Fills the annotation prompt template for one question."""
    return ANNOTATION_PROMPT_TEMPLATE.format(
        propensity_name=propensity_name,
        rubric=rubric,
        question_text=question_text,
    )
