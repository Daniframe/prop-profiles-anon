"""Renders context-grounded seed questions (from context_generation.py, live
or pre-generated) into multiple choice questions with shuffled option order.
Ported from the original render_Introv_Ultracrep.py,
which was already generic over the option field names -- works unchanged for
both Ex (interaction/reflection/distractor) and Ul (epistemic_assertion/
epistemic_caution/distractor) seed files.
"""

import random

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_seed_questions(seeds: list[dict], benchmark_code: str, rng: random.Random) -> list[dict]:
    """Renders each seed dict (question_text, options: {type: text},
    correct_option, context_name, intended_propensity_lower/higher) into a
    multiple choice question with a randomized option order."""
    questions = []
    for i, seed in enumerate(seeds):
        options_list = list(seed["options"].items())
        rng.shuffle(options_list)

        labels = {}
        option_order = []
        options_dict = {}
        lines = [seed["question_text"], ""]
        for pos, (opt_type, opt_text) in enumerate(options_list):
            label = f"Option {LETTERS[pos]}"
            labels[opt_type] = label
            option_order.append(opt_type)
            options_dict[label] = opt_text
            lines.append(f"- {label}: {opt_text}")
        lines.append("")
        lines.append('Which option do you choose? The option should be specified with "Option A", "Option B", or "Option C".')

        correct_option = seed["correct_option"]
        lower, higher = seed["intended_propensity_lower"], seed["intended_propensity_higher"]

        questions.append({
            "question_id": f"{benchmark_code}_{i}",
            "context_name": seed["context_name"],
            "question_text": "\n".join(lines),
            "correct_answer": labels[correct_option],
            "correct_option": correct_option,
            "option_order": option_order,
            "options": options_dict,
            "intended_propensity_lower": lower,
            "intended_propensity_higher": higher,
            "intended_propensity_band": f"[{lower:+d}, {higher:+d}]",
        })
    return questions
