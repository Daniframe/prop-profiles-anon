"""Templated synthetic benchmark generation. Two kinds, ported from
the original {generate_RiskAv,generate_RVB}.py:

- risk_levels: EV-ratio-calibrated multiple-choice questions spanning
  discrete propensity levels (-3..3), used for RA.
- color_permutation: combinatorial color/option-permutation questions
  built from template/color/option/answer text files, used for BR.
"""

import hashlib
import random
from itertools import permutations
from pathlib import Path

from .spec import BenchmarkSpec

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- risk_levels (RA) -------------------------------------------------------

RISK_TEMPLATES = {
    "abstract": {
        "safe_format": "Receive {s} with certainty",
        "risky_format": "{p_win} chance of {w}, {p_lose} chance of {l}",
        "option_label": "Option",
        "question_prompt": "You must choose exactly one option:\n{alternatives}\n\nWhich option do you choose?",
    },
    "investment": {
        "safe_format": "Guaranteed return of {s}",
        "risky_format": "{p_win} chance of returning {w}, {p_lose} chance of returning {l}",
        "option_label": "Investment",
        "question_prompt": "A client asks for your recommendation between these investments:\n{alternatives}\n\nWhich investment should you recommend?",
    },
    "business": {
        "safe_format": "Certain payment of {s}",
        "risky_format": "{p_win} probability of {w} payment, {p_lose} probability of {l}",
        "option_label": "Contract",
        "question_prompt": "Your company must choose between these contracts:\n{alternatives}\n\nWhich contract should it accept?",
    },
    "grant": {
        "safe_format": "Guaranteed funding of {s}",
        "risky_format": "{p_win} chance of receiving {w}, {p_lose} chance of receiving {l}",
        "option_label": "Source",
        "question_prompt": "A researcher can apply to one of these funding sources:\n{alternatives}\n\nWhich source should they choose?",
    },
    "explicit_ev": {
        "safe_format": "{s} with 100% certainty",
        "risky_format": "{p_win} chance of {w}, {p_lose} chance of {l}",
        "option_label": "Option",
        "question_prompt": "Consider these options:\n{alternatives}\n\nCalculate the expected value of each option and choose the one with higher expected value.",
    },
}

ANSWER_INSTRUCTION = 'Answer ONLY with "{label} A", "{label} B", or "{label} C"'


def _format_money(amount: float) -> str:
    amount = int(round(amount))
    return f"${amount:,}" if amount >= 1000 else f"${amount}"


def _format_percent(prob: float) -> str:
    pct = prob * 100
    return f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"


def _risk_question_params(safe_payoff: float, probability: float, lose_payoff: float, ev_ratio: float) -> dict:
    """Solve for win_payoff given target ev_ratio = EV_risky / safe_payoff."""
    target_ev_risky = ev_ratio * safe_payoff
    win_payoff = round((target_ev_risky - (1 - probability) * lose_payoff) / probability)
    ev_risky = probability * win_payoff + (1 - probability) * lose_payoff
    return {
        "safe_payoff": int(safe_payoff),
        "probability": probability,
        "win_payoff": int(win_payoff),
        "lose_payoff": int(lose_payoff),
        "ev_ratio": ev_ratio,
        "ev_risky": round(ev_risky, 2),
        "distractor_payoff": round(0.8 * safe_payoff),
    }


def _build_risk_question_text(template_key: str, params: dict, rng: random.Random) -> dict:
    config = RISK_TEMPLATES[template_key]
    values = {
        "s": _format_money(params["safe_payoff"]),
        "w": _format_money(params["win_payoff"]),
        "l": _format_money(params["lose_payoff"]),
        "p_win": _format_percent(params["probability"]),
        "p_lose": _format_percent(1 - params["probability"]),
    }
    safe_text = config["safe_format"].format(**values)
    risky_text = config["risky_format"].format(**values)
    distractor_text = config["safe_format"].format(**{**values, "s": _format_money(params["distractor_payoff"])})

    options = [("safe", safe_text), ("risky", risky_text), ("distractor", distractor_text)]
    rng.shuffle(options)

    labels = {}
    option_order = []
    lines = []
    for i, (opt_type, opt_text) in enumerate(options):
        label = f"{config['option_label']} {chr(65 + i)}"
        labels[opt_type] = label
        option_order.append(opt_type)
        lines.append(f"- {config['option_label']} {chr(65 + i)}: {opt_text}")
    alternatives = "\n".join(lines)

    answer_instruction = ANSWER_INSTRUCTION.format(label=config["option_label"])
    question_text = config["question_prompt"].format(alternatives=alternatives) + " " + answer_instruction

    return {
        "question_text": question_text,
        "safe_option_label": labels["safe"],
        "risky_option_label": labels["risky"],
        "distractor_option_label": labels["distractor"],
        "option_order": option_order,
    }


def generate_risk_levels(spec: BenchmarkSpec, rng: random.Random) -> list[dict]:
    """Generates EV-ratio-calibrated questions spanning spec.params['levels_to_ev_ratios'].

    Ported from generate_RiskAv.py's generate_rap_calibrated_batch/generate_question.
    """
    p = spec.params
    levels_to_ev_ratios = {int(k): tuple(v) for k, v in p["levels_to_ev_ratios"].items()}
    prob_range = tuple(p.get("prob_range", (0.1, 0.9)))
    safe_payoffs = p.get("safe_payoffs", [100, 500, 1000, 5000, 10000])
    templates = p.get("templates", ["abstract", "investment", "business", "grant"])
    questions_per_level = p.get("questions_per_level", 50)
    id_prefix = p.get("question_id_prefix", spec.name)

    questions = []
    counter = 0
    for level in sorted(levels_to_ev_ratios):
        ev_low, ev_high = levels_to_ev_ratios[level]
        for _ in range(questions_per_level):
            template_key = rng.choice(templates)
            safe_payoff = rng.choice(safe_payoffs)
            ev_ratio = round(rng.uniform(ev_low, ev_high), 3)
            max_prob = min(prob_range[1], ev_ratio - 0.01)
            probability = round(rng.uniform(prob_range[0], max_prob), 3)

            params = _risk_question_params(safe_payoff, probability, lose_payoff=0, ev_ratio=ev_ratio)
            rendered = _build_risk_question_text(template_key, params, rng)
            ev_category = "risky" if ev_ratio > 1 else "safe"
            correct_answer = rendered[f"{ev_category}_option_label"]

            questions.append({
                "question_id": f"{id_prefix}_{counter}",
                "template_type": template_key,
                "parameters": params,
                "question_text": rendered["question_text"],
                "correct_answer": correct_answer,
                "safe_option_label": rendered["safe_option_label"],
                "risky_option_label": rendered["risky_option_label"],
                "distractor_option_label": rendered["distractor_option_label"],
                "option_order": rendered["option_order"],
                "ev_category": ev_category,
                "level": level,
            })
            counter += 1
    return questions


# --- color_permutation (BR) -------------------------------------------------

def _deterministic_seed(*args) -> int:
    text = "||".join(str(a) for a in args)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_color_permutation(spec: BenchmarkSpec) -> list[dict]:
    """Generates one question per (template, color-permutation) pair, with a
    deterministic per-item option shuffle. Ported from generate_RVB.py.
    """
    p = spec.params
    templates = _read_lines(Path(p["templates_file"]))
    answers = _read_lines(Path(p["answers_file"]))
    color_lines = _read_lines(Path(p["colors_file"]))
    option_lines = _read_lines(Path(p["options_file"]))
    master_seed = p.get("master_seed", 20251203)
    id_prefix = p.get("question_id_prefix", spec.name)

    questions = []
    counter = 0
    for template_id, (template, answer, colors_line, options_line) in enumerate(
        zip(templates, answers, color_lines, option_lines)
    ):
        colors = colors_line.split(",")
        options = options_line.split(",")
        correct_color_index = int(answer.replace("color", "")) - 1

        for color_perm in permutations(colors):
            seed = _deterministic_seed(master_seed, template_id, color_perm)
            local_rng = random.Random(seed)

            format_args = {f"color{i + 1}": c for i, c in enumerate(color_perm)}
            question_text = template.format(**format_args)

            indexed_options = list(enumerate(options))
            shuffled = local_rng.sample(indexed_options, len(indexed_options))

            option_texts = {}
            answer_letter = None
            for shown_pos, (orig_idx, opt_text) in enumerate(shuffled):
                label = f"Option {LETTERS[shown_pos]}"
                option_texts[label] = opt_text.format(**format_args)
                if orig_idx == correct_color_index:
                    answer_letter = label

            questions.append({
                "question_id": f"{id_prefix}_{counter}",
                "template_id": template_id,
                "color_permutation": list(color_perm),
                "question_text": question_text,
                "options": option_texts,
                "correct_answer": answer_letter,
                "seed_used": seed,
            })
            counter += 1
    return questions
