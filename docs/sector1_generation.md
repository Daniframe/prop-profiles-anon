# Sector 1: Dataset Generation

Package: `src/propensity/generation/` · CLIs: `scripts/generate_benchmark.py`, `scripts/render_benchmark.py`
Specs: `benchmarks/*.yaml` · Config: `config/generation.yaml` (Azure defaults, read by the CLI)

Builds benchmarks where a specific propensity is the dominant factor in
predicting success/failure. Each benchmark is declared once as a
`benchmarks/{code}.yaml` spec and generated through one of four engines,
selected by the spec's `kind`:

| `kind` | Benchmark(s) | Needs credentials? | Module |
|---|---|---|---|
| `risk_levels` | RA | No -- pure local math | `engine.py` |
| `color_permutation` | BR | No -- pure local math | `engine.py` |
| `context_grounded` | Ex (live), Ul (pre-generated seeds) | Ex: yes, Azure OpenAI | `context_generation.py` + `render.py` |
| `external` | TimeMenatQA | No -- reads a local CSV | `external.py` |

## Modules

| Module | Responsibility |
|---|---|
| `spec.py` | `BenchmarkSpec` dataclass + `benchmarks/*.yaml` loader. |
| `engine.py` | `generate_risk_levels` (EV-ratio-calibrated questions across levels -3..3) and `generate_color_permutation` (combinatorial color/option-permutation questions from template files). Both deterministic, no network calls. |
| `context_generation.py` | `generate_context_seeds`: one Azure structured-output call per (context, adjacent level pair), validated against a dynamically-built Pydantic schema. **Requires `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT` and consumes API credits.** |
| `render.py` | `render_seed_questions`: turns context-grounded seed questions (live-generated or pre-existing) into multiple choice questions with a randomized, reproducible option order. No network calls. |
| `external.py` | `load_external_benchmark`: column-renaming adapter for a real/external benchmark CSV (no generation step). |

## Usage

### Loading a spec

```python
from src.propensity.generation import load_spec

spec = load_spec("benchmarks/ra.yaml")
# BenchmarkSpec(name="RA", kind="risk_levels", params={...})
```

### `risk_levels` (RA) -- no credentials needed

```python
import random
from src.propensity.generation import load_spec, generate_risk_levels

spec = load_spec("benchmarks/ra.yaml")
questions = generate_risk_levels(spec, random.Random(42))
# [{"question_id": "RA_0", "question_text": "...", "correct_answer": "Contract B", ...}, ...]
```

### `color_permutation` (BR) -- no credentials needed

```python
from src.propensity.generation import load_spec, generate_color_permutation

spec = load_spec("benchmarks/br.yaml")
questions = generate_color_permutation(spec)  # self-contained deterministic RNG, no seed argument needed
```

### `context_grounded`, live generation (Ex) -- **requires Azure credentials, consumes credits**

Shown for reference only; not executed here or in the test suite (which mocks
the client -- see `tests/generation/test_context_generation.py`).

```python
from src.propensity.common.llm_clients import get_azure_client
from src.propensity.generation import load_spec, generate_context_seeds

spec = load_spec("benchmarks/ex.yaml")
client = get_azure_client()  # reads AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT from the environment
seeds = generate_context_seeds(spec, client, model="gpt-4o", temperature=0.5)  # makes real API calls
```

### `context_grounded`, rendering (Ex, Ul) -- no credentials needed

Rendering is a separate, local-only step. Ul's spec (`benchmarks/ul.yaml`) has
no live-generation params -- it points at its pre-generated seed file instead,
since the original Ultracrepidarianism generator uses a differently-shaped
batched prompt that isn't ported here (see the comment in `ul.yaml`).

```python
import json
import random
from src.propensity.generation import render_seed_questions

seeds = json.load(open("data/benchmarks/raw/Ul/ul_generated_context_levels.json"))
questions = render_seed_questions(seeds, benchmark_code="Ul", rng=random.Random(42))
```

### `external` (TimeMenatQA) -- no credentials needed

```python
from src.propensity.generation import load_spec, load_external_benchmark

spec = load_spec("benchmarks/timenatqa.yaml")
questions = load_external_benchmark(spec)  # reads data/benchmarks/raw/TimeMenatQA/TimeAndMentalQA.csv
```

### CLI

`generate_benchmark.py` always produces the final `ready/*.jsonl` file, regardless of kind:

```bash
# risk_levels / color_permutation / external: generated directly
python scripts/generate_benchmark.py --spec benchmarks/ra.yaml --out data/benchmarks/ready/RA/RA.jsonl
python scripts/generate_benchmark.py --spec benchmarks/br.yaml --out data/benchmarks/ready/BR/BR.jsonl
python scripts/generate_benchmark.py --spec benchmarks/timenatqa.yaml --out data/benchmarks/ready/TimeMenatQA/TimeMenatQA.jsonl

# context_grounded with a seed_file param (Ul): loads it directly, no Azure call
python scripts/generate_benchmark.py --spec benchmarks/ul.yaml --out data/benchmarks/ready/Ul/Ul.jsonl

# context_grounded without a seed_file param (Ex): calls Azure to generate seeds first
# (requires AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT -- not run as part of this repo's docs/tests)
python scripts/generate_benchmark.py --spec benchmarks/ex.yaml --out data/benchmarks/ready/Ex/Ex.jsonl
```

`render_benchmark.py` is the standalone render-only entry point -- useful to
re-render an existing seed file (e.g. with a different `--hex-seed`, for a
different option shuffle) without touching Azure at all:

```bash
python scripts/render_benchmark.py \
  --seed-file data/benchmarks/raw/Ex/ex_generated_context_levels.json \
  --benchmark-code Ex \
  --out data/benchmarks/ready/Ex/Ex.jsonl
```

All CLI runs print a reproducible `Generation ID: YYYY-MM-DD-XXXX (hex seed: XXXX)` line;
pass `--hex-seed XXXX` to reproduce a previous run exactly.
