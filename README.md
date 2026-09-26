# Propensity Profiles

Anonymized code and data accompanying a double-blind submission: a
reproduction pipeline and framework for inferring propensity profiles of AI
systems from instance-level success/failure results and LLM-annotated
demand-interval rubrics.

The pipeline has 5 sectors, each a subpackage of `src/propensity/`:

| # | Sector | Package | Docs | Status |
|---|---|---|---|---|
| 1 | Dataset generation | `src/propensity/generation/` | [docs/sector1_generation.md](docs/sector1_generation.md) | Implemented |
| 2 | Annotation (rubric-based demand intervals) | `src/propensity/annotation/` | [docs/sector2_annotation.md](docs/sector2_annotation.md) | Implemented |
| 3 | Model inference (instance-level outcomes across incitement levels) | `src/propensity/inference/` | [docs/sector3_inference.md](docs/sector3_inference.md) | Implemented (judge + reunite included) |
| 4 | Propensity surfaces (curve/surface fitting) | `src/propensity/surfaces/` | [docs/sector4_surfaces.md](docs/sector4_surfaces.md) | Implemented |
| 5 | Predictability (does propensity improve outcome prediction?) | `src/propensity/predictability/` | -- | Not yet implemented |

## Setup

```bash
conda env create -f environment.yaml   # or: pip install -r requirements.txt
```

Sectors that call an LLM provider (annotation, part of generation, part of
inference) read `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT` or
`OPENAI_API_KEY` from the environment (via `.env`, `python-dotenv`) -- see
each sector's docs for which functionality needs them. None of these sectors
are hardcoded to a specific provider: any client exposing the OpenAI
Chat Completions interface works (OpenAI, Azure OpenAI, or a self-hosted
OpenAI-compatible server via `base_url`). Sector 3 additionally supports
local GPU inference via vLLM (`pip install vllm torch` separately -- not a
default dependency). Sector 4 needs no credentials or GPU at all.

## Repository layout

```
src/propensity/{common,generation,annotation,inference,surfaces,predictability}/
scripts/            # one CLI per pipeline stage
config/             # one YAML per sector
benchmarks/         # Sector 1 declarative benchmark specs
rubrics/            # canonical propensity rubrics (RA, BR, Ex, Ul)
prompts/incitement/ # canonical incitement system prompts (RA, BR, Ex, Ul)
data/               # small/final artifacts checked in; data/inference/raw/ gitignored (regenerable dumps)
tests/              # pytest, one folder per sector once implemented
docs/               # per-sector documentation + usage examples
```

## Testing

```bash
pytest tests/ -q
```

Every test is synthetic-fixture-based -- no network calls, no GPU, no
credentials required to run the suite (functionality that needs an LLM
provider or GPU is exercised against a mocked client/fake backend instead).
