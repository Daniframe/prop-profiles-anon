# Sector 2: Annotation

Package: `src/propensity/annotation/` · CLI: `scripts/annotate_benchmark.py`
Config: `config/annotation.yaml` (default model/temperature, credential env vars)

Asks an LLM to read a rubric and a benchmark question, then output a
propensity **demand interval** `[lower, upper]` -- the range of propensity
levels for which an unbiased agent would still pick the correct answer. The
output schema (`propensity_lower`/`propensity_upper`) is exactly what
Sector 4's `surfaces.data.load_demands_single_dim` already consumes; this is
verified by `tests/annotation/test_surfaces_compat.py`.

Every function takes a duck-typed `client` (`common.llm_clients.get_azure_client()`
or `get_openai_client()`) -- nothing here is hardcoded to Azure.

**All examples below that make a real LLM call need `AZURE_OPENAI_API_KEY`/
`AZURE_OPENAI_ENDPOINT` (or `OPENAI_API_KEY` for `--provider openai`) and
consume API credits.** They are not executed here or in the test suite,
which mocks the client instead (see `tests/annotation/test_sequential.py`,
`test_batch.py`).

## Modules

| Module | Responsibility |
|---|---|
| `rubrics.py` | `rubric_path_for(code)` resolves the canonical `rubrics/{code}/{code}_v1.md`; `load_rubric_text(path)` reads it. |
| `prompts.py` | `ANNOTATION_SYSTEM_PROMPT` + `build_annotation_prompt(...)` -- the proven, rubric-following chain-of-thought template that already produced the real checked-in annotation data. |
| `parsing.py` | `parse_propensity_range(response_text)` -- extracts `[LOWER, UPPER]` from the model's free-text response via regex. No network calls. |
| `sequential.py` | `annotate_sequential`: one LLM call per question. |
| `batch.py` | `build_batch_requests`/`submit_batch`/`check_batch_status`/`process_batch`: OpenAI/Azure Batch API on `/v1/chat/completions`, for annotating many questions cheaply. |

## Usage

### Loading a rubric -- no credentials needed

```python
from src.propensity.annotation import rubric_path_for, load_rubric_text

rubric = load_rubric_text(rubric_path_for("RA"))  # rubrics/RA/RA_v1.md
```

### Building the prompt -- no credentials needed

```python
from src.propensity.annotation import build_annotation_prompt

prompt = build_annotation_prompt(
    propensity_name="risk aversion",
    rubric=rubric,
    question_text="Which contract should it accept? ...",
)
```

### Parsing a response -- no credentials needed

```python
from src.propensity.annotation import parse_propensity_range

parse_propensity_range("... The propensity range is [-1, +2].")
# (-1, 2)
```

### Sequential annotation (one call per question) -- **requires credentials, consumes credits**

```python
from src.propensity.common import read_jsonl
from src.propensity.common.llm_clients import get_azure_client
from src.propensity.annotation import rubric_path_for, load_rubric_text, annotate_sequential

questions = list(read_jsonl("data/benchmarks/ready/RA/RA.jsonl"))
rubric = load_rubric_text(rubric_path_for("RA"))
client = get_azure_client()

annotated = annotate_sequential(questions, rubric, propensity_name="risk aversion", client=client, model="gpt-4.1")
# each item gains propensity_lower, propensity_upper, raw_annotation_response
```

### Batch annotation (cheaper at scale) -- **requires credentials, consumes credits**

```python
from src.propensity.annotation import build_batch_requests, submit_batch, check_batch_status, process_batch

requests = build_batch_requests(questions, rubric, propensity_name="risk aversion", model="gpt-4.1")
batch_id = submit_batch(client, requests, "data/inference/raw/RA/annotation_batch_input.jsonl")

check_batch_status(client, batch_id)  # poll until status == "completed"

annotated = process_batch(client, batch_id, questions)
```

### CLI

Matches the exact `submit|status|process|sequential` mode contract documented
in the original annotation scripts' README, now provider-agnostic via `--provider`:

```bash
# Sequential
python scripts/annotate_benchmark.py sequential \
  --dataset_path data/benchmarks/ready/RA/RA.jsonl \
  --rubric_path rubrics/RA/RA_v1.md \
  --prop_name "risk aversion" \
  --annotation_output_path data/benchmarks/annotated/RA/RA_RA_annotations.jsonl \
  --provider azure

# Batch
python scripts/annotate_benchmark.py submit \
  --dataset_path data/benchmarks/ready/RA/RA.jsonl \
  --rubric_path rubrics/RA/RA_v1.md \
  --prop_name "risk aversion" \
  --batch_input_path data/inference/raw/RA/annotation_batch_input.jsonl \
  --provider azure

python scripts/annotate_benchmark.py status --batch_id <id> --provider azure

python scripts/annotate_benchmark.py process \
  --batch_id <id> \
  --dataset_path data/benchmarks/ready/RA/RA.jsonl \
  --annotation_output_path data/benchmarks/annotated/RA/RA_RA_annotations.jsonl \
  --provider azure
```

`--model`/`--temperature` override `config/annotation.yaml`'s `annotator_model`/`annotator_temperature` defaults.
