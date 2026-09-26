# Sector 3: Model Inference

Package: `src/propensity/inference/` · CLIs: `scripts/run_inference.py`, `scripts/judge_outcomes.py`
Config: `config/inference.yaml`

Runs every incitement-level system prompt against every question in a ready
benchmark, for one model, producing raw per-item responses. Deliberately
**not** tied to a specific client provider: the real fork is *local
in-process GPU inference* (`VLLMBackend`) vs. *any OpenAI-Chat-Completions-
compatible HTTP endpoint* (`OpenAICompatibleBackend` -- OpenAI, Azure OpenAI,
or a self-hosted server), both behind one `InferenceBackend.generate_batch()`
interface and one shared `run_inference()` loop.

**Everything that makes a real GPU or API call below is shown for reference
only and not executed here or in the test suite**, which uses fakes instead
(see `tests/inference/test_openai_compatible.py`, `test_vllm_backend.py`).

## Modules

| Module | Responsibility |
|---|---|
| `backends/base.py` | `InferenceBackend` protocol: `generate_batch(chats) -> list[str]`. |
| `backends/vllm_backend.py` | `VLLMBackend` -- local/cluster GPU inference via `vllm.LLM`, OOM-safe batch-halving retry. `vllm`/`torch` are imported lazily, not a hard dependency of this package. |
| `backends/openai_compatible.py` | `OpenAICompatibleBackend` -- any OpenAI-Chat-Completions-compatible HTTP endpoint, `ThreadPoolExecutor` concurrency, optional `reasoning_model` flag for o1-style call params. |
| `models.py` | `resolve_model(shortcut)` -- vLLM `MODEL_SHORTCUTS`/`THINKING_MODELS` registry, extendable via `config/inference.yaml`'s `models:` block. |
| `extraction.py` | `clean_text`, `strip_thinking`, `extract_final_answer` -- backend-agnostic response post-processing. |
| `runner.py` | `load_prompts(dir)` + `run_inference(backend, questions, prompts, ...)` -- the shared prompt x question loop. |
| `judge.py` | LLM-as-judge CORRECT/INCORRECT grading for open-ended benchmarks (e.g. TimeMenatQA), same submit/status/process/sequential shape as annotation's batch module. |
| `reunite.py` | A small **helper toolkit**, not a pipeline -- `score_responses`/`build_outcomes_table` for turning results into outcome columns/tables in whatever shape you want. No CLI; see the worked example below. |

## Usage

### Local GPU inference (vLLM) -- **needs a GPU + `pip install vllm torch`**

```python
import random
from src.propensity.inference import load_prompts, run_inference, resolve_model
from src.propensity.inference.backends import VLLMBackend
from src.propensity.common import read_jsonl

resolved_model, is_thinking = resolve_model("llama33")
backend = VLLMBackend(resolved_model, tensor_parallel_size=2)  # loads the model into GPU memory

questions = list(read_jsonl("data/benchmarks/ready/RA/RA.jsonl"))
prompts = load_prompts("prompts/incitement/RA")

results = run_inference(backend, questions, prompts, is_thinking_model=is_thinking)
# {"0": {"total": 350, "responses": [...]}, "-1": {...}, ...}
```

### Any OpenAI-compatible HTTP endpoint -- **requires credentials, consumes credits**

```python
from src.propensity.common.llm_clients import get_azure_client, get_openai_client
from src.propensity.inference.backends import OpenAICompatibleBackend

# Azure OpenAI
client = get_azure_client()
backend = OpenAICompatibleBackend(client, model="gpt-4o", concurrency=10)

# Or a self-hosted / alternate OpenAI-compatible server (e.g. vLLM's own
# server mode, Ollama, OpenRouter) -- same backend class, just a different client:
client = get_openai_client(base_url="http://localhost:8000/v1")
backend = OpenAICompatibleBackend(client, model="my-local-model")

results = run_inference(backend, questions, prompts)
```

### CLI

```bash
# Local GPU
python scripts/run_inference.py --backend vllm --model llama33 \
  --input data/benchmarks/ready/RA/RA.jsonl --prompts-dir prompts/incitement/RA --gpus 2

# Azure
python scripts/run_inference.py --backend openai_compatible --model gpt-4o \
  --provider azure --input data/benchmarks/ready/RA/RA.jsonl --prompts-dir prompts/incitement/RA

# Self-hosted OpenAI-compatible server
python scripts/run_inference.py --backend openai_compatible --model my-local-model \
  --provider openai --base-url http://localhost:8000/v1 \
  --input data/benchmarks/ready/RA/RA.jsonl --prompts-dir prompts/incitement/RA
```

Writes a results JSON + summary CSV to `data/inference/raw/` (gitignored -- raw/regenerable, not a final artifact).

### LLM-as-judge grading (open-ended benchmarks) -- **requires credentials, consumes credits**

```python
from src.propensity.inference import load_results, judge_sequential
from src.propensity.common.llm_clients import get_azure_client

items = load_results("data/inference/raw/TimeMenatQA/gpt-4o__TimeMenatQA.json")["results"]["0"]["responses"]
client = get_azure_client()
graded = judge_sequential(items, client, model="gpt-4.1")
# each item gains judgment (bool) and raw_judgment_response
```

CLI, same submit/status/process/sequential shape as `annotate_benchmark.py`:

```bash
python scripts/judge_outcomes.py sequential \
  --results_filename data/inference/raw/TimeMenatQA/gpt-4o__TimeMenatQA.json \
  --persona 0 \
  --judge_output_path data/inference/raw/TimeMenatQA/gpt-4o__TimeMenatQA_judged.jsonl \
  --provider azure
```

### Reunite: turning results into an outcomes table -- no credentials needed

`reunite.py` is intentionally low-level -- it doesn't know your directory
layout or column-naming convention, only how to score one prompt's responses
and pivot many scored columns together. A worked example for one benchmark,
one model, across incitement levels:

```python
from src.propensity.inference import load_results, score_responses, build_outcomes_table, exact_match

levels = ["-3", "-2", "-1", "0", "1", "2", "3", "baseline"]
results = load_results("data/inference/raw/RA/gpt-4o__RA.json")

columns = {
    f"4o_RA_{level}_outcome": score_responses(results["results"][level]["responses"], is_correct=exact_match)
    for level in levels
}
outcomes_table = build_outcomes_table(columns)
outcomes_table.to_csv("data/inference/outcomes/RA/RA_complete.csv", index=False)
```

For open-ended benchmarks graded by `judge.py`, swap `is_correct=exact_match`
for `is_correct=judgment_field` (reads the `judgment` boolean `judge_sequential`/
`process_judge_batch` wrote onto each item).
