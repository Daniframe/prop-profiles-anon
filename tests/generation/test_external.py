from src.propensity.generation.spec import BenchmarkSpec
from src.propensity.generation.external import load_external_benchmark


def test_load_external_benchmark_renames_columns(tmp_path):
    csv_path = tmp_path / "source.csv"
    csv_path.write_text(
        "instance_id,prompt,groundtruth,benchmark\n"
        "q1,What is 2+2?,four,TimeQA\n"
        "q2,What year?,unanswerable,TimeQA\n"
    )
    spec = BenchmarkSpec(
        name="TimeMenatQA",
        kind="external",
        params={
            "source_path": str(csv_path),
            "column_map": {"instance_id": "question_id", "prompt": "question_text", "groundtruth": "correct_answer"},
            "extra_columns": ["benchmark"],
        },
    )

    records = load_external_benchmark(spec)

    assert records == [
        {"question_id": "q1", "question_text": "What is 2+2?", "correct_answer": "four", "benchmark": "TimeQA"},
        {"question_id": "q2", "question_text": "What year?", "correct_answer": "unanswerable", "benchmark": "TimeQA"},
    ]


def test_load_external_benchmark_without_extra_columns(tmp_path):
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("instance_id,prompt,groundtruth\nq1,Q?,A\n")
    spec = BenchmarkSpec(
        name="TimeMenatQA",
        kind="external",
        params={
            "source_path": str(csv_path),
            "column_map": {"instance_id": "question_id", "prompt": "question_text", "groundtruth": "correct_answer"},
        },
    )

    records = load_external_benchmark(spec)

    assert records == [{"question_id": "q1", "question_text": "Q?", "correct_answer": "A"}]
