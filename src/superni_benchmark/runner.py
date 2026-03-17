from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from superni_benchmark.config import BenchmarkConfig
from superni_benchmark.dataset import BenchmarkExample, load_benchmark_examples
from superni_benchmark.metrics import score_prediction
from superni_benchmark.models import build_model_client


def run_benchmark(config: BenchmarkConfig) -> Path:
    output_dir = Path(config.output.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_benchmark_examples(config.dataset, config.prompt)
    if not examples:
        raise ValueError("No benchmark examples were collected. Adjust the dataset split or sampling settings.")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _config_to_dict(config),
        "num_examples": len(examples),
        "num_tasks": len({example.task_name for example in examples}),
    }
    _write_json(output_dir / "manifest.json", manifest)

    summary_rows: list[dict[str, object]] = []
    for model_config in config.models:
        if not model_config.enabled:
            continue
        client = build_model_client(model_config, config.generation, config.prompt)
        model_output_path = output_dir / f"{model_config.name}.jsonl"
        records = _run_or_resume_model(
            client=client,
            examples=examples,
            model_name=model_config.name,
            model_output_path=model_output_path,
            overwrite=config.output.overwrite,
            resume=config.output.resume,
        )
        summary_rows.append(_summarize_model(model_config.name, records))

    _write_json(output_dir / "summary.json", {"models": summary_rows})
    _write_markdown(output_dir / "summary.md", summary_rows)
    return output_dir


def _run_example(client, example: BenchmarkExample, model_name: str) -> dict[str, object]:
    response = client.generate(example.prompt)
    metrics = score_prediction(response.text, example.references)
    return {
        "model_name": model_name,
        "task_name": example.task_name,
        "example_id": example.example_id,
        "definition": example.definition,
        "input_text": example.input_text,
        "prompt": example.prompt,
        "references": example.references,
        "prediction": response.text,
        "metrics": metrics,
        "latency_seconds": response.latency_seconds,
        "attempts": response.attempts,
        "metadata": example.metadata,
        "raw_response": response.raw,
    }


def _run_or_resume_model(
    client,
    examples: list[BenchmarkExample],
    model_name: str,
    model_output_path: Path,
    overwrite: bool,
    resume: bool,
) -> list[dict[str, object]]:
    existing_records = {}
    if model_output_path.exists() and not overwrite and resume:
        existing_records = _load_existing_records(model_output_path)

    mode = "w" if overwrite or not model_output_path.exists() or not resume else "a"
    records: list[dict[str, object]] = []
    with model_output_path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc=model_name):
            existing = existing_records.get(example.example_id)
            if existing is not None:
                records.append(existing)
                continue

            record = _run_example(client, example, model_name)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return records


def _summarize_model(model_name: str, records: list[dict[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_task[record["task_name"]].append(record)

    metric_names = ["exact_match", "token_f1", "rouge_l"]
    overall = {
        metric: mean(float(record["metrics"][metric]) for record in records)
        for metric in metric_names
    }
    task_breakdown = []
    for task_name, task_records in sorted(by_task.items()):
        task_breakdown.append(
            {
                "task_name": task_name,
                "num_examples": len(task_records),
                **{
                    metric: mean(float(record["metrics"][metric]) for record in task_records)
                    for metric in metric_names
                },
            }
        )

    return {
        "model_name": model_name,
        "num_examples": len(records),
        "num_tasks": len(by_task),
        **overall,
        "avg_latency_seconds": mean(float(record["latency_seconds"]) for record in records),
        "task_breakdown": task_breakdown,
    }


def _config_to_dict(config: BenchmarkConfig) -> dict[str, object]:
    return asdict(config)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# SuperNI Benchmark Summary",
        "",
        "| Model | Examples | Tasks | EM | Token F1 | ROUGE-L | Avg latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_name} | {num_examples} | {num_tasks} | {exact_match:.4f} | {token_f1:.4f} | "
            "{rouge_l:.4f} | {avg_latency_seconds:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_existing_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["example_id"])] = record
    return records
