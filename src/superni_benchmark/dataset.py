from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from superni_benchmark.config import DatasetConfig, PromptConfig


@dataclass(slots=True)
class BenchmarkExample:
    task_name: str
    example_id: str
    definition: str
    input_text: str
    references: list[str]
    prompt: str
    metadata: dict[str, Any]


def load_benchmark_examples(
    dataset_config: DatasetConfig,
    prompt_config: PromptConfig,
) -> list[BenchmarkExample]:
    dataset = load_dataset(
        dataset_config.hf_dataset,
        split=dataset_config.split,
        streaming=dataset_config.streaming,
    )
    if dataset_config.streaming and dataset_config.shuffle_buffer_size > 1:
        dataset = dataset.shuffle(
            seed=dataset_config.seed,
            buffer_size=dataset_config.shuffle_buffer_size,
        )

    task_counts: Counter[str] = Counter()
    selected_tasks: list[str] = []
    selected_task_set: set[str] = set()
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(dataset, start=1):
        if index > dataset_config.max_records_to_scan:
            break

        task_name = row["task_name"]
        if dataset_config.task_allowlist and task_name not in dataset_config.task_allowlist:
            continue
        if task_name in dataset_config.task_blocklist:
            continue

        if task_name not in selected_task_set:
            if len(selected_task_set) >= dataset_config.max_tasks:
                continue
            selected_task_set.add(task_name)
            selected_tasks.append(task_name)

        if task_counts[task_name] >= dataset_config.max_instances_per_task:
            if _is_collection_complete(
                task_counts,
                selected_tasks,
                dataset_config.max_tasks,
                dataset_config.max_instances_per_task,
            ):
                break
            continue

        references = _coerce_references(row.get("targets", []))
        if not references:
            continue

        examples.append(
            BenchmarkExample(
                task_name=task_name,
                example_id=row["id"],
                definition=row["definition"],
                input_text=row["inputs"],
                references=references,
                prompt=build_prompt(row, prompt_config),
                metadata={},
            )
        )
        task_counts[task_name] += 1

        if _is_collection_complete(
            task_counts,
            selected_tasks,
            dataset_config.max_tasks,
            dataset_config.max_instances_per_task,
        ):
            break

    return examples


def build_prompt(row: dict[str, Any], config: PromptConfig) -> str:
    sections = [
        "Task definition:",
        _truncate(row["definition"], config.max_definition_chars),
    ]

    positive_examples = _render_examples(row, config, prefix="pos", limit=config.num_positive_examples)
    negative_examples = _render_examples(row, config, prefix="neg", limit=config.num_negative_examples)

    if config.include_positive_examples and positive_examples:
        sections.append("Positive examples:")
        sections.extend(positive_examples)

    if config.include_negative_examples and negative_examples:
        sections.append("Negative examples:")
        sections.extend(negative_examples)

    sections.extend(
        [
            "Input:",
            _truncate(row["inputs"], config.max_input_chars),
            "Answer:",
        ]
    )
    return "\n\n".join(section for section in sections if section)


def _is_collection_complete(
    task_counts: Counter[str],
    selected_tasks: list[str],
    max_tasks: int,
    per_task_limit: int,
) -> bool:
    if len(selected_tasks) < max_tasks:
        return False
    return all(task_counts[task_name] >= per_task_limit for task_name in selected_tasks)


def _coerce_references(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _render_examples(
    row: dict[str, Any],
    config: PromptConfig,
    prefix: str,
    limit: int,
) -> list[str]:
    rendered: list[str] = []
    for index in range(limit):
        input_key = f"{prefix}_{index}_input"
        output_key = f"{prefix}_{index}_output"
        explanation_key = f"{prefix}_{index}_explanation"
        input_text = row.get(input_key)
        output_text = row.get(output_key)
        if not input_text or not output_text:
            continue
        block = [
            f"Example {len(rendered) + 1} input:",
            _truncate(str(input_text), config.max_example_chars),
            "Example output:",
            _truncate(str(output_text), config.max_example_chars),
        ]
        explanation = row.get(explanation_key)
        if explanation:
            block.extend(
                [
                    "Why this is a good answer:",
                    _truncate(str(explanation), config.max_example_chars),
                ]
            )
        rendered.append("\n".join(block))
    return rendered


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
