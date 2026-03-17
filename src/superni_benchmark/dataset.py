from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from superni_benchmark.config import DatasetConfig, ICLConfig, PromptConfig


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


def load_icl_examples_by_task(
    dataset_config: DatasetConfig,
    icl_config: ICLConfig,
    evaluation_examples: list[BenchmarkExample],
) -> dict[str, list[BenchmarkExample]]:
    if not icl_config.enabled or icl_config.num_examples <= 0:
        return {}

    tasks = {example.task_name for example in evaluation_examples}
    if not tasks:
        return {}

    excluded_ids = {example.example_id for example in evaluation_examples}
    excluded_inputs_by_task: dict[str, set[str]] = {}
    for example in evaluation_examples:
        excluded_inputs_by_task.setdefault(example.task_name, set()).add(_normalize_input(example.input_text))

    source_split = icl_config.source_split or dataset_config.split
    dataset = load_dataset(
        dataset_config.hf_dataset,
        split=source_split,
        streaming=dataset_config.streaming,
    )
    if dataset_config.streaming and dataset_config.shuffle_buffer_size > 1:
        dataset = dataset.shuffle(
            seed=dataset_config.seed + 1,
            buffer_size=dataset_config.shuffle_buffer_size,
        )

    examples_by_task: dict[str, list[BenchmarkExample]] = {task_name: [] for task_name in tasks}
    for index, row in enumerate(dataset, start=1):
        if index > icl_config.max_records_to_scan:
            break

        task_name = row["task_name"]
        if task_name not in tasks:
            continue
        if len(examples_by_task[task_name]) >= icl_config.num_examples:
            if _all_icl_collected(examples_by_task, icl_config.num_examples):
                break
            continue

        example_id = str(row["id"])
        input_text = str(row["inputs"])
        if example_id in excluded_ids:
            continue
        if _normalize_input(input_text) in excluded_inputs_by_task.get(task_name, set()):
            continue

        references = _coerce_references(row.get("targets", []))
        if not references:
            continue

        examples_by_task[task_name].append(
            BenchmarkExample(
                task_name=task_name,
                example_id=example_id,
                definition=str(row["definition"]),
                input_text=input_text,
                references=references,
                prompt="",
                metadata={"icl_source_split": source_split},
            )
        )

        if _all_icl_collected(examples_by_task, icl_config.num_examples):
            break

    missing_tasks = sorted(
        task_name
        for task_name, task_examples in examples_by_task.items()
        if len(task_examples) < icl_config.num_examples
    )
    if missing_tasks:
        raise ValueError(
            "Unable to collect enough non-overlapping ICL examples for tasks: "
            + ", ".join(missing_tasks)
            + f". Increase icl.max_records_to_scan or choose a different icl.source_split (current: {source_split})."
        )

    return examples_by_task


def inject_icl_prompts(
    evaluation_examples: list[BenchmarkExample],
    icl_examples_by_task: dict[str, list[BenchmarkExample]],
    icl_config: ICLConfig,
) -> list[BenchmarkExample]:
    if not icl_config.enabled or icl_config.num_examples <= 0:
        return evaluation_examples

    updated_examples: list[BenchmarkExample] = []
    for example in evaluation_examples:
        icl_examples = icl_examples_by_task.get(example.task_name, [])
        source_split = (
            str(icl_examples[0].metadata.get("icl_source_split"))
            if icl_examples
            else (icl_config.source_split or "")
        )
        prompt = _inject_icl_section(example.prompt, icl_examples)
        metadata = dict(example.metadata)
        metadata["icl_example_ids"] = [icl_example.example_id for icl_example in icl_examples]
        metadata["icl_num_examples"] = len(icl_examples)
        metadata["icl_source_split"] = source_split
        updated_examples.append(
            BenchmarkExample(
                task_name=example.task_name,
                example_id=example.example_id,
                definition=example.definition,
                input_text=example.input_text,
                references=example.references,
                prompt=prompt,
                metadata=metadata,
            )
        )
    return updated_examples


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


def _all_icl_collected(examples_by_task: dict[str, list[BenchmarkExample]], per_task_limit: int) -> bool:
    return all(len(task_examples) >= per_task_limit for task_examples in examples_by_task.values())


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


def _normalize_input(text: str) -> str:
    return " ".join(text.split()).strip()


def _inject_icl_section(prompt: str, icl_examples: list[BenchmarkExample]) -> str:
    if not icl_examples:
        return prompt

    marker = "\n\nInput:\n\n"
    insertion_point = prompt.rfind(marker)
    if insertion_point == -1:
        raise ValueError("Prompt is missing the final input section required for ICL injection.")

    blocks = []
    for index, example in enumerate(icl_examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Few-shot example {index} input:",
                    example.input_text,
                    "Few-shot example output:",
                    example.references[0],
                ]
            )
        )

    icl_section = "\n\nFew-shot examples:\n\n" + "\n\n".join(blocks)
    return prompt[:insertion_point] + icl_section + prompt[insertion_point:]
