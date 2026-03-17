from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    hf_dataset: str = "Muennighoff/natural-instructions"
    split: str = "test"
    max_tasks: int = 25
    max_instances_per_task: int = 3
    seed: int = 7
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    max_records_to_scan: int = 100_000
    task_allowlist: list[str] = field(default_factory=list)
    task_blocklist: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptConfig:
    include_positive_examples: bool = False
    include_negative_examples: bool = False
    num_positive_examples: int = 0
    num_negative_examples: int = 0
    max_definition_chars: int = 4000
    max_input_chars: int = 8000
    max_example_chars: int = 1500
    system_prompt: str = (
        "You are answering one Natural Instructions task. "
        "Return only the task answer with no explanation, no chain-of-thought, "
        "and no extra labels."
    )


@dataclass(slots=True)
class ICLConfig:
    enabled: bool = False
    num_examples: int = 0
    source_split: str | None = None
    max_records_to_scan: int = 100_000


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.0
    max_output_tokens: int = 256
    reasoning_effort: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 4


@dataclass(slots=True)
class ModelConfig:
    name: str
    backend: str
    model: str
    provider: str | None = None
    api_key_env: str | None = None
    enabled: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "results/default"
    resume: bool = True
    overwrite: bool = False


@dataclass(slots=True)
class BenchmarkConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    models: list[ModelConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError("Benchmark config must be a mapping.")

        return cls(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            prompt=PromptConfig(**raw.get("prompt", {})),
            icl=ICLConfig(**raw.get("icl", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            output=OutputConfig(**raw.get("output", {})),
            models=[ModelConfig(**entry) for entry in raw.get("models", [])],
        )
