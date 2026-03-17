from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

METRIC_CHOICES = ("exact_match", "token_f1", "rouge_l")
_MARKDOWN_FIELD_MAP = {
    "model": "model_name",
    "examples": "num_examples",
    "tasks": "num_tasks",
    "em": "exact_match",
    "token f1": "token_f1",
    "rouge-l": "rouge_l",
    "avg latency (s)": "avg_latency_seconds",
}


@dataclass(slots=True)
class TaskSummary:
    task_name: str
    num_examples: int
    exact_match: float
    token_f1: float
    rouge_l: float


@dataclass(slots=True)
class ModelSummary:
    model_name: str
    num_examples: int
    num_tasks: int
    exact_match: float
    token_f1: float
    rouge_l: float
    avg_latency_seconds: float
    task_breakdown: list[TaskSummary] = field(default_factory=list)

    def metric_value(self, metric_name: str) -> float:
        return float(getattr(self, metric_name))


@dataclass(slots=True)
class BenchmarkSummary:
    source_path: Path
    base_dir: Path
    models: list[ModelSummary]
    manifest: dict[str, object] | None = None


def load_benchmark_summary(source: str | Path) -> BenchmarkSummary:
    source_path = Path(source).expanduser().resolve()
    summary_path, base_dir = _resolve_summary_path(source_path)
    manifest = _load_manifest(base_dir)

    if summary_path.suffix == ".json":
        models = _load_models_from_json(summary_path)
    elif summary_path.suffix == ".md":
        models = _load_models_from_markdown(summary_path)
    else:
        raise ValueError(f"Unsupported summary file type: {summary_path}")

    if not models:
        raise ValueError(f"No models found in {summary_path}")

    return BenchmarkSummary(
        source_path=summary_path,
        base_dir=base_dir,
        models=models,
        manifest=manifest,
    )


def render_benchmark_figures(
    source: str | Path,
    output_dir: str | Path | None = None,
    heatmap_metric: str = "token_f1",
    top_tasks: int = 20,
    image_format: str = "png",
) -> list[Path]:
    if heatmap_metric not in METRIC_CHOICES:
        choices = ", ".join(METRIC_CHOICES)
        raise ValueError(f"Unsupported metric '{heatmap_metric}'. Choose from: {choices}")
    if top_tasks <= 0:
        raise ValueError("top_tasks must be positive.")

    summary = load_benchmark_summary(source)
    destination = Path(output_dir).expanduser().resolve() if output_dir else summary.base_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    plt, mpl = _load_matplotlib()
    _configure_style(mpl)

    written_paths = [
        _plot_model_scores(summary, destination, image_format, plt),
        _plot_latency_vs_quality(summary, destination, image_format, plt),
    ]

    if any(model.task_breakdown for model in summary.models):
        written_paths.append(
            _plot_task_heatmap(summary, destination, heatmap_metric, top_tasks, image_format, plt, mpl)
        )

    plt.close("all")
    return written_paths


def _resolve_summary_path(source_path: Path) -> tuple[Path, Path]:
    if source_path.is_dir():
        json_path = source_path / "summary.json"
        markdown_path = source_path / "summary.md"
        if json_path.exists():
            return json_path, source_path
        if markdown_path.exists():
            return markdown_path, source_path
        raise FileNotFoundError(f"No summary.json or summary.md found in {source_path}")

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.name in {"summary.json", "summary.md"}:
        return source_path, source_path.parent

    raise ValueError("Source must be a results directory, summary.json, or summary.md.")


def _load_manifest(base_dir: Path) -> dict[str, object] | None:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_models_from_json(path: Path) -> list[ModelSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected 'models' to be a list in {path}")
    return [_model_from_dict(row) for row in rows]


def _load_models_from_markdown(path: Path) -> list[ModelSummary]:
    models: list[ModelSummary] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        return models

    headers = [_normalize_markdown_header(cell) for cell in table_lines[0].strip("|").split("|")]
    fields = [_MARKDOWN_FIELD_MAP.get(header, header) for header in headers]

    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(fields):
            continue
        row = dict(zip(fields, cells, strict=True))
        models.append(
            ModelSummary(
                model_name=row["model_name"],
                num_examples=int(row["num_examples"]),
                num_tasks=int(row["num_tasks"]),
                exact_match=float(row["exact_match"]),
                token_f1=float(row["token_f1"]),
                rouge_l=float(row["rouge_l"]),
                avg_latency_seconds=float(row["avg_latency_seconds"]),
            )
        )
    return models


def _normalize_markdown_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _model_from_dict(payload: dict[str, object]) -> ModelSummary:
    breakdown_rows = payload.get("task_breakdown", [])
    if not isinstance(breakdown_rows, list):
        breakdown_rows = []
    task_breakdown = [_task_from_dict(row) for row in breakdown_rows]

    return ModelSummary(
        model_name=str(payload["model_name"]),
        num_examples=int(payload["num_examples"]),
        num_tasks=int(payload["num_tasks"]),
        exact_match=float(payload["exact_match"]),
        token_f1=float(payload["token_f1"]),
        rouge_l=float(payload["rouge_l"]),
        avg_latency_seconds=float(payload["avg_latency_seconds"]),
        task_breakdown=task_breakdown,
    )


def _task_from_dict(payload: dict[str, object]) -> TaskSummary:
    return TaskSummary(
        task_name=str(payload["task_name"]),
        num_examples=int(payload["num_examples"]),
        exact_match=float(payload["exact_match"]),
        token_f1=float(payload["token_f1"]),
        rouge_l=float(payload["rouge_l"]),
    )


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install it with `uv pip install -e \".[viz]\"`."
        ) from exc

    return plt, matplotlib


def _configure_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fcfbf7",
            "savefig.facecolor": "#fcfbf7",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1f2933",
            "xtick.color": "#52606d",
            "ytick.color": "#52606d",
        }
    )


def _plot_model_scores(summary: BenchmarkSummary, output_dir: Path, image_format: str, plt) -> Path:
    metric_order = ["exact_match", "token_f1", "rouge_l"]
    labels = {
        "exact_match": "Exact match",
        "token_f1": "Token F1",
        "rouge_l": "ROUGE-L",
    }
    colors = {
        "exact_match": "#d1495b",
        "token_f1": "#edae49",
        "rouge_l": "#00798c",
    }

    models = sorted(summary.models, key=lambda model: (-model.token_f1, model.avg_latency_seconds, model.model_name))
    y_positions = list(range(len(models)))
    bar_height = 0.22

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(models) * 0.9)))
    for index, metric_name in enumerate(metric_order):
        offset = (index - 1) * bar_height
        positions = [position + offset for position in y_positions]
        values = [model.metric_value(metric_name) for model in models]
        ax.barh(
            positions,
            values,
            height=bar_height,
            color=colors[metric_name],
            label=labels[metric_name],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_display_model_name(model.model_name) for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title(f"Model Score Overview\n{_build_run_subtitle(summary)}")
    ax.legend(loc="lower right", frameon=False, ncol=3)

    for position, model in zip(y_positions, models, strict=True):
        ax.text(
            min(model.token_f1 + 0.015, 0.99),
            position,
            f"{model.token_f1:.3f}",
            va="center",
            ha="left",
            color="#102a43",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = output_dir / f"model_scores.{image_format}"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return output_path


def _plot_latency_vs_quality(summary: BenchmarkSummary, output_dir: Path, image_format: str, plt) -> Path:
    models = sorted(summary.models, key=lambda model: (-model.token_f1, model.avg_latency_seconds, model.model_name))
    colors = ["#00798c", "#d1495b", "#edae49", "#30638e", "#8f5c2c", "#5f0f40"]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.set_title(f"Latency vs Token F1\n{_build_run_subtitle(summary)}")
    ax.set_xlabel("Average latency (seconds)")
    ax.set_ylabel("Token F1")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)

    for index, model in enumerate(models):
        color = colors[index % len(colors)]
        size = 90 + (model.num_examples * 2.5)
        ax.scatter(
            model.avg_latency_seconds,
            model.token_f1,
            s=size,
            color=color,
            edgecolor="#102a43",
            linewidth=0.8,
            alpha=0.92,
        )
        ax.annotate(
            _display_model_name(model.model_name),
            (model.avg_latency_seconds, model.token_f1),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="#102a43",
        )

    fig.tight_layout()
    output_path = output_dir / f"latency_vs_quality.{image_format}"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return output_path


def _plot_task_heatmap(
    summary: BenchmarkSummary,
    output_dir: Path,
    metric_name: str,
    top_tasks: int,
    image_format: str,
    plt,
    mpl,
) -> Path:
    selected_models = sorted(summary.models, key=lambda model: (-model.metric_value(metric_name), model.model_name))
    matrix, task_names = _build_task_matrix(selected_models, metric_name, top_tasks)

    fig_height = max(6.0, len(task_names) * 0.4)
    fig, ax = plt.subplots(figsize=(max(8.0, len(selected_models) * 1.7), fig_height))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("superni_heatmap", ["#b23a48", "#f6bd60", "#2a9d8f"])
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)

    ax.set_title(f"Task Heatmap ({_pretty_metric_name(metric_name)})\nHardest tasks first")
    ax.set_xticks(list(range(len(selected_models))))
    ax.set_xticklabels([_display_model_name(model.model_name) for model in selected_models], rotation=25, ha="right")
    ax.set_yticks(list(range(len(task_names))))
    ax.set_yticklabels([_display_task_name(task_name) for task_name in task_names])

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            is_missing = isinstance(value, float) and math.isnan(value)
            label = "--" if is_missing else f"{value:.2f}"
            text_color = "#102a43" if is_missing or value >= 0.38 else "#fcfbf7"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.ax.set_ylabel(_pretty_metric_name(metric_name), rotation=90, va="bottom")

    fig.tight_layout()
    output_path = output_dir / f"task_heatmap_{metric_name}.{image_format}"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return output_path


def _build_task_matrix(
    models: list[ModelSummary],
    metric_name: str,
    top_tasks: int,
) -> tuple[list[list[float]], list[str]]:
    rows_by_task: dict[str, dict[str, float]] = {}
    for model in models:
        for task in model.task_breakdown:
            rows_by_task.setdefault(task.task_name, {})[model.model_name] = float(getattr(task, metric_name))

    ranked_tasks = sorted(
        rows_by_task,
        key=lambda task_name: (
            mean(rows_by_task[task_name].values()),
            -_score_spread(rows_by_task[task_name].values()),
            task_name,
        ),
    )[:top_tasks]

    matrix = [
        [rows_by_task[task_name].get(model.model_name, math.nan) for model in models]
        for task_name in ranked_tasks
    ]
    return matrix, ranked_tasks


def _score_spread(values) -> float:
    numeric_values = list(values)
    return max(numeric_values) - min(numeric_values)


def _build_run_subtitle(summary: BenchmarkSummary) -> str:
    examples = sorted({model.num_examples for model in summary.models})
    tasks = sorted({model.num_tasks for model in summary.models})
    run_name = summary.base_dir.name
    example_text = _compact_count("example", examples[0]) if len(examples) == 1 else "mixed examples"
    task_text = _compact_count("task", tasks[0]) if len(tasks) == 1 else "mixed tasks"
    return f"{run_name} - {len(summary.models)} models - {example_text}/model - {task_text}/model"


def _compact_count(noun: str, value: int) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {noun}{suffix}"


def _display_model_name(model_name: str) -> str:
    return model_name.replace("_", " ")


def _display_task_name(task_name: str) -> str:
    cleaned = re.sub(r"^task\d+_?", "", task_name)
    return cleaned.replace("_", " ")


def _pretty_metric_name(metric_name: str) -> str:
    return {
        "exact_match": "Exact match",
        "token_f1": "Token F1",
        "rouge_l": "ROUGE-L",
    }[metric_name]
