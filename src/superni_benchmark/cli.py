from __future__ import annotations

import argparse

from superni_benchmark.config import BenchmarkConfig
from superni_benchmark.plotting import METRIC_CHOICES, render_benchmark_figures
from superni_benchmark.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark remote LLMs on SuperNI without fine-tuning.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark.")
    run_parser.add_argument("--config", required=True, help="Path to a YAML benchmark config.")

    plot_parser = subparsers.add_parser("plot", help="Render figures from an existing results directory.")
    plot_parser.add_argument(
        "--results",
        required=True,
        help="Path to a results directory, summary.json, or summary.md.",
    )
    plot_parser.add_argument(
        "--output-dir",
        help="Directory where figures will be written. Defaults to <results>/figures.",
    )
    plot_parser.add_argument(
        "--heatmap-metric",
        choices=METRIC_CHOICES,
        default="token_f1",
        help="Metric to visualize in the task heatmap.",
    )
    plot_parser.add_argument(
        "--top-tasks",
        type=int,
        default=20,
        help="Number of tasks to include in the heatmap, ranked by difficulty.",
    )
    plot_parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Image format for generated figures.",
    )
    args = parser.parse_args()

    if args.command == "run":
        config = BenchmarkConfig.from_file(args.config)
        output_dir = run_benchmark(config)
        print(output_dir)
    elif args.command == "plot":
        try:
            output_paths = render_benchmark_figures(
                source=args.results,
                output_dir=args.output_dir,
                heatmap_metric=args.heatmap_metric,
                top_tasks=args.top_tasks,
                image_format=args.format,
            )
        except RuntimeError as exc:
            parser.exit(1, f"{exc}\n")
        for path in output_paths:
            print(path)


if __name__ == "__main__":
    main()
