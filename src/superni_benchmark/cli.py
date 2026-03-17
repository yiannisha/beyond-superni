from __future__ import annotations

import argparse

from superni_benchmark.config import BenchmarkConfig
from superni_benchmark.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark remote LLMs on SuperNI without fine-tuning.")
    parser.add_argument("command", choices=["run"], help="Command to execute.")
    parser.add_argument("--config", required=True, help="Path to a YAML benchmark config.")
    args = parser.parse_args()

    if args.command == "run":
        config = BenchmarkConfig.from_file(args.config)
        output_dir = run_benchmark(config)
        print(output_dir)


if __name__ == "__main__":
    main()
