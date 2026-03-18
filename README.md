# beyond-superni

`beyond-superni` is a benchmark harness for measuring how remote LLMs perform on Super Natural Instructions (SuperNI) without fine-tuning.

The repo is built for comparative evaluation, not training. It samples SuperNI tasks, prompts each configured model through an API, scores the outputs with a shared metric pipeline, and writes reusable result artifacts for later analysis and plotting.

Documentation:

- [Benchmark metrics guide](docs/benchmark-metrics.md)

## What This Repository Is For

This repository is mainly for benchmarking SuperNI across different model providers and model snapshots.

It is designed to answer questions like:

- How strong are current remote models on SuperNI in a zero-shot setup?
- How much does matched same-task ICL help?
- Which models give the best quality / latency tradeoff on the same sampled benchmark slice?
- How do results change when you widen the model sweep or increase examples per task?

## Benchmark Design

The harness:

- loads `Muennighoff/natural-instructions` from Hugging Face
- samples a bounded subset of tasks and examples from a chosen split
- builds prompts from each task definition and input
- optionally injects matched few-shot examples from the same task
- calls remote models only
- scores predictions with normalized exact match, token F1, and ROUGE-L
- records latency and raw response payloads
- writes per-example JSONL plus aggregate summaries and plots

This is intentionally not a SuperNI fine-tuning repo. The point is to treat SuperNI as an instruction-following benchmark for already-trained models.

## Supported Model Backends

- OpenAI via the official `openai` SDK
- Hugging Face Inference Providers via `huggingface_hub.InferenceClient`

The current configs are set up for OpenAI and Hyperbolic-routed Hugging Face models, but the YAML format is general enough to swap model IDs, providers, and credentials.

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ".[dev,viz]"  # optional, for plotting
```

Set credentials:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

If you want to use a provider-specific key instead of `HF_TOKEN`, set `api_key_env` on the model entry in [configs/default.yaml](configs/default.yaml) and export that variable.

## Running Benchmarks

Run the default benchmark:

```bash
superni-benchmark run --config configs/default.yaml
```

Run the matched ICL benchmark:

```bash
superni-benchmark run --config configs/icl.yaml
```

Run a wider, more expensive sweep:

```bash
superni-benchmark run --config configs/wide.yaml
```

## Included Benchmark Presets

- [configs/default.yaml](configs/default.yaml): default cross-provider benchmark, 25 tasks x 3 examples
- [configs/default_10.yaml](configs/default_10.yaml): smaller smoke-test style run, 10 tasks x 3 examples
- [configs/default_25_examples.yaml](configs/default_25_examples.yaml): larger zero-shot benchmark, 25 tasks x 25 examples
- [configs/icl.yaml](configs/icl.yaml): matched ICL run with 3 same-task support examples
- [configs/icl_big.yaml](configs/icl_big.yaml): larger matched ICL run with 5 support examples
- [configs/wide.yaml](configs/wide.yaml): broader model sweep at 25 tasks x 25 examples
- [configs/high_reasoning_effort_25_examples.yaml](configs/high_reasoning_effort_25_examples.yaml): single-model high-reasoning run
- [configs/hyperbolic_only.yaml](configs/hyperbolic_only.yaml): Hyperbolic-backed subset only
- [configs/qwq_only.yaml](configs/qwq_only.yaml): single-model QwQ run
- [configs/kimi_only.yaml](configs/kimi_only.yaml): single-model Kimi run

## Outputs

Each run writes a results directory such as `results/default/` with:

- `manifest.json`: timestamp, config snapshot, task count, example count
- `<model>.jsonl`: one record per evaluated example
- `summary.json`: aggregate metrics and per-task breakdowns
- `summary.md`: quick human-readable summary table

If `output.resume: true` is enabled, rerunning the same benchmark reuses existing model outputs and only requests missing examples. If prompts changed, the harness falls back to a fresh write for that model output file.

## Plotting

Generate figures from an existing results directory:

```bash
superni-benchmark plot --results results/default
superni-benchmark plot --results results/wide --heatmap-metric rouge_l --top-tasks 15 --format svg
```

By default, plots are written to `results/<run>/figures/` and include:

- model score overview
- latency vs quality scatter
- per-task heatmap when task breakdowns are available

## How Sampling Works

The benchmark does not evaluate the full dataset by default. Instead, configs cap:

- `dataset.max_tasks`
- `dataset.max_instances_per_task`
- `dataset.max_records_to_scan`

This keeps API cost manageable while preserving a comparable benchmark slice across models.

The loader supports:

- streaming dataset reads
- buffered shuffle for approximate randomization
- task allowlists and blocklists
- bounded task-balanced sampling

Total API calls are roughly:

```text
enabled_models * max_tasks * max_instances_per_task
```

ICL increases prompt size, but not the number of model calls.

## How ICL Works

When `icl.enabled: true`, the harness collects non-overlapping same-task support examples and injects them ahead of the evaluated input.

By default, [configs/icl.yaml](configs/icl.yaml) uses `icl.source_split: test` and excludes support examples whose example ID or normalized input overlaps with the evaluation set. If you want a stricter separation, point `icl.source_split` at another split such as `train`.

## Metrics

Each prediction is scored against one or more references with:

- exact match
- token F1
- ROUGE-L

Latency is reported separately as `avg_latency_seconds`.

See the full metric behavior in [docs/benchmark-metrics.md](docs/benchmark-metrics.md).

## Customizing a Benchmark

The main knobs live in the YAML config:

- `dataset`: sampling limits, split, allowlist/blocklist, streaming behavior
- `prompt`: prompt construction and truncation
- `icl`: matched few-shot support examples
- `generation`: retries, timeout, output cap, reasoning effort
- `models`: backend, model ID, provider, API key env var, extra request body
- `output`: result directory, resume behavior, overwrite policy

For reproducible comparisons, prefer pinned model snapshots over floating aliases.

## Notes

- SuperNI is heterogeneous, so automatic metrics are useful but imperfect.
- Overall scores are averaged across examples, not macro-averaged equally across tasks.
- The repo benchmarks the flattened Hugging Face dataset representation directly.
- Plotting requires the optional `viz` dependencies.
