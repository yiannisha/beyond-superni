# beyond-superni

Self-contained benchmark harness for evaluating remote LLMs on the [Super Natural Instructions dataset](https://huggingface.co/datasets/Muennighoff/natural-instructions) without fine-tuning.

Documentation:

- [Benchmark metrics guide](docs/benchmark-metrics.md)

## What this does

- Downloads `Muennighoff/natural-instructions` from Hugging Face.
- Streams a fixed subset of tasks and instances per task from the held-out split.
- Builds instruction-following prompts from each task definition and optional examples.
- Supports an ICL benchmark mode that prepends random same-task few-shot examples from a disjoint support split.
- Queries remote models only:
  - OpenAI via the official `openai` API.
  - Hugging Face Inference Providers via `huggingface_hub.InferenceClient`, with `provider=hyperbolic` by default for Llama and Qwen.
- Scores predictions against task references with normalized exact match, token F1, and ROUGE-L.
- Writes per-example outputs plus aggregate summaries to `results/...`.

## Why this setup

SuperNI is primarily used as a continual-learning or per-task SFT resource. This experiment deliberately does **no fine-tuning**. It treats SuperNI as a zero-shot benchmark for current frontier models to measure how difficult the benchmark still is for strong base models.

The repo also supports a matched ICL mode. In that mode, each evaluation prompt is augmented with `icl.num_examples` random support examples from the same task. By default, the support examples can come from the benchmark split itself as long as they are disjoint from the evaluated inputs. If you want a stricter setup, set `icl.source_split: train`. In either case, the harness excludes any support example whose `id` or normalized input overlaps an evaluation example.

The Hugging Face dataset artifact exposed here is a flattened instance-level view with the fields `task_name`, `definition`, `inputs`, and `targets`. The harness benchmarks that view directly instead of reconstructing the original task JSON files.

The default config is pinned to current model snapshots as of **March 17, 2026**:

- OpenAI `gpt-5.4-2026-03-05`
- OpenAI `gpt-5-mini-2025-08-07`
- Hyperbolic `meta-llama/Llama-3.3-70B-Instruct`
- Hyperbolic `Qwen/Qwen3-235B-A22B-Instruct-2507`

Adjust `configs/default.yaml` if you want different snapshots, smaller models, or cheaper runs.

There is also a broader preset in `configs/wide.yaml` that expands the open-source sweep with additional Llama, Qwen, QwQ, and Kimi models and increases the sample count to 10 examples per task.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# optional, for figure rendering:
uv pip install -e ".[dev,viz]"
```

Set credentials:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

For Hugging Face Inference Providers, `HF_TOKEN` works when you route through Hugging Face. If you want to use a provider-specific key instead, set `api_key_env` on the model entry in [configs/default.yaml](/workspace/beyond-superni/configs/default.yaml) and export that variable instead, for example `HYPERBOLIC_API_KEY`.

## Run

```bash
superni-benchmark run --config configs/default.yaml
# or, for the matched ICL variant:
superni-benchmark run --config configs/icl.yaml
# or, for a wider/longer run:
superni-benchmark run --config configs/wide.yaml
```

Artifacts:

- `results/default/manifest.json`
- `results/default/*.jsonl`
- `results/default/summary.json`
- `results/default/summary.md`

If a run stops mid-way, rerun the same command. Existing per-model JSONL files are reused and only missing examples are requested again.

## Plot

The repo now includes a reusable plotting command that reads any `results/*` directory. It prefers `summary.json` when present and falls back to `summary.md`.

```bash
superni-benchmark plot --results results/default
# choose a different metric for the heatmap and write SVGs:
superni-benchmark plot --results results/wide --heatmap-metric rouge_l --top-tasks 15 --format svg
```

By default, figures are written to `results/<run>/figures/` and include:

- grouped score bars across models
- latency-vs-quality scatter
- per-task heatmap for the selected metric

## Cost control

The full dataset is too large for practical API benchmarking. The default config uses the held-out `test` split and:

- 25 tasks
- 3 instances per task
- 75 total model calls per model

That is enough to get a first pass on relative difficulty without turning the run into a large API bill. Increase `max_tasks` and `max_instances_per_task` gradually.

To increase examples per task, raise `dataset.max_instances_per_task`. If you do that, also raise `dataset.max_records_to_scan` so the streaming sampler has enough rows to fill every task quota. Total calls are roughly `enabled_models * max_tasks * max_instances_per_task`.

For ICL runs, the number of API calls does not change, but prompt size does. The default [configs/icl.yaml](/workspace/beyond-superni/configs/icl.yaml) uses `icl.source_split: test` and excludes any overlap with benchmarked inputs. If you want support examples from a different split, set `icl.source_split: train`.

## Notes

- SuperNI contains heterogeneous tasks, so automatic scoring is necessarily approximate.
- The harness uses multiple reference strings when available and takes the best match per example.
- The loader uses streaming plus buffered shuffle so it does not need to materialize the full 6M+ row training split.
- For reproducibility, use snapshot model IDs instead of floating aliases whenever possible.
