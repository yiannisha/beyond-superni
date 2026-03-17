# Benchmark Metrics Guide

This document explains the metrics used by the SuperNI benchmark harness to evaluate model outputs. The benchmark applies the same scoring pipeline to every configured model, regardless of provider, and reports quality and latency separately rather than collapsing them into a single score.

The implementation lives in `src/superni_benchmark/metrics.py`, and aggregate reporting is handled in `src/superni_benchmark/runner.py`.

## Evaluation pipeline

For each benchmark example, the harness:

1. Builds a prompt from the SuperNI task definition and input.
2. Sends that prompt to a configured remote model.
3. Collects one prediction string and response metadata.
4. Scores the prediction against one or more reference answers.
5. Stores per-example results in JSONL and aggregates them into summary files.

Each record contains:

- `prediction`
- `references`
- `metrics.exact_match`
- `metrics.token_f1`
- `metrics.rouge_l`
- `latency_seconds`
- `attempts`

The reported summary table includes:

- `EM`
- `Token F1`
- `ROUGE-L`
- `Avg latency (s)`

There is no weighted composite benchmark score in the current implementation.

## Text normalization

All three text-quality metrics share the same normalization step before scoring.

Normalization rules:

- Trim leading and trailing whitespace
- Convert to lowercase
- Replace non-alphanumeric punctuation with spaces
- Collapse repeated whitespace into a single space

In code, punctuation is removed with the regex `[^0-9a-zA-Z\s]`, then whitespace is collapsed with `\s+`.

Implications:

- `Yes!` and `yes` are treated as identical
- `New   York` and `new york` are treated as identical
- `e-mail` becomes `e mail`
- `don't` becomes `don t`

This makes the metrics robust to formatting noise, but it also means punctuation-sensitive distinctions are ignored.

## Exact match

`exact_match` is the strictest metric.

Definition:

```text
exact_match = 1.0 if normalized prediction exactly equals any normalized reference
exact_match = 0.0 otherwise
```

How it works:

- The model prediction is normalized once.
- Every reference string is normalized.
- If any normalized reference exactly matches the normalized prediction, the score is `1.0`.
- Otherwise, the score is `0.0`.

Properties:

- Binary only
- No partial credit
- Case-insensitive after normalization
- Punctuation-insensitive after normalization
- Multiple references are handled with best-match semantics

When it is most useful:

- Short-form classification outputs
- Yes/no tasks
- Tasks with one canonical answer form

Failure mode:

A semantically correct paraphrase still receives `0.0` if it is not textually identical after normalization.

Example:

- Prediction: `Yes!`
- Reference: `yes`
- Score: `1.0`

Counterexample:

- Prediction: `yes, it is`
- Reference: `yes`
- Score: `0.0`

## Token F1

`token_f1` gives partial credit based on token overlap and ignores token order.

Definition:

```text
overlap = size of the multiset intersection between prediction tokens and reference tokens
precision = overlap / number of prediction tokens
recall = overlap / number of reference tokens
token_f1 = 2 * precision * recall / (precision + recall)
```

The benchmark computes this score against every reference and keeps the maximum value.

Tokenization behavior:

- Tokens are produced by splitting normalized text on whitespace
- Token counts matter because overlap uses `collections.Counter`
- Repeated tokens can increase or limit overlap depending on both sides

Properties:

- Rewards correct content words even when formatting differs
- Penalizes extra words through precision
- Penalizes missing words through recall
- Ignores token order entirely

Edge cases:

- If both the prediction and references are empty at the top level, the score is `1.0`
- If the prediction is empty and any reference is also empty, the score is `1.0`
- If either side is empty and the other is not, that reference contributes no positive score
- If there is no token overlap, the score is `0.0`

Example:

- Prediction: `the red fox`
- Reference: `red fox`
- Overlap: `2`
- Precision: `2/3`
- Recall: `2/2`
- F1: `0.8`

Interpretation:

High Token F1 with low Exact Match usually means the model captured the right information but not the exact phrasing.

## ROUGE-L

`rouge_l` gives partial credit based on the longest common subsequence, so it captures overlap and sequence order.

Definition:

```text
LCS = longest common subsequence length between prediction tokens and reference tokens
precision = LCS / number of prediction tokens
recall = LCS / number of reference tokens
rouge_l = 2 * precision * recall / (precision + recall)
```

As with Token F1, the benchmark computes ROUGE-L against every reference and keeps the maximum value.

Implementation details:

- It uses dynamic programming to compute LCS
- Tokens do not have to be contiguous
- Tokens do have to appear in the same order

Properties:

- More forgiving than Exact Match
- More order-sensitive than Token F1
- Useful for longer answers where answer structure matters

Edge cases:

- If both the prediction and references are empty at the top level, the score is `1.0`
- If the prediction is empty and any reference is also empty, the score is `1.0`
- If either side is empty and the other is not, that reference contributes no positive score
- If LCS is zero, the score is `0.0`

Example:

- Prediction: `red fox jumps`
- Reference: `fox red jumps`

This pair can still achieve strong Token F1 because the same tokens are present, but ROUGE-L drops because the ordering differs.

Interpretation:

High ROUGE-L with moderate Exact Match often indicates that the model produced a structurally similar answer but not the exact expected string.

## Multiple references

SuperNI tasks can provide multiple acceptable targets. The harness handles this by scoring a prediction against every reference and taking the best score for each metric.

That means:

- `exact_match` succeeds if any reference matches exactly after normalization
- `token_f1` reports the maximum overlap-based F1 across references
- `rouge_l` reports the maximum LCS-based score across references

This is important for instruction-following tasks where multiple wordings may be valid.

## Latency

The benchmark also records `latency_seconds` for each response.

What it measures:

- Wall-clock time spent inside the model generation call

How it is reported:

```text
avg_latency_seconds = arithmetic mean of per-example latency_seconds
```

What it does not do:

- It does not normalize by output length
- It does not normalize by token count
- It does not combine latency with quality metrics
- It does not summarize retry counts, even though `attempts` are stored per example

Interpretation:

Latency is an operational metric. It should be read as a tradeoff against quality, not as part of correctness.

## Aggregation behavior

Per-example scoring happens first. Aggregate reporting happens later.

Overall model summary:

- `exact_match` is the arithmetic mean of per-example exact-match scores
- `token_f1` is the arithmetic mean of per-example Token F1 scores
- `rouge_l` is the arithmetic mean of per-example ROUGE-L scores
- `avg_latency_seconds` is the arithmetic mean of per-example latency values

Task breakdown:

- Records are grouped by `task_name`
- Each task receives its own mean `exact_match`, `token_f1`, and `rouge_l`
- The summary output also includes `num_examples` per task

Important nuance:

The overall summary is averaged across examples, not macro-averaged equally across tasks. If some tasks contribute more examples, they carry more weight in the overall score.

Under the default configuration, tasks are sampled evenly enough that the weighting is close to balanced, but the code still aggregates at the example level.

## How to read the metrics together

The three quality metrics answer different questions:

- Exact Match: Did the model produce the exact normalized target?
- Token F1: Did the model include the right answer content?
- ROUGE-L: Did the model preserve answer content and ordering?

Common patterns:

- High Token F1, low Exact Match: content is mostly right, phrasing differs
- High Token F1, lower ROUGE-L: content overlaps, but the structure or order is weaker
- High Exact Match and high ROUGE-L: strong alignment with expected answer form
- High quality metrics with high latency: better quality at higher runtime cost
- Lower quality metrics with low latency: faster or cheaper tradeoff

## Limitations

The benchmark README calls out an important limitation: SuperNI is heterogeneous, so automatic scoring is necessarily approximate.

Practical consequences:

- Exact Match can under-credit valid paraphrases
- Token F1 can over-credit bag-of-words overlap
- ROUGE-L is still lexical and does not measure deep semantic equivalence
- None of the metrics directly evaluate factual grounding or reasoning faithfulness
- The benchmark does not compute confidence intervals or significance tests

These metrics are best used as a practical comparative signal across models, not as a complete measure of instruction-following quality.

## Source references

- `src/superni_benchmark/metrics.py`
- `src/superni_benchmark/runner.py`
- `tests/test_metrics.py`
- `README.md`
