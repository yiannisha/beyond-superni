from __future__ import annotations

import re
from collections import Counter


_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^0-9a-zA-Z\s]")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def exact_match_score(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(reference) for reference in references))


def token_f1_score(prediction: str, references: list[str]) -> float:
    prediction_tokens = normalize_text(prediction).split()
    if not prediction_tokens and not references:
        return 1.0

    best = 0.0
    for reference in references:
        reference_tokens = normalize_text(reference).split()
        if not prediction_tokens and not reference_tokens:
            return 1.0
        if not prediction_tokens or not reference_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def rouge_l_score(prediction: str, references: list[str]) -> float:
    prediction_tokens = normalize_text(prediction).split()
    if not prediction_tokens and not references:
        return 1.0

    best = 0.0
    for reference in references:
        reference_tokens = normalize_text(reference).split()
        if not prediction_tokens and not reference_tokens:
            return 1.0
        if not prediction_tokens or not reference_tokens:
            continue
        lcs = _longest_common_subsequence(prediction_tokens, reference_tokens)
        precision = lcs / len(prediction_tokens)
        recall = lcs / len(reference_tokens)
        if precision + recall == 0:
            continue
        f_measure = 2 * precision * recall / (precision + recall)
        best = max(best, f_measure)
    return best


def score_prediction(prediction: str, references: list[str]) -> dict[str, float]:
    return {
        "exact_match": exact_match_score(prediction, references),
        "token_f1": token_f1_score(prediction, references),
        "rouge_l": rouge_l_score(prediction, references),
    }


def _longest_common_subsequence(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]
