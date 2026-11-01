"""Quantitative evaluation metrics for text generation in Question 4.

Implements standard generation quality, diversity, and repetition metrics:
1. Distinct-1 and Distinct-2: ratio of unique unigrams/bigrams to total generated n-grams.
2. Repetition Rate (Rep-4): intra-story 4-gram repetition rate, measuring phrase loops.
3. Token length statistics: average length, min, max, standard deviation.
4. Premature termination and non-termination rates.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def extract_ngrams(tokens: List[int], n: int) -> List[Tuple[int, ...]]:
    """Extract n-grams from a sequence of token IDs."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_sample_diversity(token_ids: List[int]) -> Dict[str, float]:
    """Compute Distinct-1, Distinct-2, and Rep-4 for a single generated continuation."""
    n_tokens = len(token_ids)
    if n_tokens == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "rep_4": 0.0, "length": 0}

    # Distinct-1
    u1 = extract_ngrams(token_ids, 1)
    dist1 = len(set(u1)) / max(1, len(u1))

    # Distinct-2
    u2 = extract_ngrams(token_ids, 2)
    dist2 = len(set(u2)) / max(1, len(u2)) if len(u2) > 0 else 0.0

    # Repetition-4 (fraction of duplicate 4-grams)
    u4 = extract_ngrams(token_ids, 4)
    if len(u4) > 0:
        rep4 = 1.0 - (len(set(u4)) / len(u4))
    else:
        rep4 = 0.0

    return {
        "distinct_1": float(dist1),
        "distinct_2": float(dist2),
        "rep_4": float(rep4),
        "length": n_tokens,
    }


def aggregate_generation_metrics(
    generations: List[Dict[str, Any]],
    min_len_threshold: int = 15,
    max_len_threshold: int = 100,
) -> Dict[str, float]:
    """Aggregate diversity, repetition, and termination statistics over a collection of generations."""
    if not generations:
        return {}

    dist1_scores: List[float] = []
    dist2_scores: List[float] = []
    rep4_scores: List[float] = []
    lengths: List[int] = []

    premature_count = 0
    unterminated_count = 0

    all_unigrams: List[Tuple[int, ...]] = []
    all_bigrams: List[Tuple[int, ...]] = []

    for gen in generations:
        token_ids = gen.get("generated_token_ids", [])
        eos_triggered = gen.get("eos_triggered", False)
        L = len(token_ids)
        lengths.append(L)

        # Premature EOS check
        if eos_triggered and L < min_len_threshold:
            premature_count += 1

        # Unterminated / hit maximum context
        if not eos_triggered and L >= max_len_threshold:
            unterminated_count += 1

        sample_metrics = compute_sample_diversity(token_ids)
        dist1_scores.append(sample_metrics["distinct_1"])
        dist2_scores.append(sample_metrics["distinct_2"])
        rep4_scores.append(sample_metrics["rep_4"])

        u1 = extract_ngrams(token_ids, 1)
        u2 = extract_ngrams(token_ids, 2)
        all_unigrams.extend(u1)
        all_bigrams.extend(u2)

    total_samples = len(generations)
    corpus_dist1 = len(set(all_unigrams)) / max(1, len(all_unigrams)) if all_unigrams else 0.0
    corpus_dist2 = len(set(all_bigrams)) / max(1, len(all_bigrams)) if all_bigrams else 0.0

    return {
        "sample_distinct_1_mean": float(np.mean(dist1_scores)),
        "sample_distinct_2_mean": float(np.mean(dist2_scores)),
        "corpus_distinct_1": float(corpus_dist1),
        "corpus_distinct_2": float(corpus_dist2),
        "rep_4_mean": float(np.mean(rep4_scores)),
        "avg_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "premature_eos_pct": float((premature_count / total_samples) * 100.0),
        "unterminated_pct": float((unterminated_count / total_samples) * 100.0),
        "num_samples": total_samples,
    }