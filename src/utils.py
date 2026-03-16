"""
utils.py — Helper utilities for the LLM Eval Benchmark.

Provides I/O operations, text similarity, report formatting, and
metric extraction helpers used across the evaluation pipeline.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any


def load_evaluation_tasks(path: str) -> list[dict[str, Any]]:
    """
    Load and validate evaluation tasks from a JSON file.

    The JSON file must contain a top-level "evaluation_tasks" array.
    Each task must have: task_id, category, input, ideal_output, difficulty.

    Args:
        path: Absolute or relative path to the evaluation_tasks.json file.

    Returns:
        A list of validated task dictionaries.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON is malformed or missing required fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation tasks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_key = "evaluation_tasks"
    if required_key not in data:
        raise ValueError(
            f"JSON must contain a top-level '{required_key}' array. "
            f"Found keys: {list(data.keys())}"
        )

    tasks = data[required_key]
    required_fields = {"task_id", "category", "input", "ideal_output", "difficulty"}
    valid_categories = {"reasoning", "coding", "factuality", "summarization", "instruction_following"}
    valid_difficulties = {"easy", "medium", "hard"}

    validated: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        missing = required_fields - set(task.keys())
        if missing:
            print(f"  [WARNING] Task at index {i} is missing fields {missing}, skipping.")
            continue
        if task["category"] not in valid_categories:
            print(f"  [WARNING] Task '{task['task_id']}' has unknown category "
                  f"'{task['category']}', skipping.")
            continue
        if task["difficulty"] not in valid_difficulties:
            print(f"  [WARNING] Task '{task['task_id']}' has unknown difficulty "
                  f"'{task['difficulty']}', skipping.")
            continue
        validated.append(task)

    print(f"  Loaded {len(validated)} valid tasks from {path}")
    return validated


def save_results(results: dict[str, Any], output_path: str) -> None:
    """
    Write evaluation results to a JSON file, creating parent directories
    if they do not exist.

    Args:
        results: The structured results dictionary to serialize.
        output_path: The file path where results will be written.
    """
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results written to {output_path}")


def format_report(report: dict[str, Any]) -> str:
    """
    Convert a structured results dict into a human-readable text report.

    Args:
        report: A benchmark report dict as returned by run_benchmark().

    Returns:
        A formatted multi-line string suitable for console output.
    """
    meta = report.get("metadata", {})
    lines = [
        "=" * 60,
        "  LLM EVAL BENCHMARK — RESULTS REPORT",
        "=" * 60,
        f"  Model       : {meta.get('model', 'unknown')}",
        f"  Timestamp   : {meta.get('timestamp', 'N/A')}",
        f"  Total Tasks : {meta.get('total_tasks', 0)}",
        f"  Version     : {meta.get('benchmark_version', 'N/A')}",
        "",
        f"  OVERALL SCORE: {report.get('overall_score', 0):.1f} / 100",
        "",
        "  CATEGORY SCORES",
        "  " + "-" * 40,
    ]

    category_summary = report.get("category_summary", {})
    for cat, stats in sorted(category_summary.items()):
        lines.append(
            f"  {cat:<25}  avg={stats['average_composite']:5.1f}  "
            f"min={stats['min_composite']:5.1f}  max={stats['max_composite']:5.1f}  "
            f"n={stats['task_count']}"
        )

    lines += ["", "  PER-TASK SCORES (composite)", "  " + "-" * 40]

    task_results = report.get("task_results", [])
    for task in sorted(task_results, key=lambda x: x["task_id"]):
        scores = task.get("scores", {})
        lines.append(
            f"  {task['task_id']:<35}  "
            f"composite={scores.get('composite', 0):5.1f}  "
            f"correctness={scores.get('correctness', 0):5.1f}  "
            f"halluc_risk={scores.get('hallucination_risk', 0):5.1f}"
        )

    lines += ["", "=" * 60]
    return "\n".join(lines)


def text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute a normalised similarity score between two text strings.

    Uses a combination of Jaccard token overlap and length ratio to
    produce a score in [0.0, 1.0]. Useful for quick fuzzy comparison.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Similarity score in [0.0, 1.0] (1.0 = identical).
    """
    def tokenize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return set(text.split())

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union

    # Penalize large length disparities
    len_a, len_b = len(text_a.split()), len(text_b.split())
    length_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 1.0

    return round((jaccard * 0.7 + length_ratio * 0.3), 4)


def extract_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-task results into summary statistics for each
    scoring dimension.

    Args:
        results: List of task result dicts (from evaluate_response()).

    Returns:
        Dict mapping each dimension name to {"mean", "min", "max"}.
    """
    if not results:
        return {}

    dimensions = [
        "correctness", "reasoning", "clarity",
        "instruction_following", "hallucination_risk", "composite"
    ]
    metrics: dict[str, Any] = {}

    for dim in dimensions:
        values = [
            r["scores"][dim]
            for r in results
            if "scores" in r and dim in r["scores"]
        ]
        if values:
            metrics[dim] = {
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "count": len(values),
            }

    return metrics


def get_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()

