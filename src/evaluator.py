"""
evaluator.py — Main evaluation engine for the LLM Eval Benchmark.

Loads evaluation tasks, scores AI responses against ideal outputs across
multiple quality dimensions, and generates structured evaluation reports.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

from scoring import (
    accuracy_score,
    clarity_score,
    composite_score,
    hallucination_risk_score,
    instruction_following_score,
    reasoning_quality_score,
)
from utils import format_report, load_evaluation_tasks, save_results


# ---------------------------------------------------------------------------
# Simulated AI responses (used when no real model is available)
# These represent a realistic mix of good, mediocre, and poor outputs.
# ---------------------------------------------------------------------------

SIMULATED_RESPONSES: dict[str, str] = {
    "reasoning_001": "Yes, whales are warm-blooded because they are mammals and all mammals are warm-blooded.",
    "reasoning_002": "Yes, it rained because the ground is wet.",  # Demo: weak reasoning (affirming the consequent)
    "reasoning_003": "The ball costs $0.10.",  # Demo: common incorrect answer (intuitive error)
    "reasoning_004": "Yes, Product X is effective since 80% of users reported improvement.",  # Demo: missing control group
    "reasoning_005": "This is an ad hominem fallacy — attacking the person rather than the argument.",
    "reasoning_006": "The probability is 5/8 times 4/7 which equals 5/14.",
    "reasoning_007": "They meet at approximately 11:36 AM.",
    "reasoning_008": "The argument is invalid because 'some politicians' does not imply all, so John might not be dishonest.",
    "coding_001": "import re\n\ndef is_palindrome(s):\n    cleaned = re.sub(r'[^a-z0-9]', '', s.lower())\n    return cleaned == cleaned[::-1]",
    "coding_002": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "coding_003": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    "coding_004": "The code has a SQL injection vulnerability. Fix it by using parameterized queries: db.query('SELECT * FROM users WHERE id = ?', (user_id,))",
    "coding_005": "import time\n\nclass Timer:\n    def __enter__(self):\n        self.start = time.perf_counter()\n        return self\n    def __exit__(self, *args):\n        print(f'Elapsed: {time.perf_counter() - self.start:.4f}s')",
    "coding_006": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)",
    "coding_007": "def fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b",
    "coding_008": "Time: O(n) for the loop, O(1) per set operation. Space: O(n) for the set. This is efficient.",
    "factuality_001": "The speed of light is approximately 299,792,458 meters per second.",
    "factuality_002": "Pride and Prejudice was written by Jane Austen and published in 1813.",
    "factuality_003": "Seasons are caused by Earth's axial tilt, not its distance from the Sun.",
    "factuality_004": "Bacteria are living single-celled organisms treatable with antibiotics. Viruses require host cells to replicate and are not treated with antibiotics.",
    "factuality_005": "The adult human body has 206 bones.",
    "factuality_006": "The Turing Test was proposed by Alan Turing in 1950 to assess a machine's ability to exhibit human-like intelligence.",
    "factuality_007": "The greenhouse effect traps solar heat in the atmosphere. It is naturally essential for life but human emissions are enhancing it harmfully.",
    "factuality_008": "DNA stands for Deoxyribonucleic Acid and stores genetic instructions for all living organisms.",
    "summarization_001": "The Industrial Revolution transformed economies through steam power and manufacturing, shifting populations to cities and raising living standards, though workers faced harsh early conditions.",
    "summarization_002": "Social media may increase loneliness by encouraging superficial interactions and social comparisons rather than meaningful connection.",
    "summarization_003": "Transformers use self-attention to process sequences, powering modern LLMs like GPT and BERT. They handle long-range dependencies better than previous architectures.",
    "summarization_004": "Key metrics: Revenue $4.2B (+12% YoY), Net Revenue Retention 128% (all-time high), Gross Margin 62.4% (+180 bps).",
    "summarization_005": "1984 follows Winston Smith, who rebels against the totalitarian Party and Big Brother, only to be captured and broken by the Thought Police through psychological torture.",
    "summarization_006": "Pros: no commute, flexibility, global talent pool. Cons: isolation, blurred work-life boundary, weaker collaboration.",
    "summarization_007": "Machine learning lets computers learn from data patterns instead of explicit rules, improving over time like a person gaining experience.",
    "summarization_008": "Adults sleeping 6 hours or less showed significantly worse memory, attention, and decision-making, suggesting sleep deprivation impairs cognitive function.",
    "instruction_following_001": "1. Germany (Berlin)\n2. France (Paris)\n3. Italy (Rome)\n4. Spain (Madrid)\n5. Poland (Warsaw)",
    "instruction_following_002": "Silicon neurons\nLearn the patterns of our world\nThinking without thought",
    "instruction_following_003": "Formal: The meeting has been cancelled.\nCasual: The meeting's off.\nHumorous: The meeting got ghosted.",
    "instruction_following_004": "Python attracts developers for several reasons. Its clean syntax lowers the learning curve. A vast library ecosystem makes it powerful for data science and AI.",
    "instruction_following_005": "Step 1: Crack eggs into a bowl.\nStep 2: Whisk with salt and pepper.\nStep 3: Heat a pan with butter.\nStep 4: Pour in eggs and stir gently.\nStep 5: Remove before fully set.",
    "instruction_following_006": "Meet the Apex Pro smartwatch — your ultimate health companion. Track heart rate, sleep, and workouts. Get calls and messages on a stunning AMOLED display. Seven-day battery life and aerospace-grade aluminum design make it perfect for every lifestyle.",
    "instruction_following_007": "Recursion is when a function calls itself. Like Russian dolls, each call opens a smaller version. Example: countdown(3) calls countdown(2) calls countdown(1). Warning: no base case causes infinite loops.",
    "instruction_following_008": "| Name | Role | Department |\n|------|------|------------|\n| Alice | Software Engineer | Engineering |\n| Bob | Product Manager | Product |\n| Carol | Data Scientist | Analytics |\n| Dave | UX Designer | Design |",
}


def evaluate_response(
    task: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    """
    Evaluate a single AI response against a task's ideal output.

    Args:
        task: A task dictionary with keys: task_id, category, input,
              ideal_output, difficulty.
        response: The AI-generated response string to evaluate.

    Returns:
        A dictionary containing per-dimension scores, a composite score,
        and metadata about the evaluation.
    """
    ideal = task["ideal_output"]
    category = task["category"]

    correctness = accuracy_score(response, ideal, category)
    reasoning = reasoning_quality_score(response, ideal)
    clarity = clarity_score(response)
    instruction = instruction_following_score(response, task["input"])
    hallucination = hallucination_risk_score(response, ideal)
    composite = composite_score(correctness, reasoning, clarity, instruction, hallucination)

    return {
        "task_id": task["task_id"],
        "category": category,
        "difficulty": task["difficulty"],
        "scores": {
            "correctness": correctness,
            "reasoning": reasoning,
            "clarity": clarity,
            "instruction_following": instruction,
            "hallucination_risk": hallucination,
            "composite": composite,
        },
        "response_length": len(response.split()),
        "ideal_length": len(ideal.split()),
    }


def run_benchmark(
    tasks_path: str,
    responses: dict[str, str] | None = None,
    model_name: str = "simulated-model-v1",
) -> dict[str, Any]:
    """
    Run the full evaluation benchmark over all tasks.

    Args:
        tasks_path: Path to the evaluation_tasks.json file.
        responses: Optional dict mapping task_id to AI response string.
                   If None, uses built-in simulated responses.
        model_name: Identifier for the model being evaluated.

    Returns:
        A structured report dict with per-task results, per-category
        summaries, and an overall benchmark score.
    """
    tasks = load_evaluation_tasks(tasks_path)
    if responses is None:
        responses = SIMULATED_RESPONSES

    task_results: list[dict[str, Any]] = []
    category_scores: dict[str, list[float]] = {}

    for task in tasks:
        task_id = task["task_id"]
        response = responses.get(task_id, "")
        if not response:
            print(f"  [WARNING] No response found for task '{task_id}', skipping.")
            continue

        result = evaluate_response(task, response)
        task_results.append(result)

        cat = result["category"]
        category_scores.setdefault(cat, []).append(result["scores"]["composite"])

    # Build per-category summary
    category_summary: dict[str, dict[str, Any]] = {}
    for cat, scores in category_scores.items():
        category_summary[cat] = {
            "task_count": len(scores),
            "average_composite": round(sum(scores) / len(scores), 2),
            "min_composite": round(min(scores), 2),
            "max_composite": round(max(scores), 2),
        }

    all_composite = [r["scores"]["composite"] for r in task_results]
    overall_score = round(sum(all_composite) / len(all_composite), 2) if all_composite else 0.0

    report = {
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tasks": len(task_results),
            "benchmark_version": "1.0.0",
        },
        "overall_score": overall_score,
        "category_summary": category_summary,
        "task_results": task_results,
    }

    return report


def main() -> None:
    """Entry point: run the benchmark and save + print the report."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks_path = os.path.join(base_dir, "data", "evaluation_tasks.json")
    output_path = os.path.join(base_dir, "outputs", "sample_results.json")

    print("=" * 60)
    print("  LLM Eval Benchmark — Running Evaluation")
    print("=" * 60)

    report = run_benchmark(tasks_path)

    save_results(report, output_path)
    print(f"\nResults saved to: {output_path}\n")

    readable = format_report(report)
    print(readable)


if __name__ == "__main__":
    main()
