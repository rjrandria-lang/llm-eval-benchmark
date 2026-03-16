# LLM Eval Benchmark

A production-ready Python framework for evaluating AI-generated responses
across five core quality dimensions. Designed to surface failure modes,
measure response quality, and benchmark language model performance without
requiring external APIs or paid services.

---

## Project Overview

Large language models are increasingly used in high-stakes applications —
from customer support to code generation to medical information retrieval.
Yet evaluating the *quality* of their outputs remains challenging: existing
metrics like BLEU and ROUGE measure surface-level text overlap, not whether
a response is logically correct, safe, or useful.

This benchmark addresses that gap with a rule-based evaluation engine that
scores AI responses across **five quality dimensions** and **five task
categories**, producing structured, interpretable results that help
identify where models excel and where they fail.

---

## Key Features

### Evaluation Dimensions
| Dimension | Description | Range |
|-----------|-------------|-------|
| **Correctness** | Factual/semantic accuracy vs. ideal output | 0–100 |
| **Reasoning** | Quality of logical explanation and inference | 0–100 |
| **Clarity** | Communication quality and structure | 0–100 |
| **Instruction Following** | Adherence to format and task requirements | 0–100 |
| **Hallucination Risk** | Likelihood of false/fabricated content | 0–100 *(lower is better)* |

### Task Categories
- **Reasoning** — Logical deduction, fallacy detection, probability, syllogisms
- **Coding** — Function implementation, code review, complexity analysis
- **Factuality** — Knowledge retrieval, scientific accuracy, historical facts
- **Summarization** — Concise abstraction, key point extraction, plain-language translation
- **Instruction Following** — Format compliance, constraint adherence, multi-step tasks

### Dataset
- **40 evaluation tasks** across all five categories (8 per category)
- Three difficulty levels: easy / medium / hard
- Each task has a curated `ideal_output` serving as the gold standard

---

## Project Structure

```
llm-eval-benchmark/
├── data/
│   └── evaluation_tasks.json    # 40 evaluation tasks with ideal outputs
├── src/
│   ├── evaluator.py             # Main evaluation engine and CLI entry point
│   ├── scoring.py               # Scoring algorithms (all five dimensions)
│   └── utils.py                 # I/O helpers, text similarity, report formatting
├── outputs/
│   └── sample_results.json      # Pre-generated evaluation results
├── docs/
│   ├── error_analysis.md        # Taxonomy of common AI failure modes
│   └── prompt_improvements.md   # Weak vs. improved prompt examples
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10 or later
- No external API keys required

### Installation

```bash
git clone https://github.com/rjrandria-lang/llm-eval-benchmark.git
cd llm-eval-benchmark
pip install -r requirements.txt
```

---

## Usage Guide

### Run the full benchmark (simulated responses)

```bash
cd src
python evaluator.py
```

This will:
1. Load all 40 tasks from `data/evaluation_tasks.json`
2. Score built-in simulated responses across all five dimensions
3. Print a formatted report to the console
4. Save full results to `outputs/sample_results.json`

### Evaluate your own model's responses

```python
from src.evaluator import run_benchmark

# Map each task_id to your model's response string
my_responses = {
    "reasoning_001": "Yes, whales are warm-blooded because they are mammals.",
    "coding_001": "import re\ndef is_palindrome(s): ...",
    # ... one entry per task_id
}

report = run_benchmark(
    tasks_path="data/evaluation_tasks.json",
    responses=my_responses,
    model_name="my-llm-v1",
)

print(f"Overall score: {report['overall_score']}")
print(f"Category breakdown: {report['category_summary']}")
```

### Use individual scoring functions

```python
from src.scoring import accuracy_score, hallucination_risk_score, composite_score

response = "Paris is the capital of France and has a population of about 2 million."
ideal    = "Paris is the capital of France."

correctness = accuracy_score(response, ideal, category="factuality")
hallucination = hallucination_risk_score(response, ideal)
overall = composite_score(correctness, 75, 80, 90, hallucination)

print(f"Correctness: {correctness}")
print(f"Hallucination risk: {hallucination}")
print(f"Composite: {overall}")
```

---

## Sample Output

```
============================================================
  LLM EVAL BENCHMARK — RESULTS REPORT
============================================================
  Model       : simulated-model-v1
  Timestamp   : 2026-03-16T04:55:57+00:00
  Total Tasks : 40
  Version     : 1.0.0

  OVERALL SCORE: 43.2 / 100

  CATEGORY SCORES
  ----------------------------------------
  coding                     avg= 47.5  min= 30.1  max= 63.7  n=8
  factuality                 avg= 39.2  min= 33.8  max= 44.2  n=8
  instruction_following      avg= 57.0  min= 34.3  max= 81.0  n=8
  reasoning                  avg= 35.1  min= 29.9  max= 47.3  n=8
  summarization              avg= 36.9  min= 30.7  max= 39.7  n=8
```

### Interpreting Results

| Score Range | Interpretation |
|-------------|----------------|
| 80–100 | Excellent — responses closely match ideal outputs |
| 60–79 | Good — mostly correct with minor gaps |
| 40–59 | Moderate — partial correctness, some errors |
| 20–39 | Weak — significant factual or reasoning failures |
| 0–19 | Poor — largely incorrect or off-topic |

**Note:** The simulated responses intentionally include both strong and
weak examples to demonstrate the full scoring range. Real-world LLMs
typically score 55–80 on this benchmark depending on model size.

---

## Architecture and Implementation

### Scoring Approach

All scoring is deterministic and rule-based — no external LLM judge is
required. This makes results reproducible and interpretable:

- **`accuracy_score`**: Uses token-level Jaccard overlap and bigram overlap
  (BLEU-2 inspired), weighted by task category (coding tasks weight
  structural bigram overlap more; factuality tasks weight keyword presence).

- **`reasoning_quality_score`**: Detects 18 linguistic reasoning signals
  (logical connectives, step markers, hedging language) and combines them
  with ideal-output overlap.

- **`clarity_score`**: Analyses sentence length distribution, markdown
  formatting signals, and vague filler language patterns.

- **`instruction_following_score`**: Matches 12 explicit instruction patterns
  (numbered lists, tables, haiku structure, tone requirements, exact counts).

- **`hallucination_risk_score`**: Penalises low overlap with the ideal,
  confident absolute language, and suspicious novel numeric claims.

- **`composite_score`**: Weighted combination — Correctness 35%, Instruction
  Following 20%, Reasoning 20%, Clarity 15%, Hallucination (inverted) 10%.

### Design Decisions

- **Zero external dependencies for core scoring** — no API calls, no
  sentence transformers, no GPU required. The benchmark runs in under 1
  second on any modern laptop.
- **Modular architecture** — `scoring.py`, `evaluator.py`, and `utils.py`
  are fully decoupled. Drop in a different scoring backend without touching
  the evaluation pipeline.
- **Type-annotated** — all public functions carry Python type hints for
  IDE support and runtime validation.

---

## Documentation

- [`docs/error_analysis.md`](docs/error_analysis.md) — Taxonomy of five
  common AI failure modes with 15 concrete examples across categories.
- [`docs/prompt_improvements.md`](docs/prompt_improvements.md) — Five
  weak-to-improved prompt transformations demonstrating key prompt
  engineering principles.

---

## Relevance for AI/ML Roles

This project demonstrates practical skills relevant to **AI evaluation,
prompt engineering, and ML engineering** roles:

- **Evaluation design** — building ground-truth datasets and scoring rubrics
- **Prompt engineering** — systematic analysis of what makes prompts effective
- **Python engineering** — modular, type-annotated, production-quality code
- **AI safety awareness** — hallucination detection and failure mode analysis
- **Data pipeline skills** — JSON schema design, validation, and result serialisation

---

## License

MIT — see [LICENSE](LICENSE) for details.

## Contact

Rojoni Randria — rojoni.randria@gmail.com

