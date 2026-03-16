# Error Analysis: Common AI Response Failure Modes

This document catalogues recurring failure patterns observed in LLM-generated
responses across the five evaluation categories of this benchmark. Each section
includes concrete examples and an explanation of why the failure occurs.

---

## 1. Factual Mistakes

Factual errors occur when a model confidently states incorrect information,
often because it was undertrained on a topic, its training data contained
errors, or it confuses similar concepts.

### Example 1.1 — Wrong Historical Date

**Prompt:** Who wrote *Pride and Prejudice* and when was it published?

**Flawed Response:**
> *Pride and Prejudice* was written by Jane Austen and published in 1815.

**What went wrong:** The novel was published in **1813**, not 1815. The model
has plausible surface knowledge (correct author) but an off-by-two-years error
on the date — a typical factual slip caused by imprecise memorization.

---

### Example 1.2 — Confusing Cause and Effect for Seasons

**Prompt:** What causes the seasons on Earth?

**Flawed Response:**
> The seasons are caused by Earth moving closer to and farther from the Sun
> throughout the year.

**What went wrong:** This is a very common misconception. Earth's seasons are
caused by its **axial tilt** (~23.5°), not by orbital distance. In fact, Earth
is slightly *closer* to the Sun in January (Northern Hemisphere winter). The
model reproduced a popular misconception present in low-quality training text.

---

### Example 1.3 — Misattributing a Scientific Concept

**Prompt:** What is the Turing Test and who proposed it?

**Flawed Response:**
> The Turing Test was proposed by John McCarthy in 1956 as part of the
> Dartmouth AI conference.

**What went wrong:** The Turing Test was proposed by **Alan Turing** in his
1950 paper *Computing Machinery and Intelligence*. John McCarthy organized the
1956 Dartmouth conference where the term "artificial intelligence" was coined —
a related but distinct event. The model conflated two landmark moments in AI
history.

---

## 2. Incomplete Answers

Incomplete answers address part of a question but omit critical information,
leaving the user with a misleading or partial picture.

### Example 2.1 — Missing the Counterargument

**Prompt:** Is the greenhouse effect entirely harmful?

**Flawed Response:**
> Yes, the greenhouse effect is harmful because it traps heat and causes
> global warming.

**What went wrong:** The response fails to distinguish between the *natural*
greenhouse effect (essential for sustaining life on Earth — without it, average
temperatures would be −18 °C) and the *enhanced* greenhouse effect driven by
human emissions. A complete answer requires both perspectives.

---

### Example 2.2 — Code Solution Missing Error Handling

**Prompt:** Write a function that reads a JSON file and returns its contents.

**Flawed Response:**
```python
import json

def read_json(path):
    with open(path) as f:
        return json.load(f)
```

**What went wrong:** The function will crash with an unhandled `FileNotFoundError`
if the file does not exist, a `json.JSONDecodeError` for malformed JSON, and a
`PermissionError` if the file is not readable. A production-ready response must
include exception handling and communicate error conditions to the caller.

---

### Example 2.3 — Summary Omitting Key Tension

**Prompt:** Summarize the pros and cons of remote work.

**Flawed Response:**
> Remote work allows employees to work from home, save commute time, and
> have greater flexibility in their schedules.

**What went wrong:** The response only covers benefits. A balanced summary
must also address drawbacks: social isolation, collaboration challenges,
home office requirements, and the difficulty of separating work from personal
life. Answering only one side of a "pros and cons" request is a common
completeness failure.

---

## 3. Weak Reasoning

Weak reasoning occurs when a model arrives at a conclusion through flawed
logic, overgeneralisation, or by skipping essential inferential steps.

### Example 3.1 — Affirming the Consequent

**Prompt:** If it rains, the ground gets wet. The ground is wet. Did it
necessarily rain?

**Flawed Response:**
> Yes, it rained because the ground is wet.

**What went wrong:** This commits the logical fallacy of *affirming the
consequent*. The ground being wet is consistent with rain, but it does not
*prove* rain — a garden hose, sprinkler, or flooding could also explain it.
The model matched a surface-level pattern without applying logical reasoning.

---

### Example 3.2 — Correlation Confused with Causation

**Prompt:** A study found that cities with more ice cream sales have higher
crime rates. Does ice cream cause crime?

**Flawed Response:**
> This data suggests that ice cream consumption may lead to increased
> criminal behaviour, possibly due to sugar affecting mood.

**What went wrong:** The model invents a causal mechanism for what is
actually a *spurious correlation*. Both ice cream sales and crime rates
rise in summer (a confounding variable: hot weather). Concluding causation
from correlation without controlling for confounders is a fundamental
statistical reasoning error.

---

### Example 3.3 — Undistributed Middle

**Prompt:** Some politicians are dishonest. John is a politician. Is John
dishonest?

**Flawed Response:**
> Yes, John is likely dishonest because politicians are known to be dishonest.

**What went wrong:** The premise is that *some* (not all) politicians are
dishonest. The model overgeneralised "some" to "all" and applied a group
characteristic to an individual. This is the *undistributed middle*
fallacy — a very common LLM reasoning error on syllogism-style prompts.

---

## 4. Off-Topic Answers

Off-topic answers occur when a model provides a response that is technically
coherent but does not address the actual question asked.

### Example 4.1 — Answering a Related but Different Question

**Prompt:** What is the difference between a virus and a bacterium?

**Flawed Response:**
> Viruses and bacteria are both responsible for many diseases. Some common
> viral diseases include influenza and COVID-19, while bacterial diseases
> include strep throat and tuberculosis. Vaccines can prevent many of these
> illnesses.

**What went wrong:** The prompt asks about the *structural and biological
differences* between viruses and bacteria. The response instead lists
diseases caused by each — a tangential answer that ignores the core
distinction (living vs. non-living, replication mechanisms, treatment
approaches).

---

### Example 4.2 — Answering a Different Coding Task

**Prompt:** Write a Python function that checks if a number is prime.

**Flawed Response:**
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

**What went wrong:** The model returned a `factorial` function — a different
common mathematical function — instead of a primality check. This can happen
when the model confuses similar-sounding mathematical tasks or suffers from
catastrophic output forgetting mid-generation.

---

### Example 4.3 — Generic Advice Instead of Specific Analysis

**Prompt:** Identify the logical fallacy in: "We should ignore Dr. Smith's
climate research because she drives a gas-powered car."

**Flawed Response:**
> When evaluating scientific claims, it is important to consider the
> credibility of the source, the methodology used, and whether findings
> have been peer-reviewed. Personal behaviour may sometimes reflect on
> a researcher's objectivity.

**What went wrong:** The prompt asks for a *specific identification* of the
fallacy (ad hominem / tu quoque). Instead, the model gave generic advice
about evaluating scientific claims and even partially legitimised the
fallacious argument.

---

## 5. Vague Language

Vague responses use imprecise wording that sounds plausible but does not
commit to any concrete, verifiable claim.

### Example 5.1 — Non-Answer to a Specific Question

**Prompt:** How many bones are in the adult human body?

**Flawed Response:**
> The human body has many bones. The number can vary from person to person,
> but adults generally have quite a few bones, though babies have more because
> some bones fuse together as you grow.

**What went wrong:** The response correctly notes that bones fuse during
development but never states the actual number (206). Phrases like "many
bones," "quite a few," and "can vary" are evasive and provide no actionable
information when a specific, well-established answer exists.

---

### Example 5.2 — Uncommitted Technical Explanation

**Prompt:** What is the time complexity of binary search?

**Flawed Response:**
> Binary search is faster than linear search. It works by dividing the array
> and only looking at part of it, making it quite efficient for large datasets.

**What went wrong:** A correct answer requires stating *O(log n)* time
complexity and explaining *why* (the search space is halved with each
comparison). Saying "quite efficient" and "faster" without a mathematical
characterisation is vague and unhelpful to a developer who needs to reason
about performance.

---

### Example 5.3 — Machine Learning Explanation Without Substance

**Prompt:** Explain what overfitting is in machine learning.

**Flawed Response:**
> Overfitting is when a model doesn't perform well. It can happen when the
> model learns the wrong things and doesn't generalise. This is a common
> problem in machine learning and should be avoided.

**What went wrong:** Every sentence is either circular ("doesn't perform
well"), non-specific ("learns the wrong things"), or a platitude ("should
be avoided"). A useful response would explain that overfitting occurs when
a model memorises training data noise rather than learning underlying patterns,
leading to high training accuracy but poor test accuracy — and suggest
remedies like regularisation, dropout, or cross-validation.

---

*Last updated: March 2026 | LLM Eval Benchmark v1.0.0*
