# Prompt Engineering: Weak vs. Improved Prompts

Well-crafted prompts dramatically improve the quality, accuracy, and
usefulness of LLM responses. This document shows side-by-side comparisons
of weak prompts and their improved counterparts, with explanations of why
each improvement works.

---

## Improvement 1: Adding Specificity and Context

### ❌ Weak Prompt
> Tell me about machine learning.

**Problem:** The question is too open-ended. A model has no guidance on
scope, depth, audience, or format. It will likely produce a generic,
surface-level overview that may not be relevant to what the user actually
needs.

**Typical weak response:** A 500-word essay covering everything from
neural networks to regression, useful to no one in particular.

---

### ✅ Improved Prompt
> Explain the concept of supervised learning in machine learning to a
> software engineer who is new to AI. Use one concrete real-world example
> and keep the explanation under 150 words.

**Why it works:**
- **Audience specification** ("software engineer new to AI") calibrates
  vocabulary and assumed knowledge.
- **Scope constraint** ("supervised learning") focuses the answer.
- **Format requirement** ("concrete real-world example") ensures
  actionable content.
- **Length limit** ("under 150 words") forces conciseness and prevents
  padding.

---

## Improvement 2: Role-Playing / Persona Assignment

### ❌ Weak Prompt
> Review my code.

**Problem:** Without context, the model doesn't know what language, what
kind of review (security, style, performance, correctness), or what
audience the feedback is for. The review will be generic and miss
domain-specific concerns.

---

### ✅ Improved Prompt
> Act as a senior Python engineer conducting a code review. Review the
> following function for: (1) correctness, (2) security vulnerabilities,
> (3) PEP 8 style compliance, and (4) edge cases. For each issue, provide
> a specific line reference and a suggested fix.
>
> ```python
> def get_user(user_id):
>     db = connect_to_database()
>     result = db.query(f'SELECT * FROM users WHERE id = {user_id}')
>     return result
> ```

**Why it works:**
- **Role assignment** ("senior Python engineer") primes domain expertise
  and tone.
- **Numbered checklist** structures the output into predictable, parseable
  sections.
- **Specific criteria** (correctness, security, style, edge cases) prevents
  the model from reviewing only the easy surface-level issues.
- **Code included in prompt** eliminates ambiguity about what to review.

---

## Improvement 3: Chain-of-Thought Instruction

### ❌ Weak Prompt
> Is this argument valid? "All birds can fly. Penguins are birds.
> Therefore, penguins can fly."

**Problem:** Without instructing the model to reason step by step, it may
jump to a surface-level answer ("No") without explaining *which* premise
is false, *why* the argument structure fails, or *what* kind of error
it represents.

---

### ✅ Improved Prompt
> Evaluate the following argument for logical validity. Think step by step:
> (1) Identify the logical form (premises and conclusion). (2) Check
> whether the conclusion follows from the premises. (3) Identify any
> false or overstated premises. (4) State whether the argument is valid
> and/or sound.
>
> Argument: "All birds can fly. Penguins are birds. Therefore, penguins
> can fly."

**Why it works:**
- **"Think step by step"** triggers chain-of-thought reasoning, which
  consistently improves accuracy on logical and mathematical tasks.
- **Numbered analysis framework** forces the model to address each logical
  component: form, validity, and soundness.
- **Distinction between valid and sound** elevates the analysis beyond
  a simple yes/no answer.

---

## Improvement 4: Output Format Specification

### ❌ Weak Prompt
> What are the pros and cons of using microservices?

**Problem:** Without a format requirement, the model may produce a wall
of prose, an unbalanced list, or a response that mixes pros and cons
in an unstructured way — hard to scan and compare.

---

### ✅ Improved Prompt
> Compare the pros and cons of microservices architecture for a mid-size
> engineering team (20–50 engineers). Format your response as a Markdown
> table with three columns: **Dimension**, **Pros**, **Cons**. Cover at
> least 5 dimensions (e.g., scalability, deployment, team autonomy,
> complexity, cost). After the table, add a one-paragraph recommendation
> for when to adopt microservices.

**Why it works:**
- **Target audience** ("20–50 engineers") makes the response practical
  rather than theoretical.
- **Explicit Markdown table format** produces output that can be directly
  pasted into documentation or a presentation.
- **Minimum coverage requirement** ("at least 5 dimensions") prevents a
  superficial two-point answer.
- **Appended recommendation** forces a synthesized conclusion rather than
  just listing facts.

---

## Improvement 5: Few-Shot Examples

### ❌ Weak Prompt
> Classify the sentiment of the following customer reviews.
>
> Reviews:
> - "The product arrived broken."
> - "Absolutely love it, best purchase this year!"
> - "It's okay, nothing special."

**Problem:** Without examples of the expected output format, the model
may classify as "positive/negative/neutral," "1-5 stars," "happy/sad/
meh," or use different inconsistent labels. For automated pipelines,
inconsistent output format is a critical failure.

---

### ✅ Improved Prompt
> Classify the sentiment of each customer review as exactly one of:
> **Positive**, **Negative**, or **Neutral**. Return the results as a
> JSON array where each item has "review" and "sentiment" keys.
>
> Here are two examples:
>
> Input: "Shipping was fast and the item is perfect."
> Output: {"review": "Shipping was fast and the item is perfect.", "sentiment": "Positive"}
>
> Input: "Took three weeks to arrive and the box was damaged."
> Output: {"review": "Took three weeks to arrive and the box was damaged.", "sentiment": "Negative"}
>
> Now classify:
> - "The product arrived broken."
> - "Absolutely love it, best purchase this year!"
> - "It's okay, nothing special."

**Why it works:**
- **Constrained label set** ("exactly one of: Positive, Negative, Neutral")
  eliminates label inconsistency.
- **JSON output format** makes the response directly parseable in code.
- **Two-shot examples** (few-shot prompting) demonstrate the exact expected
  format, reducing formatting errors by over 40% on structured output tasks.
- **"Now classify"** cleanly separates examples from the actual task.

---

## Summary of Key Prompt Engineering Principles

| Principle | Benefit |
|-----------|---------|
| Specify the audience | Calibrates vocabulary and depth |
| Assign a role/persona | Activates domain expertise |
| Use chain-of-thought ("step by step") | Improves reasoning accuracy |
| Define output format explicitly | Produces consistent, parseable results |
| Provide few-shot examples | Reduces formatting errors |
| Set scope and length constraints | Prevents vague, padded responses |
| Ask for specific criteria | Ensures all relevant dimensions are covered |

---

*Last updated: March 2026 | LLM Eval Benchmark v1.0.0*
