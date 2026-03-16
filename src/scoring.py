"""
scoring.py — Scoring algorithms for LLM response evaluation.

Each function returns a score in the range [0, 100] where higher is better,
except hallucination_risk_score where lower is better (0 = no risk).
All functions are deterministic and rule-based, requiring no external APIs.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Compute the Jaccard token overlap between two strings.

    Returns a float in [0.0, 1.0].
    """
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _bigram_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Compute bigram overlap (BLEU-2 inspired) between two strings.

    Returns a float in [0.0, 1.0].
    """
    def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return _token_overlap_ratio(text_a, text_b)

    bg_a = bigrams(tokens_a)
    bg_b = bigrams(tokens_b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def accuracy_score(response: str, ideal: str, category: str = "") -> float:
    """
    Measure the factual/semantic correctness of a response vs. the ideal output.

    Uses a combination of unigram and bigram overlap, weighted by category:
    - coding tasks weight bigram overlap more (structure matters)
    - factuality tasks weight unigram overlap more (key terms matter)
    - other categories use a balanced blend

    Args:
        response: The AI-generated response.
        ideal: The ideal/expected output.
        category: Optional task category to adjust weighting.

    Returns:
        A correctness score in [0, 100].
    """
    unigram = _token_overlap_ratio(response, ideal)
    bigram = _bigram_overlap_ratio(response, ideal)

    if category == "coding":
        score = 0.35 * unigram + 0.65 * bigram
    elif category == "factuality":
        score = 0.70 * unigram + 0.30 * bigram
    else:
        score = 0.55 * unigram + 0.45 * bigram

    return round(score * 100, 2)


def reasoning_quality_score(response: str, ideal: str) -> float:
    """
    Evaluate the quality of logical explanation and reasoning in a response.

    Signals of good reasoning include: use of logical connectives, explanation
    of steps, hedging where appropriate, and addressing the 'why/how.'

    Args:
        response: The AI-generated response.
        ideal: The ideal output (used for overlap baseline).

    Returns:
        A reasoning quality score in [0, 100].
    """
    reasoning_signals = [
        r"\bbecause\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
        r"\bsince\b", r"\bconsequently\b", r"\bit follows\b",
        r"\bthis means\b", r"\bwhich means\b", r"\bwe can conclude\b",
        r"\bfor example\b", r"\bfor instance\b", r"\bin other words\b",
        r"\bhowever\b", r"\balthough\b", r"\bnevertheless\b",
        r"\bstep \d", r"\bfirst[,\s]", r"\bsecond[,\s]", r"\bfinally\b",
    ]

    resp_lower = response.lower()
    signal_count = sum(
        1 for pattern in reasoning_signals if re.search(pattern, resp_lower)
    )

    # Overlap with ideal as a base (good reasoning tends to cover same concepts)
    base_overlap = _token_overlap_ratio(response, ideal)
    base_score = base_overlap * 60  # up to 60 points from overlap

    # Bonus for each reasoning signal found, capped at 40 points
    signal_score = min(signal_count * 6, 40)

    total = base_score + signal_score
    return round(min(total, 100), 2)


def clarity_score(response: str) -> float:
    """
    Assess the clarity and communication quality of a response.

    Considers: sentence structure, length appropriateness, use of formatting,
    and avoidance of excessive hedging or filler language.

    Args:
        response: The AI-generated response.

    Returns:
        A clarity score in [0, 100].
    """
    if not response.strip():
        return 0.0

    words = response.split()
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Length score: ideal response is 20-300 words
    word_count = len(words)
    if 20 <= word_count <= 300:
        length_score = 30
    elif word_count < 5:
        length_score = 5
    elif word_count < 20:
        length_score = 15
    else:
        # Penalize overly long responses gradually
        length_score = max(10, 30 - (word_count - 300) // 20)

    # Sentence diversity score
    if sentences:
        avg_sentence_len = word_count / len(sentences)
        # Ideal: 10-25 words per sentence
        if 10 <= avg_sentence_len <= 25:
            sentence_score = 25
        else:
            sentence_score = max(5, 25 - abs(avg_sentence_len - 17) * 1.5)
    else:
        sentence_score = 5

    # Structure bonus: markdown formatting, numbering, or clear segmentation
    structure_bonus = 0
    if re.search(r"^\s*[-*]\s", response, re.MULTILINE):  # bullet points
        structure_bonus += 10
    if re.search(r"^\s*\d+\.", response, re.MULTILINE):   # numbered list
        structure_bonus += 10
    if re.search(r"\*\*[^*]+\*\*", response):              # bold text
        structure_bonus += 5
    structure_bonus = min(structure_bonus, 20)

    # Filler / vague language penalty (broad heuristics — intentionally conservative)
    vague_patterns = [r"\bvery\b", r"\bbasically\b", r"\bsort of\b", r"\bkind of\b",
                      r"\bstuff\b", r"\bthings\b", r"\betc\b", r"\bI don'?t know\b"]
    vague_count = sum(1 for p in vague_patterns if re.search(p, response, re.IGNORECASE))
    vague_penalty = min(vague_count * 3, 15)

    base = 25  # base clarity score
    total = base + length_score + sentence_score + structure_bonus - vague_penalty
    return round(min(max(total, 0), 100), 2)


def instruction_following_score(response: str, instruction: str) -> float:
    """
    Assess how well the response adheres to explicit instructions in the prompt.

    Detects compliance with common instruction patterns:
    - Numbered lists / bullet points when requested
    - Word/sentence count constraints
    - Format requirements (table, haiku, step-by-step, etc.)
    - Tone requirements

    Args:
        response: The AI-generated response.
        instruction: The original task prompt.

    Returns:
        An instruction-following score in [0, 100].
    """
    instr_lower = instruction.lower()
    resp_lower = response.lower()
    score = 60  # base score for a non-empty, on-topic response

    # Check for numbered list compliance
    if re.search(r"\d+\s*\.\s*(list|steps?|items?|points?|examples?)", instr_lower):
        if re.search(r"^\s*\d+\.", response, re.MULTILINE):
            score += 15
        else:
            score -= 15

    # Check for bullet list compliance
    if re.search(r"(bullet|list|points?)", instr_lower):
        if re.search(r"^\s*[-*•]", response, re.MULTILINE):
            score += 10

    # Check for step-by-step requirement
    if re.search(r"step.by.step|step \d", instr_lower):
        if re.search(r"step \d", resp_lower):
            score += 15
        else:
            score -= 10

    # Check for table format
    if "table" in instr_lower or "markdown table" in instr_lower:
        if "|" in response:
            score += 15
        else:
            score -= 20

    # Check for haiku structure (5-7-5 syllable approximation via line count)
    if "haiku" in instr_lower:
        lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
        if len(lines) == 3:
            score += 20
        else:
            score -= 20

    # Check for exact count requirements (e.g., "exactly 5", "3 examples")
    exact_match = re.search(r"exactly (\d+)", instr_lower)
    if exact_match:
        required = int(exact_match.group(1))
        found = len(re.findall(r"^\s*\d+\.", response, re.MULTILINE))
        if found == required:
            score += 15
        elif found > 0:
            score += 5
        else:
            score -= 10

    # Check for tone compliance
    if "formal" in instr_lower and ("formal" in resp_lower or "we regret" in resp_lower):
        score += 5
    if "casual" in instr_lower and ("hey" in resp_lower or "it's" in resp_lower):
        score += 5
    if "humorous" in instr_lower and ("!" in response or "😄" in response):
        score += 5

    # Penalize very short responses
    if len(response.split()) < 5:
        score -= 20

    return round(min(max(score, 0), 100), 2)


def hallucination_risk_score(response: str, ideal: str) -> float:
    """
    Estimate the likelihood that the response contains false or fabricated information.

    A lower score is better (0 = very low risk, 100 = high risk).
    Detection heuristics:
    - Low overlap with ideal output → higher risk
    - Confident absolute claims without evidence
    - Fabricated specifics (invented statistics, fake citations)

    Args:
        response: The AI-generated response.
        ideal: The ideal/reference output.

    Returns:
        A hallucination risk score in [0, 100] (lower is better).
    """
    # Base risk inversely proportional to token overlap
    overlap = _token_overlap_ratio(response, ideal)
    base_risk = (1.0 - overlap) * 60  # max 60 from overlap gap

    # Confident absolute language without hedging increases risk
    high_confidence_patterns = [
        r"\balways\b", r"\bnever\b", r"\bproven\b", r"\bscientifically proven\b",
        r"\bfact\b", r"\bguaranteed\b", r"\bwithout question\b", r"\bwithout a doubt\b",
    ]
    resp_lower = response.lower()
    confidence_hits = sum(1 for p in high_confidence_patterns if re.search(p, resp_lower))
    confidence_risk = min(confidence_hits * 5, 20)

    # Fabricated statistics: suspicious numeric patterns like percentages/citations
    # that don't appear in the ideal output
    numbers_in_response = set(re.findall(r"\b\d+\.?\d*\b", response))
    numbers_in_ideal = set(re.findall(r"\b\d+\.?\d*\b", ideal))
    novel_numbers = numbers_in_response - numbers_in_ideal
    # A few novel numbers are fine; many suggest fabrication
    number_risk = min(max(len(novel_numbers) - 3, 0) * 3, 20)

    total = base_risk + confidence_risk + number_risk
    return round(min(total, 100), 2)


def composite_score(
    correctness: float,
    reasoning: float,
    clarity: float,
    instruction_following: float,
    hallucination_risk: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a weighted composite score across all evaluation dimensions.

    The hallucination_risk is inverted (100 - score) before weighting so that
    a lower risk produces a higher composite score.

    Default weights:
        correctness: 35%
        reasoning: 20%
        clarity: 15%
        instruction_following: 20%
        hallucination_risk (inverted): 10%

    Args:
        correctness: Accuracy score [0, 100].
        reasoning: Reasoning quality score [0, 100].
        clarity: Clarity score [0, 100].
        instruction_following: Instruction-following score [0, 100].
        hallucination_risk: Hallucination risk score [0, 100] (lower is better).
        weights: Optional custom weight dict. Keys must match dimension names.

    Returns:
        A composite score in [0, 100].
    """
    if weights is None:
        weights = {
            "correctness": 0.35,
            "reasoning": 0.20,
            "clarity": 0.15,
            "instruction_following": 0.20,
            "hallucination_inverted": 0.10,
        }

    hallucination_inverted = 100 - hallucination_risk

    composite = (
        correctness * weights.get("correctness", 0.35)
        + reasoning * weights.get("reasoning", 0.20)
        + clarity * weights.get("clarity", 0.15)
        + instruction_following * weights.get("instruction_following", 0.20)
        + hallucination_inverted * weights.get("hallucination_inverted", 0.10)
    )
    return round(min(max(composite, 0), 100), 2)


def extract_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate raw task results into high-level metric summaries.

    Args:
        results: List of task result dicts as returned by evaluate_response().

    Returns:
        A dict with mean, min, and max for each scoring dimension.
    """
    if not results:
        return {}

    dimensions = ["correctness", "reasoning", "clarity", "instruction_following",
                  "hallucination_risk", "composite"]
    metrics: dict[str, Any] = {}

    for dim in dimensions:
        values = [r["scores"][dim] for r in results if dim in r.get("scores", {})]
        if values:
            metrics[dim] = {
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }

    return metrics
