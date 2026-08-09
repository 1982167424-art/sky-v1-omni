"""sky_v1.eval.mmlu: MMLU-style 5-shot multiple-choice evaluation.

Implements spec §10.1 Benchmark layer. Uses a hard-coded mini-MMLU of 20
questions spanning 4 subjects (STEM / Humanities / Social Sciences / Other,
5 questions each). No real dataset download is performed.
"""
from __future__ import annotations

import re
import time
from typing import Any

from .benchmark import BenchmarkResult


MINI_MMLU_QUESTIONS: list[dict[str, Any]] = [
    # --- STEM (5) ---
    {
        "subject": "STEM",
        "question": "What is the value of 7 x 8?",
        "choices": ["54", "56", "64", "58"],
        "answer": "B",
    },
    {
        "subject": "STEM",
        "question": "What is the SI unit of force?",
        "choices": ["Joule", "Watt", "Newton", "Pascal"],
        "answer": "C",
    },
    {
        "subject": "STEM",
        "question": "What is the chemical symbol for gold?",
        "choices": ["Go", "Gd", "Au", "Ag"],
        "answer": "C",
    },
    {
        "subject": "STEM",
        "question": "Which organ is responsible for pumping blood through the human body?",
        "choices": ["Liver", "Heart", "Lung", "Kidney"],
        "answer": "B",
    },
    {
        "subject": "STEM",
        "question": "Which data structure follows the First-In-First-Out (FIFO) ordering?",
        "choices": ["Stack", "Queue", "Tree", "Graph"],
        "answer": "B",
    },
    # --- Humanities (5) ---
    {
        "subject": "Humanities",
        "question": "In what year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": "C",
    },
    {
        "subject": "Humanities",
        "question": "Who wrote the philosophical work 'The Republic'?",
        "choices": ["Aristotle", "Plato", "Socrates", "Kant"],
        "answer": "B",
    },
    {
        "subject": "Humanities",
        "question": "Who is the author of the play 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "answer": "B",
    },
    {
        "subject": "Humanities",
        "question": "Which book is the central religious text of Islam?",
        "choices": ["Bible", "Torah", "Quran", "Vedas"],
        "answer": "C",
    },
    {
        "subject": "Humanities",
        "question": "Who painted the Mona Lisa?",
        "choices": ["Vincent van Gogh", "Pablo Picasso", "Leonardo da Vinci", "Claude Monet"],
        "answer": "C",
    },
    # --- Social Sciences (5) ---
    {
        "subject": "Social Sciences",
        "question": "What does the acronym GDP stand for in economics?",
        "choices": [
            "Gross Domestic Product",
            "General Domestic Product",
            "Gross Demand Product",
            "General Demand Product",
        ],
        "answer": "A",
    },
    {
        "subject": "Social Sciences",
        "question": "Who is regarded as the founder of psychoanalysis?",
        "choices": ["Carl Jung", "Sigmund Freud", "B. F. Skinner", "Ivan Pavlov"],
        "answer": "B",
    },
    {
        "subject": "Social Sciences",
        "question": "How many branches does the United States federal government have?",
        "choices": ["2", "3", "4", "5"],
        "answer": "B",
    },
    {
        "subject": "Social Sciences",
        "question": "The concept of 'cultural diffusion' in sociology refers to what?",
        "choices": [
            "The spread of cultural elements from one group to another",
            "The decline of a culture over time",
            "A government's control over its media",
            "The isolation of a culture from outside influence",
        ],
        "answer": "A",
    },
    {
        "subject": "Social Sciences",
        "question": "What is the capital city of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "C",
    },
    # --- Other (5) ---
    {
        "subject": "Other",
        "question": "How many chambers does the human heart have?",
        "choices": ["2", "3", "4", "5"],
        "answer": "C",
    },
    {
        "subject": "Other",
        "question": "Which branch of government is responsible for interpreting laws?",
        "choices": ["Executive", "Legislative", "Judicial", "Military"],
        "answer": "C",
    },
    {
        "subject": "Other",
        "question": "What is the typical duration of a bachelor's degree in many countries?",
        "choices": ["2 years", "3 years", "4 years", "5 years"],
        "answer": "C",
    },
    {
        "subject": "Other",
        "question": "Which crop is the staple food for the largest share of the global population?",
        "choices": ["Wheat", "Rice", "Corn", "Potato"],
        "answer": "B",
    },
    {
        "subject": "Other",
        "question": "In electricity, what does the abbreviation 'DC' stand for?",
        "choices": ["Direct Current", "Dynamic Charge", "Dual Circuit", "Discharged Cell"],
        "answer": "A",
    },
]


_ANSWER_RE = re.compile(
    r"(?:answer\s*[:\-]?\s*is\s*[:\-]?\s*|answer\s*[:\-]\s*)[\"'*]?([ABCD])[\"'*]?",
    re.IGNORECASE,
)
_FALLBACK_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _format_question(q: dict[str, Any]) -> str:
    choices = q["choices"]
    lines = [f"Question: {q['question']}"]
    for idx, letter in enumerate("ABCD"):
        lines.append(f"{letter}. {choices[idx]}")
    return "\n".join(lines)


def _build_few_shot_prompt(num_few_shot: int) -> str:
    """Build a fixed few-shot demonstration prefix from the first N questions."""
    n = max(0, min(num_few_shot, len(MINI_MMLU_QUESTIONS)))
    blocks: list[str] = []
    for q in MINI_MMLU_QUESTIONS[:n]:
        blocks.append(f"{_format_question(q)}\nAnswer: {q['answer']}")
    return "\n\n".join(blocks)


def _build_prompt(q: dict[str, Any], few_shot_text: str) -> str:
    if few_shot_text:
        return (
            f"{few_shot_text}\n\n"
            f"{_format_question(q)}\nAnswer with a single letter (A, B, C, or D).\nAnswer:"
        )
    return (
        f"{_format_question(q)}\nAnswer with a single letter (A, B, C, or D).\nAnswer:"
    )


def _extract_answer(text: str) -> str | None:
    """Extract the predicted answer letter (A/B/C/D) from model output."""
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _FALLBACK_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def eval_mmlu(
    engine: Any,
    max_samples: int = 100,
    num_few_shot: int = 5,
) -> BenchmarkResult:
    """Run the mini-MMLU 5-shot evaluation.

    Accuracy = correct / total. If the engine is unavailable or returns an
    answer that cannot be parsed, the question is counted as incorrect.
    """
    questions = MINI_MMLU_QUESTIONS[: max(0, min(max_samples, len(MINI_MMLU_QUESTIONS)))]
    few_shot_text = _build_few_shot_prompt(num_few_shot)

    correct = 0
    latencies: list[float] = []
    for q in questions:
        prompt = _build_prompt(q, few_shot_text)
        messages = [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        try:
            out = engine.chat(messages, max_new_tokens=8, temperature=0.0)
            text = out.get("text", "") if isinstance(out, dict) else ""
        except Exception:
            text = ""
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pred = _extract_answer(text or "")
        if pred is not None and pred == q["answer"]:
            correct += 1

    total = len(questions)
    accuracy = (correct / total) if total else 0.0
    latency_avg = (sum(latencies) / len(latencies)) if latencies else 0.0

    return BenchmarkResult(
        task_name="mmlu",
        accuracy=accuracy,
        total_samples=total,
        correct=correct,
        latency_ms_avg=latency_avg,
        throughput_tokens_per_s=0.0,
        metadata={
            "num_few_shot": num_few_shot,
            "subjects": sorted({q["subject"] for q in questions}),
        },
    )
