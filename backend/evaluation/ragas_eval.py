"""
LLM-as-judge evaluation for the RAG pipeline.

Replaces the word-overlap heuristics with actual LLM judgment calls.
Each metric costs a single Groq/Grok/OpenAI completion (fractions of a cent)
and gives a number you can actually trust vs. a keyword overlap approximation.

Metrics:
  Faithfulness      — Does the answer stay within the retrieved context?
  Answer Relevancy  — Does the answer address the question asked?
  Context Precision — Were the retrieved chunks actually useful?

Usage:
  Via API:  POST /api/evaluate  { "file_id": "abc123", "questions": [...] }
  Via CLI:  python -m evaluation.ragas_eval --file_id abc123
"""

import re
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from retrieval.searcher import search, format_results_as_context
from generation.generator import generate_answer_full


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    question: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    faithfulness: float = 0.0
    relevancy: float = 0.0
    context_precision: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalReport:
    file_id: str
    total_questions: int = 0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    samples: list[dict] = field(default_factory=list)


# ── LLM judge helper ──────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a rigorous evaluation judge for RAG (retrieval-augmented generation) systems. "
    "When asked to score something, you respond with ONLY a decimal number between 0.0 and 1.0. "
    "No explanations, no reasoning, just the score."
)

_FAITHFULNESS_PROMPT = """\
Rate the FAITHFULNESS of the following answer.

DEFINITION: A faithful answer contains ONLY information that is present in or
directly supported by the provided context. It does not add facts, make up
numbers, or speculate beyond the context.

SCORING:
  1.0 = Every claim in the answer is explicitly supported by the context
  0.7 = Most claims are supported; minor inferences present
  0.4 = Some claims are supported but significant unsupported content
  0.0 = Answer contradicts or entirely ignores the context

Context:
{context}

Question: {question}
Answer: {answer}

Score (0.0–1.0):"""

_RELEVANCY_PROMPT = """\
Rate the ANSWER RELEVANCY of the following response.

DEFINITION: A relevant answer directly addresses what was asked. It does not
answer a different question, give a generic response, or say "I don't know"
when the information is clearly present in the answer.

SCORING:
  1.0 = Answer directly and completely addresses the question
  0.7 = Answer mostly addresses the question with minor gaps
  0.4 = Answer partially addresses the question or is off-topic
  0.0 = Answer is entirely unrelated or refuses to answer

Question: {question}
Answer: {answer}

Score (0.0–1.0):"""

_CONTEXT_PRECISION_PROMPT = """\
Rate the CONTEXT PRECISION for the following retrieval result.

DEFINITION: Context precision measures whether the retrieved chunks are
actually relevant for answering the question. High precision means every
retrieved chunk was useful. Low precision means irrelevant chunks were
returned (noise in the retrieval).

SCORING:
  1.0 = Every retrieved chunk is clearly relevant to the question
  0.7 = Most chunks are relevant; 1–2 are tangential
  0.4 = About half the chunks are relevant; significant noise
  0.0 = Retrieved chunks are entirely unrelated to the question

Question: {question}

Retrieved contexts:
{contexts_numbered}

Score (0.0–1.0):"""


async def _llm_judge(prompt: str) -> float:
    """
    Ask the configured LLM to score something. Returns float 0.0–1.0.
    Falls back to 0.5 if the response cannot be parsed.
    """
    from core.llm import get_llm
    llm = get_llm()

    tokens: list[str] = []
    try:
        async for token in llm.stream_answer(
            question=prompt,
            context="",
            system_prompt=_JUDGE_SYSTEM,
        ):
            tokens.append(token)
    except Exception:
        return 0.5

    response = "".join(tokens).strip()

    # Parse the first decimal in the response and clamp to [0, 1]
    matches = re.findall(r'\b([01](?:\.\d+)?|\.\d+)\b', response)
    if matches:
        try:
            return min(1.0, max(0.0, float(matches[0])))
        except ValueError:
            pass

    return 0.5


# ── Per-metric scorers ────────────────────────────────────────────────────────

async def _score_faithfulness(
    question: str, answer: str, contexts: list[str]
) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    context_blob = "\n\n".join(contexts[:4])  # cap to avoid huge prompts
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context_blob[:3000],
        question=question,
        answer=answer[:1500],
    )
    return await _llm_judge(prompt)


async def _score_relevancy(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer[:1500])
    return await _llm_judge(prompt)


async def _score_context_precision(
    question: str, contexts: list[str]
) -> float:
    if not contexts:
        return 0.0
    numbered = "\n".join(
        f"[{i+1}] {ctx[:400]}" for i, ctx in enumerate(contexts[:5])
    )
    prompt = _CONTEXT_PRECISION_PROMPT.format(
        question=question, contexts_numbered=numbered
    )
    return await _llm_judge(prompt)


# ── Evaluation pipeline ───────────────────────────────────────────────────────

async def evaluate_question(question: str, file_id: str) -> EvalSample:
    """Run the full RAG pipeline on one question and score all three metrics."""
    sample = EvalSample(question=question)
    try:
        result = await generate_answer_full(question=question, file_id=file_id)
        sample.answer = result.get("answer", "")

        search_results = await search(query=question, file_id=file_id)
        sample.contexts = [r.text for r in search_results]

        # Run the three LLM judge calls concurrently
        faith, relevancy, precision = await asyncio.gather(
            _score_faithfulness(question, sample.answer, sample.contexts),
            _score_relevancy(question, sample.answer),
            _score_context_precision(question, sample.contexts),
        )
        sample.faithfulness      = round(faith,     3)
        sample.relevancy         = round(relevancy,  3)
        sample.context_precision = round(precision,  3)

    except Exception as e:
        sample.error = str(e)

    return sample


async def evaluate_file(file_id: str, questions: list[str]) -> EvalReport:
    """Evaluate the RAG pipeline on a set of questions for a specific file."""
    report = EvalReport(file_id=file_id, total_questions=len(questions))
    samples: list[EvalSample] = []

    for q in questions:
        sample = await evaluate_question(q, file_id)
        samples.append(sample)

    valid = [s for s in samples if s.error is None]
    if valid:
        report.avg_faithfulness      = round(sum(s.faithfulness      for s in valid) / len(valid), 3)
        report.avg_relevancy         = round(sum(s.relevancy         for s in valid) / len(valid), 3)
        report.avg_context_precision = round(sum(s.context_precision for s in valid) / len(valid), 3)

    report.samples = [asdict(s) for s in samples]
    return report


# ── CLI runner ────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "How many rows are in this dataset?",
    "What are the column names?",
    "What is the most common value in the first column?",
    "Show me a summary of the data.",
    "How many unique values are there?",
]


async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for DataRAG")
    parser.add_argument("--file_id",   required=True, help="File ID to evaluate")
    parser.add_argument("--questions", nargs="*",     help="Custom questions")
    parser.add_argument("--output",    default=None,  help="Output JSON path")
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS
    print(f"Evaluating {len(questions)} questions against file '{args.file_id}'...\n")

    from db.database import init_db
    from core.config import settings
    await init_db(settings.db_path)

    report = await evaluate_file(args.file_id, questions)

    print("=" * 60)
    print(f"  Evaluation Report — file: {report.file_id}")
    print("=" * 60)
    print(f"  Questions evaluated: {report.total_questions}")
    print(f"  Avg Faithfulness:    {report.avg_faithfulness:.1%}")
    print(f"  Avg Relevancy:       {report.avg_relevancy:.1%}")
    print(f"  Avg Context Prec:    {report.avg_context_precision:.1%}")
    print("=" * 60)

    for i, s in enumerate(report.samples):
        status = "ERROR" if s.get("error") else "OK"
        print(f"\n  Q{i+1} [{status}]: {s['question'][:60]}")
        if s.get("error"):
            print(f"       Error: {s['error']}")
        else:
            print(
                f"       Faith={s['faithfulness']:.0%}  "
                f"Rel={s['relevancy']:.0%}  "
                f"CtxPrec={s['context_precision']:.0%}"
            )
            print(f"       Answer: {s['answer'][:100]}...")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(asdict(report), indent=2, default=str))
        print(f"\nFull report saved to {out}")


if __name__ == "__main__":
    asyncio.run(_cli_main())
