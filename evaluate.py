import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Retrieval settings under comparison. Each run evaluates one variant against the same
# cached pharma goldens, so scores are directly comparable (see eval/compare.py).
VARIANTS = {
    "baseline":     {},  # app defaults: 1000/200 chunks, plain top-k similarity
    "small_chunks": {"PAPEER_CHUNK_SIZE": "500", "PAPEER_CHUNK_OVERLAP": "100"},
    "score_filter": {"PAPEER_MIN_SCORE": "0.65"},
    "combined":     {"PAPEER_CHUNK_SIZE": "500", "PAPEER_CHUNK_OVERLAP": "100", "PAPEER_MIN_SCORE": "0.65"},
}

parser = argparse.ArgumentParser(description="Evaluate PharmaLens retrieval on pharma papers.")
parser.add_argument("--variant", choices=VARIANTS, default="baseline")
parser.add_argument("--full", action="store_true", help="run all 5 metrics on every case (~100+ judge calls)")
parser.add_argument("--cases", type=int, default=6, help="cases judged in quick mode")
ARGS = parser.parse_args()
# Backend modules read these at import time, so they must be set before the imports below.
os.environ.update(VARIANTS[ARGS.variant])

import groq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseEmbeddingModel, GeminiModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from backend.paper_loader import load_arxiv
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, embeddings, get_collection_name, qdrant_client

load_dotenv()

# Open-access arXiv papers spanning clinical trials, drug discovery and drug molecules.
PAPERS = {
    "2307.15051": "Matching Patients to Clinical Trials with Large Language Models (TrialGPT)",
    "2304.05376": "ChemCrow: Augmenting large-language models with chemistry tools",
    "2309.03907": "DrugChat: Towards Enabling ChatGPT-Like Capabilities on Drug Molecule Graphs",
}
EVAL_DIR            = Path("eval")
GOLDENS_FILE        = EVAL_DIR / "goldens_pharma.json"
CONTEXTS_PER_PAPER  = 2
CHUNKS_PER_CONTEXT  = 3
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7
JUDGE_MODEL         = "gemini-2.5-flash"
RATE_LIMIT_RETRIES  = 5
RATE_LIMIT_WAIT_S   = 30  # Groq's on-demand TPM window is per minute
ANSWERS_FILE        = EVAL_DIR / f"answers_{ARGS.variant}.json"
RESULTS_FILE        = EVAL_DIR / f"results_{ARGS.variant}.json"
JUDGE_MIN_INTERVAL_S = 13  # Gemini free tier: 5 requests/min per model


class ThrottledGeminiModel(GeminiModel):
    """GeminiModel that spaces out calls to stay under the free-tier request limit."""

    _lock = threading.Lock()
    _next_slot = 0.0
    calls = 0

    def _reserve_slot(self) -> float:
        with self._lock:
            now = time.monotonic()
            slot = max(now, ThrottledGeminiModel._next_slot)
            ThrottledGeminiModel._next_slot = slot + JUDGE_MIN_INTERVAL_S
            ThrottledGeminiModel.calls += 1
            return slot - now

    def generate(self, *args, **kwargs):
        time.sleep(self._reserve_slot())
        return super().generate(*args, **kwargs)

    async def a_generate(self, *args, **kwargs):
        await asyncio.sleep(self._reserve_slot())
        return await super().a_generate(*args, **kwargs)


class GeminiEmbedder(DeepEvalBaseEmbeddingModel):
    """Adapts the app's cached Gemini embeddings to DeepEval's embedder interface."""

    def __init__(self):
        super().__init__(model="gemini-embedding-001")

    def load_model(self):
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.model.aembed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.name


def load_papers() -> dict[str, list]:
    return {arxiv_id: load_arxiv(arxiv_id) for arxiv_id in PAPERS}


def build_contexts(chunks: list, rng: random.Random) -> list[list[str]]:
    """Pick windows of consecutive chunks from each paper's body, skipping the title/author
    block at the start and the references at the end, which make poor questions."""
    body = chunks[2 : int(len(chunks) * 0.8)]
    starts = range(0, max(len(body) - CHUNKS_PER_CONTEXT, 0) + 1, CHUNKS_PER_CONTEXT)
    picked = sorted(rng.sample(list(starts), min(CONTEXTS_PER_PAPER, len(starts))))
    return [[c.page_content for c in body[i : i + CHUNKS_PER_CONTEXT]] for i in picked]


def generate_goldens(judge: GeminiModel, papers: dict[str, list]) -> list[dict]:
    synthesizer = Synthesizer(model=judge)
    rng = random.Random(42)
    per_paper = []
    for arxiv_id, chunks in papers.items():
        goldens = synthesizer.generate_goldens_from_contexts(
            contexts=build_contexts(chunks, rng),
            include_expected_output=True,
            max_goldens_per_context=GOLDENS_PER_CONTEXT,
        )
        per_paper.append([
            {"source": arxiv_id, "input": g.input, "expected_output": g.expected_output}
            for g in goldens
            if g.input and g.expected_output
        ])
    # Interleave papers so a quick run's first N cases cover every paper.
    shortest = min(map(len, per_paper))
    pairs = [p for group in zip(*per_paper) for p in group]
    pairs += [p for group in per_paper for p in group[shortest:]]
    EVAL_DIR.mkdir(exist_ok=True)
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str, session_id: str, thread_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def generate_answers(pairs: list[dict], papers: dict[str, list]) -> list[dict]:
    graph = build_graph(db_path="eval_checkpoints.db")

    # Ingest the papers once into a single throwaway collection; each query gets its
    # own checkpoint thread so conversation history doesn't leak between goldens.
    session_id = f"evaluation_session_{uuid4()}"
    for chunks in papers.values():
        add_paper(chunks, session_id)

    answers = []
    try:
        for pair in pairs:
            query = pair["input"] + " as per the papers in the knowledge base"
            for attempt in range(1, RATE_LIMIT_RETRIES + 1):
                try:
                    # Fresh thread per attempt so a half-finished run doesn't leave history behind.
                    answer, retrieval_context = run_rag_query(graph, query, session_id, str(uuid4()))
                    break
                except groq.RateLimitError:
                    if attempt == RATE_LIMIT_RETRIES:
                        raise
                    print(f"Groq rate limit hit, waiting {RATE_LIMIT_WAIT_S}s (attempt {attempt}/{RATE_LIMIT_RETRIES})")
                    time.sleep(RATE_LIMIT_WAIT_S)
            print(f"Answered {len(answers) + 1}/{len(pairs)}")
            answers.append({
                "input": pair["input"],
                "actual_output": answer,
                "expected_output": pair["expected_output"],
                "retrieval_context": retrieval_context,
            })
    finally:
        qdrant_client.delete_collection(get_collection_name(session_id))

    ANSWERS_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    return answers


def main() -> None:
    print(f"Variant: {ARGS.variant} {VARIANTS[ARGS.variant] or '(app defaults)'}")
    judge = ThrottledGeminiModel(model=JUDGE_MODEL)

    # Answers are cached so a failed judging run doesn't re-spend the Groq quota.
    # Delete the variant's answers file to re-run the RAG pipeline after changing it.
    if ANSWERS_FILE.exists():
        answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
        print(f"Loaded {len(answers)} cached answers from {ANSWERS_FILE}")
    else:
        # Papers are chunked with this variant's settings; goldens are generated once and
        # shared by every variant so all runs answer the same questions.
        papers = load_papers()
        pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens(judge, papers)
        answers = generate_answers(pairs, papers)

    if not ARGS.full:
        # Fits Gemini's free tier: Contextual Relevancy (the weakest metric on the earlier
        # run) plus Faithfulness, without reason strings (each reason is an extra judge call).
        metrics = [
            ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
            FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        ]
        answers = answers[:ARGS.cases]
    else:
        metrics = [
            ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
            ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
            ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
            AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
            FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
        ]
    test_cases = [LLMTestCase(**a) for a in answers]

    # Sequential so the judge's own throttle is the only thing pacing requests.
    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
    )
    print(f"Judge calls used: {ThrottledGeminiModel.calls}")

    summary = []
    for test_result in results.test_results:
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}. Compare variants with: uv run python eval/compare.py")


if __name__ == "__main__":
    main()
