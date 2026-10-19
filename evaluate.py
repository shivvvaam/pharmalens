import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

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

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, embeddings, get_collection_name, qdrant_client

load_dotenv()

PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7
JUDGE_MODEL         = "gemini-2.5-flash"
RATE_LIMIT_RETRIES  = 5
RATE_LIMIT_WAIT_S   = 30  # Groq's on-demand TPM window is per minute
ANSWERS_FILE        = Path("eval_answers.json")
JUDGE_MIN_INTERVAL_S = 13  # Gemini free tier: 5 requests/min per model
QUICK_EVAL          = True  # False runs all 5 metrics on every case (~100+ judge calls)
QUICK_EVAL_CASES    = 5


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


def generate_goldens(judge: GeminiModel) -> list[dict]:
    synthesizer = Synthesizer(model=judge)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
            embedder=GeminiEmbedder(),
            critic_model=judge,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
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


def generate_answers(pairs: list[dict]) -> list[dict]:
    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    # Ingest the paper once into a single throwaway collection; each query gets its
    # own checkpoint thread so conversation history doesn't leak between goldens.
    session_id = f"evaluation_session_{uuid4()}"
    add_paper(docs, session_id)

    answers = []
    try:
        for pair in pairs:
            query = pair["input"] + " as per the report in knowledge base"
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
    judge = ThrottledGeminiModel(model=JUDGE_MODEL)
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens(judge)

    # Answers are cached so a failed judging run doesn't re-spend the Groq quota.
    # Delete eval_answers.json to re-run the RAG pipeline after changing it.
    if ANSWERS_FILE.exists():
        answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
        print(f"Loaded {len(answers)} cached answers from {ANSWERS_FILE}")
    else:
        answers = generate_answers(pairs)

    if QUICK_EVAL:
        # Fits Gemini's free tier (20 requests/day): one retrieval metric, one generation
        # metric, no reason strings (each reason is an extra judge call). ~4 calls/case.
        metrics = [
            ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
            FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        ]
        answers = answers[:QUICK_EVAL_CASES]
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

    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_path}.")


if __name__ == "__main__":
    main()
