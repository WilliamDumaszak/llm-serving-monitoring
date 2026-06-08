"""
RAGAS-based LLM evaluation for llm-serving-monitoring.

Metrics:
  faithfulness       — is the answer grounded in the retrieved context?
  answer_relevancy   — is the answer relevant to the question?
  context_precision  — is the retrieved context precise for the question?

The evaluate_from_db() function pulls recent interactions from PostgreSQL
and runs RAGAS on them — enabling continuous quality monitoring.

Requires: ragas>=0.2.0
"""

import logging
import os

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _build_ragas_llm():
    from ragas.llms import LangchainLLMWrapper

    provider = CONFIG["llm"]["provider"]

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", CONFIG["llm"].get("azure_openai_endpoint", "")),
            azure_deployment=CONFIG["llm"]["azure_openai_deployment"],
            api_version=CONFIG["llm"]["azure_openai_api_version"],
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            temperature=0,
        )
    else:
        # local Ollama (default)
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=CONFIG["llm"]["model"],
            base_url=CONFIG["llm"]["ollama_base_url"],
            temperature=0,
        )

    return LangchainLLMWrapper(llm)


def evaluate_with_ragas(samples: list[dict]) -> dict:
    """
    Run RAGAS evaluation on a list of samples.

    Each sample must have:
      question  (str)
      answer    (str)
      contexts  (list[str])  — retrieved passages used to generate the answer
      ground_truth (str, optional)
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    ragas_llm = _build_ragas_llm()
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm)
    context_precision = ContextPrecision(llm=ragas_llm)

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("ground_truth", s["answer"]),
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    try:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        scores = result.to_pandas()
        return {
            "faithfulness": round(float(scores["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"].mean()), 4),
            "context_precision": round(float(scores["context_precision"].mean()), 4),
            "n_samples": len(samples),
        }
    except Exception as exc:
        logger.error(f"RAGAS evaluation failed: {exc}")
        return {"error": str(exc), "n_samples": len(samples)}


def evaluate_from_db(limit: int = 20) -> dict:
    """
    Pull recent interactions from PostgreSQL and run RAGAS on them.
    This enables continuous quality monitoring without manual input.
    """
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=CONFIG["database"]["host"],
            port=CONFIG["database"]["port"],
            user=CONFIG["database"]["user"],
            password=os.getenv("POSTGRES_PASSWORD", CONFIG["database"]["password"]),
            dbname=CONFIG["database"]["dbname"],
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT query, answer FROM interactions ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        logger.error(f"DB query failed: {exc}")
        return {"error": f"DB unavailable: {exc}", "n_samples": 0}

    if not rows:
        return {"error": "No interactions found in database.", "n_samples": 0}

    samples = [
        {"question": row[0], "answer": row[1], "contexts": [row[1]]}
        for row in rows
    ]
    return evaluate_with_ragas(samples)
