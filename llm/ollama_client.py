"""
LLM client — supports Ollama (local), HuggingFace Inference API, and Azure OpenAI.
Provider is chosen via config.yaml (llm.provider).
"""

import logging
import os
import time

import requests
import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def query_ollama(question: str, context: str) -> tuple[str, float]:
    """Query Ollama REST API. Returns (answer, response_time_seconds)."""
    prompt = (
        f"You are a helpful assistant. Answer the question based on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    payload = {
        "model": CONFIG["llm"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": CONFIG["llm"]["temperature"],
            "num_predict": CONFIG["llm"]["max_tokens"],
        },
    }
    url = f"{CONFIG['llm']['ollama_base_url']}/api/generate"
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
    except Exception as exc:
        logger.error(f"Ollama request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed


def query_huggingface(question: str, context: str) -> tuple[str, float]:
    """Query HuggingFace Inference API for extractive QA."""
    api_url = f"https://api-inference.huggingface.co/models/{CONFIG['llm']['hf_model']}"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_KEY', '')}"}
    payload = {"inputs": {"question": question, "context": context}}
    t0 = time.time()
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
    except Exception as exc:
        logger.error(f"HuggingFace request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed


def query_azure_openai(question: str, context: str) -> tuple[str, float]:
    """Query Azure OpenAI chat completions. Returns (answer, response_time_seconds)."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", CONFIG["llm"].get("azure_openai_endpoint", "")),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=CONFIG["llm"]["azure_openai_api_version"],
    )
    prompt = (
        f"You are a helpful assistant. Answer the question based on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=CONFIG["llm"]["azure_openai_deployment"],
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"Azure OpenAI request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed


def query_vllm(question: str, context: str) -> tuple[str, float]:
    """
    Query a vLLM inference server via its OpenAI-compatible API.

    vLLM achieves 2-24x higher throughput than vanilla HuggingFace transformers
    by using PagedAttention for efficient KV cache management.
    The server exposes the same /v1/chat/completions endpoint as OpenAI.
    """
    from openai import OpenAI

    base_url = os.getenv("VLLM_BASE_URL", CONFIG["llm"].get("vllm_base_url", "http://vllm:8001/v1"))
    model_id = CONFIG["llm"].get("vllm_model", "meta-llama/Llama-3.2-1B-Instruct")

    client = OpenAI(api_key="no-key", base_url=base_url)
    prompt = (
        f"You are a helpful assistant. Answer the question based on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"vLLM request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed


def query_llm(question: str, context: str) -> tuple[str, float]:
    provider = CONFIG["llm"]["provider"]
    if provider == "azure_openai":
        return query_azure_openai(question, context)
    if provider == "huggingface":
        return query_huggingface(question, context)
    if provider == "vllm":
        return query_vllm(question, context)
    return query_ollama(question, context)
