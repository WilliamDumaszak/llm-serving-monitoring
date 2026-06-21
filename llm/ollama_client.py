"""
LLM client — supports Ollama (local), HuggingFace Inference API, Azure OpenAI, and vLLM.
Provider is chosen via config.yaml (llm.provider).

All query_* functions return a 4-tuple:
  (answer: str, elapsed_seconds: float, input_tokens: int, output_tokens: int)

Streaming generators (astream_*) are async generators that yield text chunks
and are used by the /query/stream SSE endpoint.
"""

import logging
import os
import time

import requests
import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def query_ollama(question: str, context: str) -> tuple[str, float, int, int]:
    """Query Ollama REST API. Returns (answer, response_time_seconds, input_tokens, output_tokens)."""
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
    in_tokens = out_tokens = 0
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "").strip()
        in_tokens = data.get("prompt_eval_count", 0)
        out_tokens = data.get("eval_count", 0)
    except Exception as exc:
        logger.error(f"Ollama request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed, in_tokens, out_tokens


def query_huggingface(question: str, context: str) -> tuple[str, float, int, int]:
    """Query HuggingFace Inference API for extractive QA. Token counts not available; returns 0, 0."""
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
    return answer, elapsed, 0, 0


def query_azure_openai(question: str, context: str) -> tuple[str, float, int, int]:
    """Query Azure OpenAI chat completions. Returns (answer, response_time_seconds, input_tokens, output_tokens)."""
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
    in_tokens = out_tokens = 0
    try:
        resp = client.chat.completions.create(
            model=CONFIG["llm"]["azure_openai_deployment"],
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )
        answer = resp.choices[0].message.content.strip()
        if resp.usage:
            in_tokens = resp.usage.prompt_tokens
            out_tokens = resp.usage.completion_tokens
    except Exception as exc:
        logger.error(f"Azure OpenAI request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed, in_tokens, out_tokens


def query_vllm(question: str, context: str) -> tuple[str, float, int, int]:
    """
    Query a vLLM inference server via its OpenAI-compatible API.

    vLLM achieves 2-24x higher throughput than vanilla HuggingFace transformers
    by using PagedAttention for efficient KV cache management.
    The server exposes the same /v1/chat/completions endpoint as OpenAI.
    Returns (answer, response_time_seconds, input_tokens, output_tokens).
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
    in_tokens = out_tokens = 0
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )
        answer = resp.choices[0].message.content.strip()
        if resp.usage:
            in_tokens = resp.usage.prompt_tokens
            out_tokens = resp.usage.completion_tokens
    except Exception as exc:
        logger.error(f"vLLM request failed: {exc}")
        answer = "Error: LLM unavailable."
    elapsed = round(time.time() - t0, 3)
    return answer, elapsed, in_tokens, out_tokens


def query_llm(question: str, context: str) -> tuple[str, float, int, int]:
    """Route to the active provider. Returns (answer, elapsed_s, input_tokens, output_tokens)."""
    provider = CONFIG["llm"]["provider"]
    if provider == "azure_openai":
        return query_azure_openai(question, context)
    if provider == "huggingface":
        return query_huggingface(question, context)
    if provider == "vllm":
        return query_vllm(question, context)
    return query_ollama(question, context)


# ── Streaming generators ───────────────────────────────────────────────────────

async def astream_ollama(question: str, context: str):
    """
    Async generator — yields text chunks from the Ollama streaming API.
    Used by the /query/stream SSE endpoint.
    """
    import json as _json
    import httpx

    prompt = (
        f"You are a helpful assistant. Answer the question based on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    payload = {
        "model": CONFIG["llm"]["model"],
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": CONFIG["llm"]["temperature"],
            "num_predict": CONFIG["llm"]["max_tokens"],
        },
    }
    url = f"{CONFIG['llm']['ollama_base_url']}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        data = _json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            return
    except Exception as exc:
        logger.error(f"Ollama streaming failed: {exc}")
        yield f"[Error: {exc}]"


async def astream_azure_openai(question: str, context: str):
    """
    Async generator — yields text chunks from Azure OpenAI streaming API.
    Used by the /query/stream SSE endpoint.
    """
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
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
    try:
        stream = await client.chat.completions.create(
            model=CONFIG["llm"]["azure_openai_deployment"],
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        logger.error(f"Azure OpenAI streaming failed: {exc}")
        yield f"[Error: {exc}]"


async def astream_llm(question: str, context: str):
    """Route to the correct async streaming provider."""
    provider = CONFIG["llm"]["provider"]
    if provider == "azure_openai":
        async for chunk in astream_azure_openai(question, context):
            yield chunk
    else:
        # Ollama, vLLM (fallback to Ollama-compatible), HuggingFace (no streaming)
        async for chunk in astream_ollama(question, context):
            yield chunk
