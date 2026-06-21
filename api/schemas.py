from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, example="What is logistics?")


class QueryResponse(BaseModel):
    doc_id: str
    answer: str
    response_time_ms: float
    hit_rate: float
    mrr: float
    cache_hit: bool = False


class FeedbackRequest(BaseModel):
    doc_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


class HealthResponse(BaseModel):
    status: str
    elasticsearch_ready: bool
    database_ready: bool
    cache_ready: bool = False


class RagasSample(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str | None = None


class RagasEvalResponse(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    n_samples: int
    error: str | None = None
