from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    language: str
    route: str
    sources: list[dict]
    graph_facts: list[dict]


class IngestRequest(BaseModel):
    text: str
    source: str = "upload"


class IngestResponse(BaseModel):
    chunk_id: str
    entities_found: int
