"""Optional API for the preserved React development frontend."""
import json
import os
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chatbot_core import DEFAULT_MODEL, KnowledgeIndex, build_messages, load_library, stream_answer

load_dotenv()
app = FastAPI(title="Mass Transfer Study API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in os.getenv("CHATBOT_CORS_ORIGINS", "http://localhost:5173").split(",") if s.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

@lru_cache(maxsize=1)
def get_index():
    return KnowledgeIndex(load_library())

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat/stream")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question.")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Configure GROQ_API_KEY on the server.")
    def generate():
        try:
            sources = get_index().search(question)
            messages = build_messages(question, [], sources, "Explain", "Undergraduate")
            for token in stream_answer(key, messages, os.getenv("GROQ_MODEL", DEFAULT_MODEL)):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'error': 'The response could not be completed. Please retry.'})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
