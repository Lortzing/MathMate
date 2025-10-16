from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from app.agents import (
    TextbookRAG,
    MockVideoRAG,
    SimpleOCR,
    TemplateMathAgent,
)
from app.graph import RootGraphDeps, build_root_graph

app = FastAPI(title="Multi-Agent Root API")

deps = RootGraphDeps(
    video_rag=MockVideoRAG(),
    textbook_rag=TextbookRAG(),
    ocr_agent=SimpleOCR(),
    math_agent=TemplateMathAgent(),
)
root_graph = build_root_graph(deps)


class AskPayload(BaseModel):
    query: str
    images_b64: List[str] | None = None


@app.post("/ask")
async def ask(payload: AskPayload) -> Dict[str, Any]:
    state: Dict[str, Any] = {"user_query": payload.query}
    if payload.images_b64:
        import base64

        images = [base64.b64decode(item) for item in payload.images_b64]
        state["images"] = images
    output = root_graph.invoke(state)
    return {
        "video": output.get("video", {}),
        "textbook": output.get("textbook", {}),
        "reply_to_user": output.get("reply_to_user", ""),
    }


@app.post("/ask-multipart")
async def ask_multipart(
    query: str = Form(...),
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, Any]:
    images: List[bytes] = []
    if files:
        for file in files:
            images.append(await file.read())
    state: Dict[str, Any] = {"user_query": query}
    if images:
        state["images"] = images
    output = root_graph.invoke(state)
    return {
        "video": output.get("video", {}),
        "textbook": output.get("textbook", {}),
        "reply_to_user": output.get("reply_to_user", ""),
    }
