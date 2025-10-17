from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from dotenv import load_dotenv
import uvicorn

from app.graph import create_default_graph

load_dotenv()

app = FastAPI(title="MathMate Copilot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

root_graph = create_default_graph()

def _agent_name() -> str:
    return os.getenv("COPILOT_AGENT_NAME", "mathmate")


def _agent_path(default_name: str) -> str:
    return os.getenv("COPILOT_AGENT_PATH", f"/agents/{default_name}")


agent_name = _agent_name()
agent_path = _agent_path(agent_name)

add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name=agent_name,
        description="MathMate LangGraph workflow exposed for CopilotKit.",
        graph=root_graph,
    ),
    path=agent_path,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for deployment probes."""
    return {"status": "ok"}


def main() -> None:
    """Run the FastAPI application with Uvicorn."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.copilot_server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
