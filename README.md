# MathMate

This project demonstrates how to orchestrate a multi-agent workflow with
[LangGraph](https://github.com/langchain-ai/langgraph). The system features a
root supervisor agent that coordinates several specialized tools to answer user
math questions.

## Architecture Overview

The root supervisor (powered by **qwen3-plus**) calls an LLM every time it needs
to decide what happens next. The model receives a sanitized view of the shared
state and returns a structured decision:

```json
{"next": "math", "params": {"instruction": "Solve"}, "reason": "..."}
```

All tool parameters (for OCR, query construction, retrieval, and the math
solver) flow through this JSON payload, so you can dynamically steer the entire
pipeline from a single place. When the model is unavailable, a deterministic
fallback policy keeps the workflow running.

The supervisor can invoke the following components:

- **video_rag** – retrieves related instructional videos.
- **textbook_rag** – retrieves textbook references.
- **ocr_agent** – transcribes user-provided images to text (the root agent
  itself is not multimodal).
- **math_agent** – synthesizes the final explanation using the user's
  question, OCR text, and RAG results.

Every response conforms to the structure:

```json
{
  "video": {...},
  "textbook": {...},
  "reply_to_user": "..."
}
```

## Project Layout

```
app/
├── agents/          # Tool and agent implementations
├── graph/           # LangGraph state, supervisor, and builder
│   ├── builder.py    # LangGraph wiring + tool nodes
│   ├── state.py      # Shared state definition
│   └── supervisor.py # Root supervisor (LLM-driven routing)
├── utils/           # Shared utilities (logging, image helpers)
├── api_server.py    # Optional FastAPI server (JSON + multipart endpoints)
├── config.py        # Runtime configuration dataclass
├── main.py          # CLI entry point
└── requirements.txt # Runtime dependencies
```

## Installation

Create a virtual environment and install dependencies with
[uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r app/requirements.txt
```

If you prefer to keep everything in sync with the `pyproject.toml`, you can
also run:

```bash
uv sync
```

## Usage

### Command Line

```bash
python -m app.main --query "讲解 ∫ x/(3x+4)^2 dx 的思路"
python -m app.main --query "根据图片讲解积分策略" --images ./samples/problem.png
```

### FastAPI Server (optional)

```bash
uvicorn app.api_server:app --reload --port 8000
```

Send requests with JSON or multipart form data:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "如何积分 x/(3x+4)^2", "images_b64": []}'
```

```bash
curl -X POST http://localhost:8000/ask-multipart \
  -F "query=根据图片讲解积分策略" \
  -F "files=@/path/to/problem.png"
```

## Extending the System

- Replace mock RAG implementations in `app/agents/video_rag.py` and
  `app/agents/textbook_rag.py` with your retrieval backend.
- Swap the OCR implementation in `app/agents/ocr_agent.py` for a real OCR
  engine (e.g. Tesseract, PaddleOCR, or a multimodal model).
- Connect the math agent (`app/agents/math_agent.py`) to your preferred LLM
  for detailed explanations.
- Adjust the routing policy in `app/graph/supervisor.py` if you want to prompt
  qwen3-plus differently or plug in additional tools.

## License

MIT