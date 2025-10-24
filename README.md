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


我们来讲解积分 $\int \frac{x}{(3x+4)^2} \, dx$ 的解题思路。

这个积分的被积函数是一个有理分式，分子是一次式 $x$，分母是 $(3x+4)^2$，属于复合函数的平方形式。直接积分不容易，因此我们需要通过适当的技巧来简化它。常用的方法包括**代换法**和**分部积分法**。这里采用**分部积分法**配合**代换法**来求解，步骤如下：

---

### **第一步：设定分部积分**

我们使用分部积分公式：
$$
\int u \, dv = uv - \int v \,du
$$

选择：
- $u = x$，则 $du = dx$
- $dv = \frac{1}{(3x+4)^2} dx$，需要先求出 $v = \int \frac{1}{(3x+4)^2} dx$

---


### **第二步：计算 $v = \int \frac{1}{(3x+4)^2} dx$**

使用代换法：
令 $t = 3x + 4$，则 $dt = 3dx$，即 $dx = \frac{1}{3} dt$

代入得：
$$v = \int \frac{1}{t^2} \cdot \frac{1}{3} dt = \frac{1}{3} \int t^{-2} dt = \frac{1}{3} \left( -\frac{1}{t} \right) = -\frac{1}{3t}
$$

代回 $t = 3x+4$ 得：
$$
v = -\frac{1}{3(3x+4)}
$$

---

### **第三步：代入分部积分公式**

$$
\int \frac{x}{(3x+4)^2} dx = uv - \int v \, du = x \cdot \left( -\frac{1}{3(3x+4)} \right) - \int \left( -\frac{1}{3(3x+4)} \right) dx
$$

化简：
$$
= -\frac{x}{3(3x+4)} + \frac{1}{3} \int \frac{1}{3x+4} dx
$$

---

### **第四步：计算 $\int \frac{1}{3x+4} dx$**

再次使用代换：令 $t = 3x + 4$，则 $dx = \frac{1}{3} dt$

$$
\int \frac{1}{3x+4} dx = \int \frac{1}{t} \cdot \frac{1}{3} dt = \frac{1}{3} \ln|t| = \frac{1}{3} \ln|3x+4|
$$

代入原式：
$$
-\frac{x}{3(3x+4)} + \frac{1}{3} \cdot \frac{1}{3} \ln|3x+4| = -\frac{x}{3(3x+4)} + \frac{1}{9} \ln|3x+4|
$$

---

### **最终结果**

$$
\boxed{\int \frac{x}{(3x+4)^2} \, dx = -\frac{x}{3(3x+4)} + \frac{1}{9} \ln|3x+4| + C}
$$

---

### **思路总结**

1. 观察到分母是复合函数的平方，考虑用代换或分部积分。
2. 分子是 $x$，不能直接拆成导数形式，所以尝试**分部积分**，把 $x$ 作为 $u$，其余部分作为 $dv$。
3. 计算 $v$ 时用**代换法**处理 $\int \frac{1}{(3x+4)^2} dx$。
4. 分部后剩下的积分 $\int \frac{1}{3x+4} dx$ 再次用代换法解决。
5. 最终合并结果，得到原函数。

这种方法结合了两种基本积分技巧，适用于这类“线性复合函数作分母 + 多项式分子”的情形。