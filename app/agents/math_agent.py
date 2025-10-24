from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Union, TypedDict
from openai import OpenAI

from .protocols import MathAgentProtocol, Instruction
from app.config import Config


API_KEY = Config.MATH_API_KEY
BASE_URL = Config.MATH_BASE_URL
MODEL = Config.MATH_MODEL

class TemplatePair(TypedDict):
    system: str
    user: str


INSTRUCTION_TO_TEMPLATES: Dict[str, TemplatePair] = {
    "Solve": {
        "system": (
            "You are Qwen-Math-plus, a rigorous math solver. Be precise with notation. "
            "Prefer LaTeX for formulas. If the problem is ambiguous, state assumptions first."
        ),
        "user": (
            "# Mode: Solve\n"
            "Task: Solve the following problem with a concise but sufficient derivation.\n"
            "- Show key steps only (no hidden leaps).\n"
            "- Simplify fractions/radicals.\n"
            "- If numerical, provide an exact form when possible.\n\n"
            "## Problem\n{question}\n\n"
            "## Output Format\n"
            "- Use Markdown and LaTeX.\n"
            "- Return a detialed answer steps wraped by <answer>, not only the final answer:\n"
            "  <answer> ... </answer>\n\n"
        ),
    },
    "GenerateProblems": {
        "system": (
            "You are Qwen-Math-plus, a math problem designer. Create varied, valid problems. "
            "Balance difficulty and cover typical traps or variations."
        ),
        "user": (
            "# Mode: GenerateProblems\n"
            "Task: Based on the given seed (knowledge topics and/or a seed question), \n"
            "generate a diverse set of problems. Avoid trivial duplicates. \n"
            "Provide a concise correct answer for each.\n\n"
            "## Seed (topic or question)\n{question}\n\n"
            "## Output Format\n"
            "- Return a JSON array wrapped by tags:\n"
            "  <problems>[\n"
            "    {{\n"
            '      "problem": "string (Markdown + LaTeX allowed)",\n'
            "      \"difficulty\": \"easy|medium|hard\",\n"
            "      \"concepts\": [\"string\", \"...\"],\n"
            "      \"answer\": \"string (concise; LaTeX allowed)\"\n"
            "    }},\n"
            "    ...\n"
            "  ]</problems>\n\n"
        ),
    },
    "MultiMethods": {
        "system": (
            "You are Qwen-Math-plus, a rigorous math solver. Be precise with notation. "
        ),
        "user": (
            "# Mode: MultiMethods\n"
            "Task: Provide at least TWO DISTINCT solution methods for the problem.\n"
            "For each method: \n"
            "- outline the idea,\n"
            "- execute steps (no big leaps),\n"
            "- then discuss pros/cons (generality, complexity, robustness).\n\n"
            "## Prompt \n{prompt}\n\n"
            "## Problem\n{question}\n\n"
            "## Output Format\n"
            "- Use Markdown and LaTeX.\n"
            "- Wrap methods within:\n"
            "  <methods>\n"
            "    <method 1>\n"
            "       (idea + steps + pros/cons)\n"
            "    </method 1>\n"
            "    <method 2>\n"
            "       (idea + steps + pros/cons)\n"
            "    </method 2>\n"
            "  </methods>\n"
            "- End with a clearly delimited final answer:\n"
            "  <final_answer> ... </final_answer>\n\n"
        )
    },
    "Heuristic": {
        "system": (
            "You are Qwen-Math-plus, a rigorous math solver. Be precise with notation. "
        ),
        "user": (
            "{prompt}"
            "{question}"
        ),
    },
    "ExplainStep": {
        "system": (
            "You are Qwen-Math-plus, a didactic explainer. "
            "For each micro-step, state goal → operation → principle/theorem used."
        ),
        "user": (
            "# Mode: ExplainStep\n"
            "Task: Explain the SPECIFIC step or transformation in detail.\n"
            "Structure each micro-reasoning as:\n"
            "1) Goal,\n"
            "2) Operation (algebraic/logic),\n"
            "3) Principle or theorem referenced,\n"
            "4) (If applicable) conditions under which it holds.\n\n"
            "## Context Problem (optional)\n{prompt}\n\n"
            "## Specifict Step\n{question}\n\n"
            "## Output Format\n"
            "- Use Markdown and LaTeX.\n"
            "- Return the focused explanation wrapped as:\n"
            "  <step_explanation>\n"
            "  (your explanation)\n"
            "  </step_explanation>\n\n"
        ),
    },
}


def _build_messages(instr: Instruction, question: str, prompt: Optional[str]) -> list[dict]:
    tpl = INSTRUCTION_TO_TEMPLATES[instr]
    system = tpl["system"]
    user = tpl["user"].format(
        question=(question or "N/A"),
        prompt=(prompt or "N/A"),
    )
    print(user)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class MathAgent(MathAgentProtocol):
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    def _call_model(self, messages: list[dict]) -> str | None:
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        try:
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print("Completion object for debugging:", completion)
            raise e

    def explain(
        self,
        question: str,
        instruction: Instruction,
        prompt: str | None
    ) -> str:
        """
        统一入口：
        - Solve（解题）
        - GenerateProblems（根据知识点/问题出题）
        - MultiMethods（使用多种方式解题）
        - ExplainStep（解释该步骤）

        约定的输出锚点（用于上游解析）：
        - Solve / MultiMethods: 必有 <answer>…</answer>
        - MultiMethods: 应有 <methods>…</methods>
        - GenerateProblems: 必有 <problems>[...]</problems>
        - ExplainStep: 必有 <explanation>…</explanation>
        """
        messages = _build_messages(instruction, question, prompt)
        result = self._call_model(messages) or ""
        print(result)

        return str(result).strip()
