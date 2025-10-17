# tests/test_math_agent.py
import pytest

from app.agents.math_agent import MathAgent
@pytest.fixture(scope="module")
def agent():
    return MathAgent()  # 需要真实 API KEY 才能跑

@pytest.mark.parametrize("instr, question, prompt, expect_tokens", [
    ("Solve", "Solve: x^2 - 4x + 4 = 0", None, ["<answer>"]),  # Solve 必须给答案
    ("GenerateProblems", "初等极限定理", None, ["<problems>"]),  # 生成题目
    ("MultiMethods", "对 f(x)=x^2 求导", "至少两种不同的方法", ["<methods>"]),  # 多方法解题
    ("ExplainStep", "从 (a-b)(a+b)=a^2-b^2 推下一步", "讲清楚这是平方差展开", ["<step_explanation>"]),  # 步骤解释
])
def test_math_agent_real_api(instr, question, prompt, expect_tokens, agent):
    result = agent.explain(question=question, instruction=instr, prompt=prompt)
    print(f"\n[{instr}] 输出示例：\n", result[:500], "\n")

    # 宽松断言，只要输出里包含关键锚点标签即可
    for token in expect_tokens:
        assert token in result