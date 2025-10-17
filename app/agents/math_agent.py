from __future__ import annotations

from openai import OpenAI

from typing import Any, Dict, Optional, Literal
from .protocols import MathAgent as MathAgentProtocol
from app.config import Config


API_KEY = Config.MATH_API_KEY
BASE_URL = Config.MATH_BASE_URL
MODEL = Config.MATH_MODEL


class MathAgent:    
    def __init__(self):        
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )    
    def _call_model(self, messages: dict) -> str | None:
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': 'Derive a universal solution for the quadratic equation $ Ax^2+Bx+C=0 $'}])
        try: 
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(completion)
            raise e
        

    def explain(
        self,
        question: str,
        instruction: Literal["Answer", "Explain",  "Multi-ideas"],
        prompt: str | None
    ) -> str:
        ...
