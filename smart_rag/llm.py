"""Pluggable generation backends.

`generate(prompt) -> str` is the only method the pipeline needs. Two
implementations are provided:

- MockLLM: deterministic, offline, zero-dependency -- lets you run and test
  the whole RAG pipeline (retrieval, prompting, citation formatting)
  without any API key. Useful for CI and for demoing retrieval quality in
  isolation from generation quality.
- AnthropicLLM: thin wrapper around the Anthropic Messages API for real
  contextual answers, used when ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import os
import textwrap
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        ...


class MockLLM(LLMBackend):
    """Offline stand-in that extractively summarizes the supplied context.

    It doesn't reason, it just demonstrates that the retrieved context is
    flowing correctly into the answer -- swap in AnthropicLLM for real
    generation quality.
    """

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if "Context:" in prompt and "Question:" in prompt:
            context = prompt.split("Context:", 1)[1].split("Question:")[0].strip()
            question = prompt.split("Question:", 1)[1].strip()
            snippet = textwrap.shorten(context, width=400, placeholder=" ...")
            return (
                f"[mock answer -- no LLM configured]\n"
                f"Based on the retrieved context, here is a relevant excerpt "
                f"addressing '{question}':\n\n{snippet}"
            )
        return "[mock answer -- no LLM configured]"


class AnthropicLLM(LLMBackend):
    """Generates answers using the Anthropic Messages API.

    Requires `pip install anthropic` and an `ANTHROPIC_API_KEY` environment
    variable (or pass api_key explicitly).
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicLLM requires the `anthropic` package: pip install anthropic"
            ) from e
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
