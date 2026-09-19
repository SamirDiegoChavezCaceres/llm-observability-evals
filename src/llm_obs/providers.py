"""Model-backed `complete` functions for ``LLMJudge``.

``LLMJudge`` takes any ``complete(prompt) -> str`` callable (the Strategy
pattern), so the judge stays provider-agnostic. This module supplies an
OpenAI-backed one; another provider is just another function with the same
shape.
"""

from __future__ import annotations

import os
from typing import Callable


def openai_complete(model: str = None) -> Callable[[str], str]:
    """A `complete(prompt) -> str` backed by an OpenAI chat model.

    Reads ``OPENAI_API_KEY`` from the environment or a local ``.env`` file, and
    asks the model for JSON so the judge can parse per-criterion scores.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    from openai import OpenAI  # lazy import

    client = OpenAI()
    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
