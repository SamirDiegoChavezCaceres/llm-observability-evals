"""LLM-as-judge evaluation.

Two judges behind one interface:

* ``HeuristicJudge`` - deterministic and offline. Groundedness is the share of
  the answer's content words that appear in the retrieved context; relevance is
  the share of the question's content words the answer picks up. Good for tests
  and a cheap regression gate.
* ``LLMJudge`` - sends a rubric to a model (any ``complete(prompt) -> str``
  callable) and parses back per-criterion scores. The model is injected, so the
  harness is provider-agnostic and testable with a fake.

``evaluate_dataset`` runs a judge over many samples and aggregates mean scores
and a pass rate per criterion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from in is it of on or the to with".split())


def _content_tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


def _coverage(needles: set, haystack: set) -> float:
    if not needles:
        return 0.0
    return len(needles & haystack) / len(needles)


@dataclass
class EvalSample:
    input: str
    output: str
    context: str = ""


@dataclass
class Score:
    name: str
    value: float
    passed: bool
    comment: str = ""


class Judge(Protocol):
    def score(self, sample: EvalSample) -> List[Score]: ...


class HeuristicJudge:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, sample: EvalSample) -> List[Score]:
        out = _content_tokens(sample.output)
        groundedness = _coverage(out, _content_tokens(sample.context))
        relevance = _coverage(_content_tokens(sample.input), out)
        return [
            Score("groundedness", round(groundedness, 3), groundedness >= self.threshold),
            Score("relevance", round(relevance, 3), relevance >= self.threshold),
        ]


_RUBRIC = """You are grading an answer. Score each criterion from 0.0 to 1.0.
Return ONLY JSON like {{"groundedness": 0.0, "relevance": 0.0}}.

Question: {input}
Context: {context}
Answer: {output}
"""


class LLMJudge:
    def __init__(self, complete: Callable[[str], str], threshold: float = 0.7) -> None:
        self._complete = complete
        self.threshold = threshold

    def score(self, sample: EvalSample) -> List[Score]:
        prompt = _RUBRIC.format(input=sample.input, context=sample.context, output=sample.output)
        raw = self._complete(prompt)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [Score("parse_error", 0.0, False, comment=f"non-JSON judge output: {raw!r}")]
        scores = []
        for name, value in data.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            scores.append(Score(name, value, value >= self.threshold))
        return scores


@dataclass
class CriterionSummary:
    mean: float
    pass_rate: float
    n: int


def evaluate_dataset(samples: List[EvalSample], judge: Judge) -> Dict[str, CriterionSummary]:
    totals: Dict[str, List[Score]] = {}
    for sample in samples:
        for score in judge.score(sample):
            totals.setdefault(score.name, []).append(score)
    summary: Dict[str, CriterionSummary] = {}
    for name, scores in totals.items():
        n = len(scores)
        summary[name] = CriterionSummary(
            mean=round(sum(s.value for s in scores) / n, 3),
            pass_rate=round(sum(1 for s in scores if s.passed) / n, 3),
            n=n,
        )
    return summary
