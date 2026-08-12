"""Judge backends: something that can actually call `JudgeBackend.judge_*`.

Same split as everywhere else model-dependent in this codebase: a real
backend that needs something running, and a deterministic no-download stub
so `pytest` doesn't need Ollama installed to exercise the judge-gated code
paths. `StubJudge` is not a quality baseline — like `HashingEmbedder`, it
exists so the plumbing has an offline leg to stand on.
"""

from __future__ import annotations

import json
import re
import urllib.request

from .base import JudgeBackend, Judgement

#: Deliberately crude and self-contained: stratum core has no corpus, no
#: retriever, and no reason to depend on any example system's tokenizer.
#: Devanagari/Tamil block ranges are included for the same reason
#: examples/reference_system/retrieval/hybrid.py's tokenizer needs them —
#: `\w` alone excludes Indic combining vowel signs and shatters every word.
_TOKEN = re.compile(r"[\wऀ-෿]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


class StubJudge:
    """Token-overlap scoring. No model, no network, fully deterministic.

    Faithfulness is coverage (how much of the answer's vocabulary appears
    somewhere in the retrieved context) rather than F1: an answer is not
    unfaithful for omitting context content, only for asserting content the
    context doesn't support. Correctness is F1 against the reference,
    bucketed onto the 0-3 rubric — precision matters there too, since a
    correct answer padded with unrelated claims shouldn't score as fully
    correct.

    Neither measures actual semantic agreement, only vocabulary overlap —
    it will misjudge a correct paraphrase as wrong and a fluent-sounding
    hallucination as faithful. That is a real, known gap versus the model
    backend, not a hidden one.

    It is also monolingual, and this is the gap that actually bites: it has
    no model and no way to know "प्रीमियम" and "premium" mean the same
    thing. Once a rendering stage translates an answer into the query's
    language, comparing it against an English `gold_answer`/context by raw
    token overlap will read as near-total disagreement regardless of
    whether the answer is right — the same category of limitation
    `HashingEmbedder` has for cross-lingual retrieval, for the same reason:
    a no-model stub cannot bridge languages, only a real model can. Use
    this to verify the judge/calibration *wiring* works end to end
    offline, not to read faithfulness/correctness numbers for non-English
    languages as a quality claim — `OllamaJudge` is the one to calibrate
    and cite for that.
    """

    judge_id = "stub-token-overlap"

    def judge_faithfulness(
        self, answer: str, context: list[str], language: str
    ) -> Judgement:
        answer_tokens = _tokens(answer)
        context_tokens = _tokens(" ".join(context))
        if not answer_tokens or not context_tokens:
            return Judgement(score=0.0, reasoning="no answer or no context to check")
        covered = len(answer_tokens & context_tokens) / len(answer_tokens)
        return Judgement(
            score=round(covered, 4),
            reasoning=f"{len(answer_tokens & context_tokens)}/{len(answer_tokens)} "
                      f"answer terms found in context",
        )

    def judge_correctness(
        self, answer: str, reference: str, language: str
    ) -> Judgement:
        f1 = _f1(_tokens(answer), _tokens(reference))
        # Thresholds, not a linear scale: a rubric step should mean a real
        # jump in agreement, not an arbitrary slice of a continuous score.
        if f1 >= 0.8:
            score = 3.0
        elif f1 >= 0.5:
            score = 2.0
        elif f1 > 0.15:
            score = 1.0
        else:
            score = 0.0
        return Judgement(score=score, reasoning=f"token F1={f1:.2f} vs reference")


class OllamaJudge:
    """A local Ollama model as judge, via its HTTP API.

    Needs `ollama serve` running and the model pulled
    (`ollama pull qwen2.5:7b`). Deterministic-ish, not deterministic:
    `temperature=0.0` is requested, but local models still vary slightly
    across quantisations and Ollama versions — that variance is exactly
    what `stratum calibrate` measures, so it isn't papered over here.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.judge_id = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except OSError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {self.host} — is `ollama serve` running, "
                f"and has `ollama pull {self.model}` been run?"
            ) from exc
        return data.get("response", "")

    @staticmethod
    def _parse(raw: str) -> tuple[float, str]:
        """Extract {"score": ..., "reasoning": ...} from a model response.

        `format: "json"` gets Ollama to constrain output to valid JSON for
        models that support it, but "valid JSON" and "the JSON we asked
        for" are different guarantees — a model can return `{"answer": 2}`
        just as easily as `{"score": 2}`. Parsing is intentionally
        permissive about surrounding text (some models wrap JSON in a
        sentence anyway) but strict about the score key existing: a judge
        that can't be parsed must fail loudly, not silently default to a
        score that looks like a real measurement.
        """
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError(f"judge did not return parseable JSON: {raw!r}") from None
            obj = json.loads(m.group(0))
        if "score" not in obj:
            raise ValueError(f"judge JSON has no 'score' key: {obj!r}")
        return float(obj["score"]), str(obj.get("reasoning", ""))

    # ------------------------------------------------------------------
    def judge_faithfulness(
        self, answer: str, context: list[str], language: str
    ) -> Judgement:
        prompt = _FAITHFULNESS_PROMPT.format(
            context="\n\n".join(context), answer=answer, language=language
        )
        raw = self._generate(prompt)
        score, reasoning = self._parse(raw)
        return Judgement(score=score, reasoning=reasoning, raw=raw)

    def judge_correctness(
        self, answer: str, reference: str, language: str
    ) -> Judgement:
        prompt = _CORRECTNESS_PROMPT.format(
            reference=reference, answer=answer, language=language
        )
        raw = self._generate(prompt)
        score, reasoning = self._parse(raw)
        return Judgement(score=score, reasoning=reasoning, raw=raw)


_FAITHFULNESS_PROMPT = """\
You are checking whether an answer is faithful to its source context — \
whether every factual claim in the answer is actually supported by the \
context, not whether the answer is complete or well-written.

The answer is in {language}. The context is the only source of truth: if \
a claim in the answer is not in the context, it is unsupported, even if it \
is true in general.

Context:
{context}

Answer:
{answer}

Break the answer into its atomic factual claims. For each claim, decide if \
the context supports it. Respond with ONLY a JSON object, no other text:
{{"score": <fraction 0.0-1.0 of claims supported>, "reasoning": "<one sentence>"}}
"""

_CORRECTNESS_PROMPT = """\
You are grading an answer against a reference answer, in {language}. Score \
how well the answer agrees with the reference on a 0-3 scale:

0 = wrong or contradicts the reference
1 = partially correct, missing or wrong on significant points
2 = mostly correct, minor omissions or imprecision only
3 = fully correct, equivalent to the reference

Reference answer:
{reference}

Answer to grade:
{answer}

Respond with ONLY a JSON object, no other text:
{{"score": <0, 1, 2, or 3>, "reasoning": "<one sentence>"}}
"""


def get_judge(name: str = "stub") -> JudgeBackend:
    if name in {"ollama", "qwen2.5:7b"}:
        return OllamaJudge()
    if name.startswith("ollama:"):
        return OllamaJudge(model=name.partition(":")[2])
    if name == "stub":
        return StubJudge()
    raise ValueError(f"unknown judge: {name}")
