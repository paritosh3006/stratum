# Running this in VS Code

## 1. Open and install

```bash
cd stratum-eval
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → ./.venv/bin/python**

Extensions worth having: Python, Pylance, Ruff.

## 2. Prove it works

```bash
pytest -q
```
Expect 24 passing.

```bash
stratum run \
  --endpoint examples/mock_endpoint.py:endpoint \
  --dataset datasets/demo.jsonl \
  --glossary datasets/glossary.json \
  --out reports/run-001
```

Exit code is **1** — deliberately. The mock system in `examples/mock_endpoint.py`
has five planted bugs and the gates catch three of them. A green run here would
mean the harness is broken.

`F5` runs the same thing under the debugger.

## 3. Point it at a real system

Two options.

**HTTP** — if your RAG has an endpoint:

```python
from stratum import HttpEndpoint
endpoint = HttpEndpoint("http://localhost:8000/query", answer_field="response")
```

**Callable** — if it's a Python function:

```python
from stratum import CallableEndpoint, RagResponse

def my_rag(query: str, language: str) -> RagResponse:
    result = my_existing_pipeline(query)
    return RagResponse(
        answer=result.text,
        retrieved_chunk_ids=[c.id for c in result.chunks],
        detected_language=result.lang,
        refused=result.refused,
    )

endpoint = CallableEndpoint(my_rag)
```

The only real requirement is that your system can return **chunk IDs**. Without
them retrieval metrics are impossible, and retrieval is half the diagnosis. If
your pipeline doesn't expose IDs today, that's the first thing to add.

---

## What is actually built

| Piece | State |
|---|---|
| Dataset loading, validation, parallel-group checks | done |
| Endpoint adapters (callable, HTTP) | done |
| S2 retrieval — recall@k, MRR, nDCG, Jaccard | done |
| S4 rendering — placeholders, glossary, numerals, entities | done |
| S0 language/script detection scoring | done |
| S3 over-refusal detection | done |
| Terminal report, JSON report, gates, run diffing | done |
| Test suite | 24 tests |

## What is not built yet — and the honest caveats

**1. `answer_quality` is a placeholder, not a real score.**
Right now it's the unweighted mean of whichever mechanical metrics are
available. That means a language with fewer applicable checks gets a score
computed from a different basis than another — not strictly comparable. It is
useful as a smoke signal and nothing more. Once judged faithfulness and
correctness land, they become the headline and the mechanical checks stay as
gates. Don't put the current number in a README chart.

**2. Attribution — the cascade — is not implemented.**
This is the headline feature and the hardest part. It needs the three
counterfactual passes (`oracle_query`, `oracle_context`, `oracle_answer`).
Everything needed to build it is in place: `parallel_id` links questions across
languages, and the dataset validator already warns when a baseline counterpart
is missing.

**3. No LLM judge, so no faithfulness or answer correctness.**
Deliberate. Judged metrics need calibration to be worth anything, and
calibration needs hand-labelled data. Mechanical metrics first.

**4. No calibration, no confidence intervals.**
Both are listed in the report format spec and neither exists in code yet. Until
CIs are in, n=16 numbers are illustrative only.

**5. Over-refusal is detected but not in the composite.**
Visible in the failure taxonomy, absent from `answer_quality`. That's why
`hi-Deva` scores level with English in the demo despite having a planted
over-refusal bug — a real hole, not a rounding artefact.

**6. No HTML report.** JSON and terminal only.

---

## Suggested order from here

1. **Confidence intervals** — bootstrap over items. Cheap, and it stops every
   later number from being overclaimed.
2. **The `oracle_query` pass** — smallest of the three counterfactuals and it
   alone separates "translation broke it" from "retrieval broke it", which is
   the single most useful split.
3. **Grow the dataset** to ~150 items per language following the slice mix in
   `docs/dataset-design.md`. Resist machine-translating the English set.
4. **LLM judge + calibration together.** Never ship one without the other.
5. **HTML report**, once there's something worth looking at.

Two languages plus English, done properly, beats five done blind.
