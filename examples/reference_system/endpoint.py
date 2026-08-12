"""Entry point for `stratum run --endpoint examples/reference_system/endpoint.py:endpoint`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference_system.pipeline.system import SystemConfig, build_endpoint  # noqa: E402
from reference_system.render.glossary import build_glossary  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"

endpoint = build_endpoint(
    SystemConfig(corpus_dir=CORPUS, embedder="hashing", min_span_score=0.20)
)

#: `stratum run` picks this up automatically (see cli.py's `_load_endpoint`)
#: so glossary_adherence is gated without a separate `--glossary` flag to
#: remember and keep in sync — it's the same eval/glossary.json the
#: renderer enforces against, loaded once in render/glossary.py.
glossary = build_glossary()
