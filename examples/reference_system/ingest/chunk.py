"""Ingestion: documents in, chunks with stable ids out.

The stable id is the point. Stratum scores retrieval by comparing returned
chunk ids against gold ones, so an id that shifts when the corpus is rebuilt
silently invalidates every dataset item that references it.

Ids are therefore derived from content, not position: same text, same id,
regardless of what else changed in the document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    index: int = 0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "text": self.text,
            "page": self.page, "index": self.index, "meta": self.meta,
        }


def stable_chunk_id(doc_id: str, text: str) -> str:
    """Content-addressed id: `doc:hash`.

    Position is excluded deliberately. Inserting a paragraph at the top of a
    document would otherwise renumber every chunk below it and break every
    gold reference in the eval dataset.
    """
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:{digest}"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_text(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number, text)]. PDFs page-wise, text files as one block."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PDF ingestion needs PyMuPDF — pip install pymupdf"
            ) from exc
        doc = fitz.open(path)
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]

    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8"))]

    raise ValueError(f"unsupported document type: {path.suffix}")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_PARA_BREAK = re.compile(r"\n\s*\n")
_SENT_END = re.compile(r"(?<=[.!?।])\s+")

#: Rough token estimate. Deliberately crude: an exact tokenizer would tie the
#: chunker to one model's vocabulary, and chunk boundaries are not that
#: sensitive. Indic scripts run longer per character, hence the low divisor.
def approx_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def split_sentences(text: str) -> list[str]:
    """Sentence split that also honours the Devanagari danda."""
    parts = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[str]:
    """Recursive split: paragraphs first, sentences only where needed.

    Splitting on paragraphs first keeps clauses intact, which matters for
    policy documents where a condition and its exceptions belong together.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n\n".join(buffer).strip())
            buffer, buffer_tokens = [], 0

    for para in _PARA_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        tokens = approx_tokens(para)

        if tokens > target_tokens:
            flush()
            # Paragraph alone exceeds the target: fall back to sentences.
            sent_buf: list[str] = []
            sent_tokens = 0
            for sent in split_sentences(para):
                st = approx_tokens(sent)
                if sent_tokens + st > target_tokens and sent_buf:
                    chunks.append(" ".join(sent_buf).strip())
                    # Carry the tail forward so a split clause stays reachable.
                    carry, carried = [], 0
                    for s in reversed(sent_buf):
                        carried += approx_tokens(s)
                        carry.insert(0, s)
                        if carried >= overlap_tokens:
                            break
                    sent_buf, sent_tokens = carry, carried
                sent_buf.append(sent)
                sent_tokens += st
            if sent_buf:
                chunks.append(" ".join(sent_buf).strip())
            continue

        if buffer_tokens + tokens > target_tokens:
            flush()
        buffer.append(para)
        buffer_tokens += tokens

    flush()
    return [c for c in chunks if c]


def build_chunks(
    path: Path, *, doc_id: str | None = None, target_tokens: int = 512
) -> list[Chunk]:
    doc_id = doc_id or path.stem
    out: list[Chunk] = []
    for page, page_text in extract_text(path):
        for piece in chunk_text(page_text, target_tokens=target_tokens):
            out.append(
                Chunk(
                    chunk_id=stable_chunk_id(doc_id, piece),
                    doc_id=doc_id,
                    text=piece,
                    page=page,
                    index=len(out),
                )
            )
    return out


#: Files that describe the corpus rather than belonging to it. Ingesting a
#: README makes it retrievable, and it will happily answer questions about
#: the corpus instead of about the policies.
SKIP_STEMS = {"readme", "index", "notes", "license"}


def build_corpus(corpus_dir: Path, *, target_tokens: int = 512) -> list[Chunk]:
    """Ingest every supported document in a directory."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        if path.stem.lower() in SKIP_STEMS:
            continue
        for chunk in build_chunks(path, target_tokens=target_tokens):
            # Content addressing means duplicate boilerplate across documents
            # collapses to one chunk. That is correct: retrieving it once is
            # the right behaviour, and it keeps gold references unambiguous.
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks
