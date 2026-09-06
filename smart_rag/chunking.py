"""Document chunking strategies.

Chunk quality drives retrieval quality: chunks that are too large dilute
the embedding signal, too small and they lose context. This module offers
three complementary strategies.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """A unit of retrievable text plus its provenance metadata."""

    text: str
    doc_id: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    position: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.doc_id}#{self.position}: '{preview}...')"


def _split_sentences(text: str) -> List[str]:
    """Naive but dependency-free sentence splitter."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/quote,
    # while keeping the punctuation attached.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_fixed(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 100,
    metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Fixed-size character chunking with sliding overlap.

    Simple and fast; good baseline. Overlap prevents relevant text from
    being severed exactly at a chunk boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    chunks: List[Chunk] = []
    start = 0
    position = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(text=piece, doc_id=doc_id, position=position, metadata=metadata or {})
            )
            position += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_sentences(
    text: str,
    doc_id: str,
    sentences_per_chunk: int = 4,
    overlap_sentences: int = 1,
    metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Group whole sentences into chunks so semantic units are never cut mid-sentence."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if overlap_sentences >= sentences_per_chunk:
        raise ValueError("overlap_sentences must be smaller than sentences_per_chunk")

    chunks: List[Chunk] = []
    step = sentences_per_chunk - overlap_sentences
    position = 0
    for i in range(0, len(sentences), step):
        group = sentences[i : i + sentences_per_chunk]
        if not group:
            continue
        chunks.append(
            Chunk(text=" ".join(group), doc_id=doc_id, position=position, metadata=metadata or {})
        )
        position += 1
        if i + sentences_per_chunk >= len(sentences):
            break
    return chunks


def chunk_recursive(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Recursively split on the largest available separator (paragraph -> line ->
    sentence -> word) until pieces fit within chunk_size, then merges small
    adjacent pieces back together with overlap. This best preserves natural
    document structure (headings, paragraphs) compared to raw fixed-size cuts.
    """
    separators = separators or ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: List[str]) -> List[str]:
        if not seps or len(t) <= chunk_size:
            return [t] if t.strip() else []
        sep, rest = seps[0], seps[1:]
        parts = [p for p in t.split(sep) if p.strip()]
        out: List[str] = []
        for p in parts:
            if len(p) > chunk_size:
                out.extend(split(p, rest))
            else:
                out.append(p)
        return out

    raw_pieces = split(text.strip(), separators)

    # Merge small adjacent pieces up to chunk_size, with a bit of overlap text
    merged: List[str] = []
    buf = ""
    for piece in raw_pieces:
        candidate = (buf + " " + piece).strip() if buf else piece
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)

    chunks: List[Chunk] = []
    prev_tail = ""
    for position, piece in enumerate(merged):
        full = (prev_tail + " " + piece).strip() if prev_tail else piece
        chunks.append(Chunk(text=full, doc_id=doc_id, position=position, metadata=metadata or {}))
        prev_tail = piece[-overlap:] if overlap > 0 else ""
    return chunks
