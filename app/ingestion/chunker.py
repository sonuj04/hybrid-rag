"""Splits parsed pages into overlapping, fixed-size text chunks.

Chunks never cross page boundaries, so every chunk has an exact page number for citations.
"""
import hashlib
from dataclasses import dataclass

from app import config
from app.ingestion.parser import ParsedPage


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int | None
    chunk_index: int  # position of this chunk within its page
    text: str


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer to end the chunk at a space (searching only the back half of the window)
            last_space = text.rfind(" ", start + chunk_size // 2, end)
            if last_space != -1:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # The next chunk starts `overlap` characters before this one ended
        start = max(end - overlap, start + 1)
        # Move forward to the next word boundary so a chunk never starts mid-word
        if text[start - 1] != " ":
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1
    return chunks


def make_chunk_id(source: str, page: int | None, chunk_index: int) -> str:
    raw = f"{source}|{page}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks = []
    for page in pages:
        for index, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(page.source, page.page, index),
                    source=page.source,
                    page=page.page,
                    chunk_index=index,
                    text=piece,
                )
            )
    return chunks