"""Turns files (PDF, Markdown, text) into cleaned text, one entry per page."""
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


@dataclass
class ParsedPage:
    source: str        # file name relative to data/raw
    page: int | None   # 1-based page number; None for files that have no pages
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")          # remove soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)    # join words split across lines: "retri-\neval" -> "retrieval"
    text = re.sub(r"\s+", " ", text)            # collapse all whitespace (including newlines) to single spaces
    return text.strip()


def parse_pdf(path: Path, source: str) -> list[ParsedPage]:
    pages = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(ParsedPage(source=source, page=page_number, text=text))
    return pages


def parse_text_file(path: Path, source: str) -> list[ParsedPage]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [ParsedPage(source=source, page=None, text=text)] if text else []


def parse_file(path: Path, source: str) -> list[ParsedPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, source)
    if suffix in {".md", ".txt"}:
        return parse_text_file(path, source)
    raise ValueError(f"Unsupported file type: {path.suffix}")