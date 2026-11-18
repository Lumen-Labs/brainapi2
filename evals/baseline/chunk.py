from __future__ import annotations

from typing import Any

from evals.baseline.textlib import sentences


def chunk_document(
    doc: dict[str, Any],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    text = str(doc.get("text") or "")
    if not text.strip():
        return []
    size = max(80, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size // 2))
    sents = sentences(text)
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_len = 0
    index = 0

    def flush() -> None:
        nonlocal buf, buf_len, index
        body = " ".join(buf).strip()
        if not body:
            buf = []
            buf_len = 0
            return
        chunk = {
            "id": f"{doc.get('id')}#c{index}",
            "doc_id": doc.get("id"),
            "statement_id": doc.get("statement_id") or doc.get("id"),
            "title": doc.get("title") or "",
            "text": body,
            "valid_from": doc.get("valid_from"),
            "valid_to": doc.get("valid_to"),
            "as_of_dates": list(doc.get("as_of_dates") or []),
            "source_url": doc.get("source_url") or "",
            "version": doc.get("version"),
            "clause_id": doc.get("clause_id"),
            "chunk_index": index,
        }
        chunks.append(chunk)
        index += 1
        if overlap > 0 and body:
            keep = body[-overlap:]
            buf = [keep]
            buf_len = len(keep)
        else:
            buf = []
            buf_len = 0

    for sent in sents:
        extra = len(sent) + (1 if buf else 0)
        if buf and buf_len + extra > size:
            flush()
        buf.append(sent)
        buf_len += extra
    if buf:
        flush()
    if not chunks:
        chunks.append(
            {
                "id": f"{doc.get('id')}#c0",
                "doc_id": doc.get("id"),
                "statement_id": doc.get("statement_id") or doc.get("id"),
                "title": doc.get("title") or "",
                "text": text.strip(),
                "valid_from": doc.get("valid_from"),
                "valid_to": doc.get("valid_to"),
                "as_of_dates": list(doc.get("as_of_dates") or []),
                "source_url": doc.get("source_url") or "",
                "version": doc.get("version"),
                "clause_id": doc.get("clause_id"),
                "chunk_index": 0,
            }
        )
    return chunks


def chunk_corpus(
    docs: list[dict[str, Any]],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return out
