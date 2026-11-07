def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) <= max_chars:
        return [cleaned] if cleaned else [""]

    # Prefer blank-line paragraphs, then single newlines (dialogue turns),
    # else hard character slices.
    if "\n\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        joiner = "\n\n"
    elif "\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n") if p.strip()]
        joiner = "\n"
    else:
        parts = [cleaned]
        joiner = "\n"

    if not parts:
        parts = [cleaned]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(part) <= max_chars:
            current = part
            continue
        for i in range(0, len(part), max_chars):
            piece = part[i : i + max_chars].strip()
            if piece:
                chunks.append(piece)
        current = ""
    if current:
        chunks.append(current)
    return chunks or [cleaned]


def source_cut(text: str, limit: int) -> int:
    """Choose a contiguous source boundary at or before the character limit."""
    import re

    if len(text) <= limit:
        return len(text)
    prefix = text[:limit]
    floor = limit // 2
    for pattern in (r"\n(?=[ \t]*#{1,6}[ \t])", r"\n[ \t]*\n", r"\n", r"[.!?。！？](?:[ \t]+|$)", r"\s+"):
        boundary = max((match.end() for match in re.finditer(pattern, prefix)), default=0)
        if boundary > floor:
            return boundary
    return limit


def source_context(text: str, start: int, *, include_excerpt: bool = True) -> str:
    """Attach headings and a short preceding excerpt as context only."""
    import re

    if not start:
        return ""
    headings = []
    for match in re.finditer(r"(?m)^\s*(#{1,6})[ \t]+([^\n]+)", text[:start]):
        level = len(match[1])
        headings = [(depth, title) for depth, title in headings if depth < level]
        headings.append((level, match[2][:120]))
    heading_path = " > ".join(title for _, title in headings)
    excerpt = text[max(0, start - 480):start].strip() if include_excerpt else ""
    if not heading_path and not excerpt:
        return ""
    return (
        f"[Prior context for reference only. Extract from the current span.]\n"
        f"{heading_path}\n{excerpt}\n\n[Current span]\n"
    )
