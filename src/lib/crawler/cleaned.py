from typing import Any, Optional


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_cleaned_page(cleaned: dict) -> str:
    content = _as_dict(cleaned.get("content"))
    facts = _as_dict(cleaned.get("facts"))
    contacts = _as_dict(facts.get("contacts"))
    lines: list[str] = []

    title = _text(content.get("title"))
    if title:
        lines.append(f"Title: {title}")

    page_type = _text(cleaned.get("page_type"))
    if page_type:
        lines.append(f"Type: {page_type}")

    url = _text(cleaned.get("canonical_url")) or _text(cleaned.get("url"))
    if url:
        lines.append(f"URL: {url}")

    heading_lines: list[str] = []
    for heading in _as_list(content.get("headings")):
        if isinstance(heading, dict):
            text = _text(heading.get("text"))
            if not text:
                continue
            level = heading.get("level") or 1
            heading_lines.append(f"- H{level} {text}")
        else:
            text = _text(heading)
            if text:
                heading_lines.append(f"- {text}")
    if heading_lines:
        lines.extend(["", "Headings:", *heading_lines])

    core = _text(content.get("core"))
    if core:
        lines.extend(["", core])

    contact_lines: list[str] = []
    for email in _as_list(contacts.get("emails")):
        text = _text(email)
        if text:
            contact_lines.append(f"- email: {text}")
    for phone in _as_list(contacts.get("phones")):
        text = _text(phone)
        if text:
            contact_lines.append(f"- phone: {text}")
    for address in _as_list(contacts.get("addresses")):
        text = _text(address)
        if text:
            contact_lines.append(f"- address: {text}")
    if contact_lines:
        lines.extend(["", "Contacts:", *contact_lines])

    cta_lines: list[str] = []
    for cta in _as_list(facts.get("ctas")):
        text = _text(cta)
        if text:
            cta_lines.append(f"- {text}")
    if cta_lines:
        lines.extend(["", "CTAs:", *cta_lines])

    return "\n".join(lines).strip()


def cleaned_title(cleaned: dict, fallback: Optional[str] = None) -> Optional[str]:
    content = _as_dict(cleaned.get("content"))
    return _text(content.get("title")) or _text(fallback) or None
