import os

PLAIN_TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".markdown",
    ".md",
    ".ndjson",
    ".tsv",
    ".txt",
}

PLAIN_TEXT_CONTENT_TYPES = {
    "application/csv",
    "application/json",
    "application/jsonl",
    "application/x-ndjson",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
    "text/x-markdown",
}

MARKUP_TEXT_CONTENT_TYPES = {
    "application/xhtml+xml",
    "application/xml",
    "text/css",
    "text/html",
    "text/rtf",
    "text/xml",
}

SPREADSHEET_EXTENSIONS = {".csv", ".xls", ".xlsx"}
SPREADSHEET_CONTENT_TYPES = {
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
}


def normalized_content_type(content_type):
    if not content_type:
        return ""
    return str(content_type).split(";", 1)[0].strip().lower()


def filename_extension(filename):
    name = str(filename or "").rsplit("/", 1)[-1].strip().lower()
    if "." not in name or name.startswith("."):
        return ""
    return "." + name.rsplit(".", 1)[-1]


def spreadsheet_experiment_brains():
    raw = os.getenv("SPREADSHEET_EXPERIMENT_BRAINS") or ""
    return {item.strip() for item in raw.split(",") if item.strip()}


def is_spreadsheet_upload(filename, content_type):
    ext = filename_extension(filename)
    if ext in {".tsv", ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson"}:
        return False
    if ext in SPREADSHEET_EXTENSIONS:
        return True
    if ext:
        return False
    return normalized_content_type(content_type) in SPREADSHEET_CONTENT_TYPES


def spreadsheet_experiment_owns_upload(brain_id, filename, content_type):
    brains = spreadsheet_experiment_brains()
    if not brain_id or brain_id not in brains:
        return False
    return is_spreadsheet_upload(filename, content_type)


def is_plain_text_upload(filename, content_type):
    ext = filename_extension(filename)
    ctype = normalized_content_type(content_type)
    if ext in PLAIN_TEXT_EXTENSIONS:
        return True
    if ctype in MARKUP_TEXT_CONTENT_TYPES:
        return False
    if ctype in PLAIN_TEXT_CONTENT_TYPES:
        return True
    return ctype.startswith("text/")


def looks_like_binary(content):
    sample = content[:8192] if content else b""
    return b"\x00" in sample


def decode_plain_text(content):
    return (content or b"").decode("utf-8-sig", errors="replace")


def uses_plain_text_ingest(brain_id, filename, content_type, content=None):
    if not is_plain_text_upload(filename, content_type):
        return False
    if content is not None and looks_like_binary(content):
        return False
    return True
