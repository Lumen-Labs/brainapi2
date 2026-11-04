import hmac
import secrets
from typing import Optional

PAT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
PAT_LENGTH = 48


def generate_pat() -> str:
    return "".join(secrets.choice(PAT_ALPHABET) for _ in range(PAT_LENGTH))


def pat_matches(provided: Optional[str], expected: Optional[str]) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
