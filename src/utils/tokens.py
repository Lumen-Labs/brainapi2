import hmac
import secrets
from typing import Optional

PAT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
PAT_LENGTH = 48


def generate_pat() -> str:
    return "".join(secrets.choice(PAT_ALPHABET) for _ in range(PAT_LENGTH))


def pat_matches(provided: Optional[str], expected: Optional[str]) -> bool:
    if provided is None or expected is None:
        return False
    if not isinstance(provided, str) or not isinstance(expected, str):
        return False
    if not provided or not expected:
        return False
    left = provided.encode("utf-8")
    right = expected.encode("utf-8")
    if len(left) != len(right):
        hmac.compare_digest(left, left)
        return False
    return hmac.compare_digest(left, right)
