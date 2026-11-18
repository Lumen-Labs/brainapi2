"""Decide whether a retrieve should include prior revisions.

The caller flag wins. Otherwise a lexical history cue or embedding
similarity to example questions can select history.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

Embed = Callable[[str], list[float]]

HISTORY_MARGIN = 0.02
_FAIL_BACKOFF = 30.0
_FAIL_BACKOFF_MAX = 300.0

HISTORY_EXAMPLES: tuple[str, ...] = (
    "what did this say before the change",
    "show the previous version of this",
    "what was deleted from this",
    "com'era prima questo testo",
    "qual era la versione precedente",
    "wie lautete der frühere Text",
    "was stand hier vorher",
    "cuál era el valor anterior",
    "qué decía antes de borrarse",
    "quelle était la version précédente",
    "qu'est-ce qui a été supprimé",
    "como era antes esta informação",
    "以前の内容は何でしたか",
    "削除される前の文章は",
    "之前的内容是什么",
    "被删掉之前写了什么",
    "ما الذي كان مكتوباً من قبل",
    "что здесь было написано раньше",
    "предыдущая версия этого текста",
    "이전 내용은 무엇이었나요",
    "wat stond hier eerder",
    "jaka była poprzednia wersja",
    "hur såg den tidigare versionen ut",
    "पहले यहाँ क्या लिखा था",
    "can you list the order in which I brought up these topics",
    "how did my discussions progress in order",
    "walk me through the chronological sequence of events",
    "have I implemented this library before",
    "how many days are there between these two deadlines",
    "these two statements contradict each other",
    "did I say I finished this or that I never started it",
    "in che ordine ho menzionato questi argomenti",
    "ho già implementato questo servizio",
    "in welcher Reihenfolge habe ich das erwähnt",
    "habe ich das schon integriert",
    "en qué orden mencioné estos temas",
    "he integrado esto antes",
    "dans quel ordre ai-je mentionné cela",
    "ai-je déjà implémenté cela",
)

CURRENT_EXAMPLES: tuple[str, ...] = (
    "what does this say now",
    "what is the current value",
    "qual è il valore attuale",
    "che cosa c'è scritto adesso",
    "was steht hier jetzt",
    "wie lautet der aktuelle Text",
    "cuál es el valor actual",
    "qué dice ahora",
    "que dit le texte actuel",
    "quelle est la valeur actuelle",
    "qual é o valor atual",
    "今の内容は何ですか",
    "现在的内容是什么",
    "ما المكتوب الآن",
    "что здесь написано сейчас",
    "현재 내용은 무엇인가요",
    "wat staat hier nu",
    "jaka jest aktualna wersja",
    "vad står det nu",
    "अभी यहाँ क्या लिखा है",
)

_HISTORY_LEXICAL = re.compile(
    r"(?i)("
    r"\b(?:list|give|reconstruct|recall|walk).{0,80}\b(?:order|sequence|chronolog)"
    r"|\bthe order in which\b"
    r"|\bin (?:which|what) order\b"
    r"|\bprogress in order\b"
    r"|\bordered (?:timeline|list|sequence)\b"
    r"|\bevent ordering\b"
    r"|\bchronolog"
    r"|\bsequence of\b"
    r"|\bcontradict"
    r"|\bconflicting (?:statements|information|claims)\b"
    r"|\binconsisten"
    r"|\bhave i\b.{0,80}\b(?:already|before|previously|earlier|originally)"
    r"|\bdid i\b.{0,80}\b(?:already|before|previously|earlier|originally|say|claim|mention)"
    r"|\bpreviously\b"
    r"|\boriginally\b"
    r"|\bbefore (?:the |that )?(?:update|change|extension|deletion|it was)"
    r"|\b(?:was|were) (?:deleted|removed|changed)\b"
    r"|\b(?:deleted|removed) from\b"
    r"|\bfirst (?:said|stated|mentioned)\b"
    r"|\bhow many (?:days|weeks|months).{0,80}\bbetween\b"
    r"|\bprevious version\b"
    r"|\bversione precedente\b"
    r"|\bversi[oó]n anterior\b"
    r"|\bversion précédente\b"
    r"|\bcom'era prima\b"
    r"|\bho già\b"
    r"|\bhabe ich\b.{0,80}\b(?:schon|bereits|vorher|früher)"
    r"|\bhe (?:integrado|implementado|completado)\b.{0,40}\b(?:antes|ya)"
    r"|\bai-je\b.{0,40}\bd[eé]j[aà]"
    r"|\bin welcher reihenfolge\b"
    r"|\bin che ordine\b"
    r"|\ben qu[eé] orden\b"
    r"|\bdans quel ordre\b"
    r")"
)
_CURRENT_VALUE_RE = re.compile(
    r"(?i)\b("
    r"now|currently|latest|updated (?:to|value|deadline)|after the (?:update|change)|"
    r"how many commits|average response time|response time of|"
    r"the current|current (?:value|deadline|version|port|setting|text)|"
    r"currently configured|deadline for completing|"
    r"what did i set|difference between|"
    r"how many (?:days|weeks|months) until|when does|"
    r"have i|did i"
    r")\b"
)
_STRONG_HISTORY_RE = re.compile(
    r"(?i)("
    r"\bthe order in which\b"
    r"|\bin (?:which|what) order\b"
    r"|\bprogress in order\b"
    r"|\bevent ordering\b"
    r"|\bchronolog"
    r"|\bsequence of\b"
    r"|\bcontradict"
    r"|\bhave i\b.{0,80}\b(?:already|before|previously|earlier|originally)"
    r"|\bdid i\b.{0,80}\b(?:already|before|previously|earlier|originally|say|claim|mention)"
    r"|\bhow many (?:days|weeks|months).{0,80}\bbetween\b"
    r"|\bprevious version\b"
    r"|\bversione precedente\b"
    r"|\bcom'era prima\b"
    r"|\bho già\b"
    r"|\bhabe ich\b.{0,80}\b(?:schon|bereits|vorher|früher)"
    r"|\bai-je\b.{0,40}\bd[eé]j[aà]"
    r")"
)

_CACHED: tuple[list[list[float]], list[list[float]]] | None = None
_FAIL_UNTIL = 0.0
_FAIL_STREAK = 0


class PrototypeWarmupError(RuntimeError):
    """Raised when prototype embeddings are unavailable and still backing off."""


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def lexical_history_intent(text: str | None) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _CURRENT_VALUE_RE.search(cleaned) and not _STRONG_HISTORY_RE.search(cleaned):
        return False
    return bool(_HISTORY_LEXICAL.search(cleaned))


def prefers_history(
    query: list[float] | None,
    history_vectors: list[list[float]],
    current_vectors: list[list[float]],
    *,
    margin: float = HISTORY_MARGIN,
) -> bool:
    """History wins only when it is closer than the current-value examples."""
    if not query or not history_vectors or not current_vectors:
        return False
    history_score = max(cosine(query, vector) for vector in history_vectors)
    current_score = max(cosine(query, vector) for vector in current_vectors)
    return history_score > current_score + margin


def resolve_include_history(
    explicit: bool | None,
    query_text: str | None = None,
    query: list[float] | None = None,
    history_vectors: list[list[float]] | None = None,
    current_vectors: list[list[float]] | None = None,
    *,
    margin: float = HISTORY_MARGIN,
) -> bool:
    if explicit is not None:
        return explicit
    if lexical_history_intent(query_text):
        return True
    cleaned = (query_text or "").strip()
    if cleaned and _CURRENT_VALUE_RE.search(cleaned) and not _STRONG_HISTORY_RE.search(
        cleaned
    ):
        return False
    return prefers_history(
        query,
        history_vectors or [],
        current_vectors or [],
        margin=margin,
    )


def reset_prototype_cache() -> None:
    global _CACHED, _FAIL_UNTIL, _FAIL_STREAK
    _CACHED = None
    _FAIL_UNTIL = 0.0
    _FAIL_STREAK = 0


def prototype_vectors(embed: Embed) -> tuple[list[list[float]], list[list[float]]]:
    global _CACHED, _FAIL_UNTIL, _FAIL_STREAK
    if _CACHED is not None:
        return _CACHED
    now = time.monotonic()
    if now < _FAIL_UNTIL:
        raise PrototypeWarmupError("history prototype warmup backing off")
    try:
        history = [embed(text) for text in HISTORY_EXAMPLES]
        current = [embed(text) for text in CURRENT_EXAMPLES]
        if any(not vector for vector in history + current):
            raise PrototypeWarmupError("empty prototype embedding")
        _CACHED = (history, current)
        _FAIL_STREAK = 0
        return _CACHED
    except Exception:
        _FAIL_STREAK += 1
        delay = min(_FAIL_BACKOFF * (2 ** (_FAIL_STREAK - 1)), _FAIL_BACKOFF_MAX)
        _FAIL_UNTIL = now + delay
        raise
