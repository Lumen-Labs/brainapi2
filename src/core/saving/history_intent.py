"""Decide whether a retrieve should include prior revisions.

The decision is similarity between the query embedding and example questions.
It does not match words in a language list.
"""

from __future__ import annotations

from collections.abc import Callable

Embed = Callable[[str], list[float]]

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

_CACHED: tuple[list[list[float]], list[list[float]]] | None = None


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


def prefers_history(
    query: list[float] | None,
    history_vectors: list[list[float]],
    current_vectors: list[list[float]],
    *,
    margin: float = 0.02,
) -> bool:
    """History wins only when it is closer than the current-value examples."""
    if not query or not history_vectors or not current_vectors:
        return False
    history_score = max(cosine(query, vector) for vector in history_vectors)
    current_score = max(cosine(query, vector) for vector in current_vectors)
    return history_score > current_score + margin


def resolve_include_history(
    explicit: bool | None,
    query: list[float] | None,
    history_vectors: list[list[float]] | None,
    current_vectors: list[list[float]] | None,
) -> bool:
    if explicit is not None:
        return explicit
    return prefers_history(query, history_vectors or [], current_vectors or [])


def prototype_vectors(embed: Embed) -> tuple[list[list[float]], list[list[float]]]:
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    history = [embed(text) for text in HISTORY_EXAMPLES]
    current = [embed(text) for text in CURRENT_EXAMPLES]
    _CACHED = (history, current)
    return _CACHED
