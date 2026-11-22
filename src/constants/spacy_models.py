"""Trained spaCy small pipelines baked into the image.

Detector codes are resolved through ``SPACY_LANG_ALIASES`` before lookup
(for example langid ``no`` / ``nn`` → Norwegian Bokmål ``nb``).

Languages with no trained spaCy pipeline (Arabic, Hindi, Turkish,
Vietnamese, Thai, Indonesian, Hebrew, and any other unresolved code) use
``spacy.blank(lang)`` at runtime. That is tokenizer-only: retrieval still
produces tokens, but NER entities are empty. ``xx_ent_wiki_sm`` was
considered for coarse multilingual NER and skipped — it adds image size
for Wikipedia PER/LOC/ORG/MISC tags, while blank() is free and still
useful for query variants.
"""

SPACY_MODEL_NAMES = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "ca": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "fi": "fi_core_news_sm",
    "el": "el_core_news_sm",
    "hr": "hr_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "ro": "ro_core_news_sm",
    "sl": "sl_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
}

SPACY_LANG_ALIASES = {
    "no": "nb",
    "nn": "nb",
}

SPACY_BLANK_FALLBACK_LANGS = frozenset(
    {
        "ar",
        "bn",
        "fil",
        "he",
        "hi",
        "id",
        "km",
        "lo",
        "ml",
        "ms",
        "my",
        "pa",
        "ta",
        "te",
        "th",
        "tr",
        "vi",
    }
)

DEFAULT_SPACY_LOAD_RETRY_TTL_SECONDS = 300.0


def resolve_spacy_lang(lang: str | None) -> str:
    """Map a detector ISO code to the spaCy pipeline key."""
    if not lang:
        return "xx"
    code = lang.strip().lower().replace("_", "-")
    if not code:
        return "xx"
    primary = code.split("-", 1)[0]
    return SPACY_LANG_ALIASES.get(code, SPACY_LANG_ALIASES.get(primary, primary))
