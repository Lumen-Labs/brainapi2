"""
File: /ner.py
Project: nlp
Created Date: Sunday February 8th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 8th 2026 11:21:15 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from pydantic import BaseModel
from src.utils.nlp.spacy import MODEL_NAMES, _spacy_manager
from src.utils.nlp.lang_detect import langid_detect

_DEFAULT_IGNORE_POS = frozenset(
    {
        "PUNCT",
        "SPACE",
        "DET",
        "PRON",
        "AUX",
        "CCONJ",
        "ADP",
        "SCONJ",
        "PART",
        "NUM",
        "SYM",
        "X",
    }
)


class ExtractElementsResponse(BaseModel):
    tokens: list[dict[str, str]]
    noun_chunks: list[str]


class MultiLangEntityExtractor:
    def __init__(self):
        self.spacy_manager = _spacy_manager

    def extract(self, text):
        lang, _ = langid_detect(text)
        if lang not in MODEL_NAMES:
            return []
        try:
            nlp = self.spacy_manager.get_model(lang)
        except Exception:
            return []
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def _process_doc(self, doc, ignore_pos_set):
        tokens = []
        i = 0
        n = len(doc)
        while i < n:
            t = doc[i]
            if t.is_space:
                i += 1
                continue
            if t.ent_iob_ == "B":
                span = [t]
                j = 1
                while i + j < n and doc[i + j].ent_iob_ == "I":
                    span.append(doc[i + j])
                    j += 1
                tokens.append(
                    {
                        "text": " ".join(x.text for x in span),
                        "lemma": " ".join(x.lemma_ for x in span),
                        "pos": span[0].pos_,
                        "entity": t.ent_type_,
                    }
                )
                i += len(span)
                continue
            if t.ent_iob_ == "I":
                i += 1
                continue
            if t.pos_ not in ignore_pos_set:
                tokens.append({"text": t.text, "lemma": t.lemma_, "pos": t.pos_})
            i += 1
        entity_spans = [(ent.start, ent.end) for ent in doc.ents]
        noun_chunks = []
        for chunk in doc.noun_chunks:
            for start, end in entity_spans:
                if start <= chunk.start and chunk.end <= end:
                    break
            else:
                noun_chunks.append(chunk.text.strip())
        for ent in doc.ents:
            noun_chunks.append(ent.text.strip())
        return {"tokens": tokens, "noun_chunks": noun_chunks}

    def extract_elements(
        self,
        text,
        lang: str | None = None,
        ignore_pos: list[str] | None = None,
    ) -> ExtractElementsResponse:
        if lang is None:
            lang, _ = langid_detect(text)
        if lang not in MODEL_NAMES:
            return {"tokens": [], "noun_chunks": []}
        try:
            nlp = self.spacy_manager.get_model(lang)
        except Exception:
            return {"tokens": [], "noun_chunks": []}
        doc = nlp(text)
        ignore_pos_set = (
            set(ignore_pos) if ignore_pos is not None else _DEFAULT_IGNORE_POS
        )
        return ExtractElementsResponse(
            tokens=self._process_doc(doc, ignore_pos_set)["tokens"],
            noun_chunks=self._process_doc(doc, ignore_pos_set)["noun_chunks"],
        )

    def extract_elements_batch(
        self,
        texts: list[str],
        lang: str | None = None,
        ignore_pos: list[str] | None = None,
        batch_size: int = 50,
    ):
        if not texts:
            return []
        if lang is None:
            lang, _ = langid_detect(texts[0])
        if lang not in MODEL_NAMES:
            return [{"tokens": [], "noun_chunks": []} for _ in texts]
        try:
            nlp = self.spacy_manager.get_model(lang)
        except Exception:
            return [{"tokens": [], "noun_chunks": []} for _ in texts]
        ignore_pos_set = (
            set(ignore_pos) if ignore_pos is not None else _DEFAULT_IGNORE_POS
        )
        out = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            out.append(self._process_doc(doc, ignore_pos_set))
        return out


_entity_extractor = MultiLangEntityExtractor()
