"""Explainable transcript features for Spanish caller conversations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+|\d+(?:[.,]\d+)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre))\b", re.IGNORECASE)
FILLERS = {"eh", "ehh", "este", "emm", "mmm", "pues", "o sea", "como"}
UNCERTAINTY = ("no sé", "no se", "no recuerdo", "no tengo", "desconozco", "no estoy seguro", "no estoy segura", "podría repetir", "puede repetir", "cuál", "cual")
REFUSAL = ("prefiero no", "no quiero", "no puedo", "no deseo")
DETAIL_MARKERS = ("mi nombre", "mi dirección", "mi direccion", "número", "numero", "fecha", "cuenta", "tarjeta", "domicilio")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _ngram_set(items: list[str], size: int = 2) -> set[tuple[str, ...]]:
    return {tuple(items[index : index + size]) for index in range(max(0, len(items) - size + 1))}


def _ratio(count: float, total: float) -> float:
    return float(count / total) if total else 0.0


@dataclass
class SentenceEmbedder:
    """Multilingual embeddings when installed; lexical cosine fallback otherwise."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enabled: bool = True
    _model: Any | None = None
    backend: str = "lexical"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.enabled and self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self.backend = "sentence_transformers"
            except Exception:
                self.enabled = False
        if self._model is not None:
            return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        vocabulary = sorted({token for text in texts for token in tokens(text)})
        if not vocabulary:
            return np.zeros((len(texts), 1), dtype=float)
        lookup = {token: index for index, token in enumerate(vocabulary)}
        matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
        for row, text in enumerate(texts):
            for token, count in Counter(tokens(text)).items():
                matrix[row, lookup[token]] = count
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def _numeric_conflicts(texts: list[str]) -> int:
    # Repeating one number is not a contradiction; multiple distinct answers are a proxy.
    values = [value.replace(",", ".") for text in texts for value in NUMBER_RE.findall(text)]
    return int(len(set(values)) > 1)


def extract_features(utterances: Iterable[dict[str, Any]], embedder: SentenceEmbedder | None = None) -> dict[str, float]:
    """Produce deterministic call-level semantic features, without acoustic/STT scores."""
    utterance_list = list(utterances)
    callers = [item for item in utterance_list if item.get("channel") == 0 and item.get("text", "").strip()]
    caller_texts = [item["text"] for item in callers]
    caller_tokens = [tokens(text) for text in caller_texts]
    flat_tokens = [token for item in caller_tokens for token in item]
    total = len(flat_tokens)
    joined = " ".join(caller_texts).lower()
    values: dict[str, float] = {
        "caller_turn_count": float(len(callers)),
        "caller_word_count": float(total),
        "caller_words_per_turn": _ratio(total, len(callers)),
        "lexical_diversity": _ratio(len(set(flat_tokens)), total),
        "filler_rate": _ratio(sum(token in FILLERS for token in flat_tokens), total),
        "uncertainty_rate": _ratio(sum(joined.count(marker) for marker in UNCERTAINTY), max(1, len(callers))),
        "refusal_rate": _ratio(sum(joined.count(marker) for marker in REFUSAL), max(1, len(callers))),
        "clarification_rate": _ratio(sum(joined.count(marker) for marker in ("repetir", "entendí", "entendi", "disculpe")), max(1, len(callers))),
        "numeric_detail_rate": _ratio(len(NUMBER_RE.findall(joined)), max(1, len(callers))),
        "date_detail_rate": _ratio(len(DATE_RE.findall(joined)), max(1, len(callers))),
        "self_detail_rate": _ratio(sum(joined.count(marker) for marker in DETAIL_MARKERS), max(1, len(callers))),
        "numeric_conflict": float(_numeric_conflicts(caller_texts)),
        "mean_response_gap_s": float(np.mean([item["response_gap_s"] for item in callers if item.get("response_gap_s") is not None])) if any(item.get("response_gap_s") is not None for item in callers) else 0.0,
    }

    ngrams = [_ngram_set(item) for item in caller_tokens]
    overlap = []
    for first, second in combinations(ngrams, 2):
        overlap.append(_ratio(len(first & second), len(first | second)))
    values["mean_bigram_repetition"] = float(np.mean(overlap)) if overlap else 0.0

    active_embedder = embedder or SentenceEmbedder()
    relevance_pairs = [(item.get("preceding_agent", ""), item["text"]) for item in callers if item.get("preceding_agent", "").strip()]
    relevance = []
    if relevance_pairs:
        vectors = active_embedder.encode([text for pair in relevance_pairs for text in pair])
        relevance = [_cosine(vectors[index * 2], vectors[index * 2 + 1]) for index in range(len(relevance_pairs))]
    values["response_relevance_mean"] = float(np.mean(relevance)) if relevance else 0.0
    values["response_relevance_std"] = float(np.std(relevance)) if relevance else 0.0
    values["low_relevance_rate"] = _ratio(sum(score < 0.15 for score in relevance), len(relevance))

    if len(caller_texts) > 1:
        caller_vectors = active_embedder.encode(caller_texts)
        similarities = [_cosine(caller_vectors[i], caller_vectors[j]) for i, j in combinations(range(len(caller_texts)), 2)]
        values["mean_turn_similarity"] = float(np.mean(similarities)) if similarities else 0.0
    else:
        values["mean_turn_similarity"] = 0.0

    unsupported = 0
    for agent_text, caller_text in relevance_pairs:
        agent_tokens, caller_words = set(tokens(agent_text)), tokens(caller_text)
        specific = [word for word in caller_words if len(word) >= 5 and word not in agent_tokens]
        unsupported += len(specific)
    values["unsupported_detail_rate"] = _ratio(unsupported, total)
    return {key: float(value) for key, value in sorted(values.items())}
