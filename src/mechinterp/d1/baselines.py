"""A TF-IDF text classifier, in numpy. Adapters cannot change it: it reads the transcript up to the follow-up.

Vocabulary, idf and weights come from probe-train families only. Test text is only ever transformed, so nothing about
test families reaches the fit. The follow-up wording is not in the text, so wording cannot leak either.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def transcript_text(messages: Sequence[dict]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def _bigrams(text: str) -> list[str]:
    words = _TOKEN.findall(text.casefold())
    return [*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]


class TfidfClassifier:
    def __init__(self, max_features: int = 50000, epochs: int = 8, learning_rate: float = 0.5, l2: float = 1e-6, seed: int = 0) -> None:
        self.max_features = max_features
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.vocabulary: dict[str, int] = {}
        self.idf = np.zeros(0)
        self.weights = np.zeros(0)
        self.bias = 0.0

    def _vector(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        counts = Counter(self.vocabulary[term] for term in _bigrams(text) if term in self.vocabulary)
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        indices = np.fromiter(counts.keys(), dtype=np.int64)
        values = np.fromiter((1.0 + math.log(count) for count in counts.values()), dtype=np.float64) * self.idf[indices]
        return indices, values / max(np.linalg.norm(values), 1e-12)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfClassifier":
        frequency: Counter = Counter()
        for text in texts:
            frequency.update(set(_bigrams(text)))
        # Term ties break by the term, so the vocabulary does not depend on hash order.
        terms = sorted((term for term, count in frequency.items() if count >= 2), key=lambda term: (-frequency[term], term))[: self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + frequency[term])) + 1.0 for term in terms])
        vectors = [self._vector(text) for text in texts]
        y = np.asarray(labels, dtype=np.float64)
        self.weights = np.zeros(len(terms))
        self.bias = 0.0
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            rate = self.learning_rate / (1.0 + epoch)
            for row in rng.permutation(n):
                indices, values = vectors[row]
                margin = min(max(float(self.weights[indices] @ values) + self.bias, -30.0), 30.0)
                # d(logistic loss)/d(margin): -sigmoid(-margin) for a positive row, sigmoid(margin) for a negative one.
                gradient = -1.0 / (1.0 + math.exp(margin)) if y[row] > 0.5 else 1.0 / (1.0 + math.exp(-margin))
                self.weights[indices] -= rate * (gradient * values + self.l2 * self.weights[indices])
                self.bias -= rate * gradient * 0.1
        return self

    def decision(self, texts: Iterable[str]) -> np.ndarray:
        scores = []
        for text in texts:
            indices, values = self._vector(text)
            scores.append(float(self.weights[indices] @ values) + self.bias)
        return np.array(scores, dtype=np.float64)
