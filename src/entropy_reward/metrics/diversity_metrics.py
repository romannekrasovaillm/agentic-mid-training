"""Diversity metrics: self-BLEU, trajectory uniqueness, pattern repetition."""

from __future__ import annotations

import hashlib
from collections import Counter, deque

import numpy as np


class SelfBLEU:
    """Self-BLEU: measures diversity by computing BLEU of each sample against all others.

    Lower self-BLEU = more diverse. Higher = more repetitive.
    """

    def __init__(self, n: int = 4, sample_size: int = 100):
        self.n = n
        self.sample_size = sample_size
        self._history: list[float] = []

    def compute(self, texts: list[str]) -> float:
        """Compute mean self-BLEU score across a set of texts."""
        if len(texts) < 2:
            return 0.0

        # Subsample if too many
        if len(texts) > self.sample_size:
            indices = np.random.choice(len(texts), self.sample_size, replace=False)
            texts = [texts[i] for i in indices]

        tokenized = [text.split() for text in texts]
        scores = []

        for i, hypothesis in enumerate(tokenized):
            references = [tokenized[j] for j in range(len(tokenized)) if j != i]
            score = self._bleu_score(hypothesis, references)
            scores.append(score)

        val = float(np.mean(scores))
        self._history.append(val)
        return val

    def _bleu_score(self, hypothesis: list[str], references: list[list[str]]) -> float:
        """Simplified BLEU computation."""
        if not hypothesis:
            return 0.0

        precisions = []
        for n in range(1, self.n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            if not hyp_ngrams:
                precisions.append(0.0)
                continue

            ref_ngrams: Counter = Counter()
            for ref in references:
                ref_counts = self._get_ngrams(ref, n)
                for ng, count in ref_counts.items():
                    ref_ngrams[ng] = max(ref_ngrams[ng], count)

            clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            precisions.append(clipped / total if total > 0 else 0.0)

        if any(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(np.log(p + 1e-10) for p in precisions) / len(precisions)
        # Brevity penalty (simplified)
        ref_len = np.mean([len(r) for r in references])
        bp = min(1.0, np.exp(1 - ref_len / max(len(hypothesis), 1)))

        return float(bp * np.exp(log_avg))

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @property
    def history(self) -> list[float]:
        return self._history


class TrajectoryUniqueness:
    """Measure fraction of unique trajectories in a batch.

    Uses hashing for efficient deduplication.
    """

    def __init__(self, window_size: int = 200):
        self._history: list[float] = []
        self._seen_hashes: set[str] = set()

    def compute(self, texts: list[str]) -> float:
        """Compute fraction of unique texts in batch."""
        if not texts:
            return 0.0

        hashes = set()
        for t in texts:
            h = hashlib.md5(t.encode()).hexdigest()
            hashes.add(h)

        uniqueness = len(hashes) / len(texts)
        self._history.append(uniqueness)

        # Track global uniqueness
        new_count = sum(1 for h in hashes if h not in self._seen_hashes)
        self._seen_hashes.update(hashes)

        return uniqueness

    @property
    def global_unique_count(self) -> int:
        return len(self._seen_hashes)

    @property
    def history(self) -> list[float]:
        return self._history


class PatternRepetition:
    """Detect repeated structural patterns in outputs.

    Extracts structural fingerprints (tag sequences) and tracks repetition rate.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._pattern_history: deque[list[str]] = deque(maxlen=window_size)
        self._history: list[float] = []

    def compute(self, texts: list[str]) -> float:
        """Compute fraction of repeated structural patterns in batch."""
        patterns = [self._extract_pattern(t) for t in texts]
        self._pattern_history.append(patterns)

        # Flatten all recent patterns
        all_patterns = []
        for batch_patterns in self._pattern_history:
            all_patterns.extend(batch_patterns)

        if not all_patterns:
            return 0.0

        counter = Counter(all_patterns)
        total = len(all_patterns)
        # Fraction that appear more than once
        repeated = sum(c for c in counter.values() if c > 1)
        repetition_rate = repeated / total

        self._history.append(repetition_rate)
        return repetition_rate

    @staticmethod
    def _extract_pattern(text: str) -> str:
        """Extract structural fingerprint: sequence of tag types."""
        import re

        tags = re.findall(r"</?(\w+)>", text)
        return "|".join(tags) if tags else "no_tags"

    @property
    def history(self) -> list[float]:
        return self._history
