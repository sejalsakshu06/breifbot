"""
NLP Analyzer — Sentiment, keywords, key phrases, readability, summary.
Uses NLTK + scikit-learn (no heavy dependencies).
"""

from __future__ import annotations
import re
import math
from collections import Counter
from typing import Dict, Any, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

for pkg in ["punkt_tab", "punkt", "stopwords", "wordnet", "vader_lexicon", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class NLPAnalyzer:
    """Full NLP analysis suite for project document text."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Add domain stop words
        self.stop_words.update(["project", "team", "work", "use", "using", "also", "would", "could"])

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run all NLP analyses and return consolidated results dict."""
        text = self._clean(text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        content_words = [
            self.lemmatizer.lemmatize(w)
            for w in words
            if w.isalpha() and w not in self.stop_words and len(w) > 2
        ]

        return {
            "sentiment": self._sentiment(text),
            "sentiment_breakdown": self._sentiment_breakdown(sentences),
            "keywords": self._tfidf_keywords(content_words),
            "key_phrases": self._key_phrases(text),
            "summary": self._extractive_summary(sentences, n=4),
            "word_count": len(words),
            "unique_terms": len(set(content_words)),
            "sentence_count": len(sentences),
            "readability_grade": self._readability(text, sentences, words),
            "top_bigrams": self._bigrams(content_words),
        }

    # ── Sentiment ────────────────────────────────────────────────────────────

    def _sentiment(self, text: str) -> Dict[str, Any]:
        scores = self.sia.polarity_scores(text[:5000])
        compound = scores["compound"]
        if compound >= 0.05:
            label, emoji = "Positive", "🟢"
        elif compound <= -0.05:
            label, emoji = "Negative", "🔴"
        else:
            label, emoji = "Neutral", "🟡"
        return {
            "label": f"{emoji} {label}",
            "score": abs(compound),
            "compound": compound
        }

    def _sentiment_breakdown(self, sentences: List[str]) -> Dict[str, float]:
        pos = neg = neu = 0
        for s in sentences:
            sc = self.sia.polarity_scores(s)["compound"]
            if sc >= 0.05:
                pos += 1
            elif sc <= -0.05:
                neg += 1
            else:
                neu += 1
        total = max(len(sentences), 1)
        return {"positive": pos / total, "negative": neg / total, "neutral": neu / total}

    # ── Keywords via TF-IDF approximation ────────────────────────────────────

    def _tfidf_keywords(self, words: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
        freq = Counter(words)
        total = sum(freq.values()) or 1
        # IDF approximation: penalize very common words
        scores = {
            w: (count / total) * math.log(total / count + 1)
            for w, count in freq.items()
            if count >= 2
        }
        sorted_kw = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Normalize scores to 0-1
        if sorted_kw:
            max_score = sorted_kw[0][1]
            return [(w, s / max_score) for w, s in sorted_kw[:top_n]]
        return []

    # ── Key phrases (noun chunks via POS tagging) ────────────────────────────

    def _key_phrases(self, text: str, top_n: int = 12) -> List[str]:
        tokens = word_tokenize(text[:3000])
        tagged = nltk.pos_tag(tokens)
        phrases = []
        current = []
        for word, tag in tagged:
            if tag.startswith("NN") or tag.startswith("JJ"):
                current.append(word.lower())
            else:
                if len(current) >= 2:
                    phrase = " ".join(current)
                    if not any(sw in phrase.split() for sw in self.stop_words):
                        phrases.append(phrase)
                current = []
        phrase_counts = Counter(phrases)
        return [p for p, _ in phrase_counts.most_common(top_n)]

    # ── Extractive summarization (top-scored sentences) ──────────────────────

    def _extractive_summary(self, sentences: List[str], n: int = 4) -> str:
        if len(sentences) <= n:
            return " ".join(sentences)
        word_freq = Counter()
        for s in sentences:
            for w in word_tokenize(s.lower()):
                if w.isalpha() and w not in self.stop_words:
                    word_freq[w] += 1

        def score(sent: str) -> float:
            return sum(word_freq.get(w.lower(), 0) for w in word_tokenize(sent))

        scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
        top_indices = sorted([i for i, _ in scored[:n]])
        return " ".join(sentences[i] for i in top_indices)

    # ── Readability (Flesch-Kincaid Grade Level) ─────────────────────────────

    def _readability(self, text: str, sentences: List[str], words: List[str]) -> str:
        alpha_words = [w for w in words if w.isalpha()]
        syllables = sum(self._count_syllables(w) for w in alpha_words)
        n_words = max(len(alpha_words), 1)
        n_sents = max(len(sentences), 1)
        fk = 0.39 * (n_words / n_sents) + 11.8 * (syllables / n_words) - 15.59
        grade = max(1, min(int(fk), 16))
        labels = {
            range(1, 6): "Elementary",
            range(6, 9): "Middle School",
            range(9, 12): "High School",
            range(12, 14): "College",
            range(14, 17): "Graduate",
        }
        for r, label in labels.items():
            if grade in r:
                return f"Grade {grade} ({label})"
        return f"Grade {grade}"

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = sum(1 for i, c in enumerate(word) if c in vowels and (i == 0 or word[i - 1] not in vowels))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    # ── Bigrams ───────────────────────────────────────────────────────────────

    def _bigrams(self, words: List[str], top_n: int = 8) -> List[Tuple[str, int]]:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        return Counter(bigrams).most_common(top_n)

    # ── Text cleaning ─────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^\w\s\.\,\!\?\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
