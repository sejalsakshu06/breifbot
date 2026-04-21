"""
Unit tests for NLP Analyzer and File Handler.
Run: pytest tests/ -v
"""

import pytest
from src.nlp.analyzer import NLPAnalyzer
from src.utils.file_handler import FileHandler
import io


# ── NLP Analyzer Tests ────────────────────────────────────────────────────────

class TestNLPAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return NLPAnalyzer()

    @pytest.fixture
    def sample_text(self):
        return """
        The AI model development project is progressing well. The team successfully
        implemented a new deep learning algorithm that improved performance by 35%.
        However, there are some risks around data quality and timeline delays.
        The next steps include deploying the model to production and monitoring
        its performance metrics. Overall sentiment is positive with minor concerns.
        """

    def test_analyze_returns_all_keys(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        expected_keys = [
            "sentiment", "sentiment_breakdown", "keywords", "key_phrases",
            "summary", "word_count", "unique_terms", "sentence_count",
            "readability_grade", "top_bigrams"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_sentiment_label(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert result["sentiment"]["label"] in [
            "🟢 Positive", "🔴 Negative", "🟡 Neutral"
        ]

    def test_keywords_not_empty(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert len(result["keywords"]) > 0

    def test_keyword_scores_normalized(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        for _, score in result["keywords"]:
            assert 0.0 <= score <= 1.0

    def test_word_count_positive(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert result["word_count"] > 0

    def test_sentiment_breakdown_sums_to_one(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        breakdown = result["sentiment_breakdown"]
        total = sum(breakdown.values())
        assert abs(total - 1.0) < 0.01

    def test_summary_is_string(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_empty_text_handled(self, analyzer):
        result = analyzer.analyze("   ")
        assert isinstance(result, dict)

    def test_readability_grade_string(self, analyzer, sample_text):
        result = analyzer.analyze(sample_text)
        assert "Grade" in result["readability_grade"]


# ── File Handler Tests ────────────────────────────────────────────────────────

class TestFileHandler:

    @pytest.fixture
    def handler(self):
        return FileHandler()

    def test_read_txt(self, handler):
        class MockFile:
            name = "test.txt"
            def read(self): return b"Hello world this is a test document."

        results = handler.process_files([MockFile()])
        assert len(results) == 1
        assert "Hello world" in results[0]["content"]

    def test_read_json(self, handler):
        import json
        data = {"project": "AI System", "status": "active", "progress": 75}

        class MockFile:
            name = "data.json"
            def read(self): return json.dumps(data).encode()

        results = handler.process_files([MockFile()])
        assert "project" in results[0]["content"]

    def test_read_csv(self, handler):
        csv_content = "name,status,score\nTask A,done,95\nTask B,pending,60\n"

        class MockFile:
            name = "tasks.csv"
            def read(self): return csv_content.encode()

        results = handler.process_files([MockFile()])
        assert "Task A" in results[0]["content"]

    def test_empty_file_skipped(self, handler):
        class MockFile:
            name = "empty.txt"
            def read(self): return b"   "

        results = handler.process_files([MockFile()])
        assert len(results) == 0
