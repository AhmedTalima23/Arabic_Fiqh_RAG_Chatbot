"""
Unit tests for retrieval module.
"""

import pytest
import json
from pathlib import Path

from retrieval.query_processing import ArabicQueryProcessor


class TestArabicQueryProcessor:
    """Tests for ArabicQueryProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ArabicQueryProcessor()
    
    def test_normalize(self):
        """Test Arabic text normalization."""
        text = "السلام عليكم ورحمة الله"
        normalized = self.processor.normalize(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_remove_stop_words(self):
        """Test stop word removal."""
        text = "ما حكم الربا في الإسلام"
        filtered = self.processor.remove_stop_words(text)
        assert "في" not in filtered
    
    def test_process(self):
        """Test full query processing."""
        query = "ما حكم الربا؟"
        processed = self.processor.process(query)
        assert isinstance(processed, str)
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "العبادة والمعاملات قسمي الفقه"
        keywords = self.processor.extract_keywords(text, max_keywords=3)
        assert isinstance(keywords, list)
        assert len(keywords) <= 3
