"""
Unit tests for generation module.
"""

import pytest
from generation.answer_generation import AnswerGenerator


class TestAnswerGenerator:
    """Tests for AnswerGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = AnswerGenerator()
    
    def test_format_answer(self):
        """Test answer formatting."""
        answer = "الربا حرام بالإجماع"
        sources = [{"book": "بداية المجتهد"}]
        result = self.generator.format_answer(answer, sources)
        
        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == answer
    
    def test_validate_answer(self):
        """Test answer validation."""
        valid_answer = "ح" * 100  # Long enough
        invalid_answer = "ح"  # Too short
        
        assert self.generator.validate_answer(valid_answer)
        assert not self.generator.validate_answer(invalid_answer)
    
    def test_export_response_json(self):
        """Test JSON export."""
        response = {
            "answer": "test",
            "confidence": 0.9,
            "sources": []
        }
        exported = self.generator.export_response(response, format="json")
        assert isinstance(exported, str)
        assert '"answer"' in exported
