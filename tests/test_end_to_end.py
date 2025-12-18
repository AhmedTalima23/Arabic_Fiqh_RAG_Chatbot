"""
End-to-end integration tests.
"""

import pytest
from generation.rag_chain import FiqhRAGChain


class TestRAGChain:
    """Tests for RAG chain integration."""
    
    def test_chain_initialization(self):
        """Test RAG chain initialization."""
        try:
            chain = FiqhRAGChain("config.yaml")
            assert chain is not None
            assert chain.retriever is not None
            assert chain.query_processor is not None
        except FileNotFoundError:
            # Config file not found in test environment
            pytest.skip("config.yaml not found")
    
    def test_prompt_template_creation(self):
        """Test prompt template creation."""
        try:
            chain = FiqhRAGChain("config.yaml")
            assert chain.prompt_template is not None
            assert "question" in chain.prompt_template.input_variables
            assert "context" in chain.prompt_template.input_variables
        except FileNotFoundError:
            pytest.skip("config.yaml not found")
