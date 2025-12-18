"""
Answer generation module.
Handles LLM-based answer generation with citations and formatting.
"""

import json
from typing import Dict, List, Optional

from loguru import logger


class AnswerGenerator:
    """Generate formatted answers with citations."""

    def __init__(self, config: Dict = None):
        """Initialize answer generator."""
        self.config = config or {}

    def format_answer(
        self,
        answer_text: str,
        sources: List[Dict],
        confidence: float,
        include_citations: bool = True
    ) -> Dict:
        """
        Format answer with proper citations.
        
        Args:
            answer_text: Generated answer text
            sources: List of source documents
            confidence: Confidence score
            include_citations: Whether to include citations
            
        Returns:
            Formatted response dictionary
        """
        response = {
            "answer": answer_text,
            "confidence": confidence,
            "sources": sources,
            "citation_count": len(sources)
        }
        
        if include_citations:
            response["formatted_answer"] = self._add_citations(answer_text, sources)
        
        return response

    def _add_citations(
        self,
        answer_text: str,
        sources: List[Dict]
    ) -> str:
        """
        Add citations to answer text.
        
        Args:
            answer_text: Original answer
            sources: Source documents
            
        Returns:
            Answer with citations
        """
        # Simple citation implementation
        if not sources:
            return answer_text
        
        # Create citation list
        citations = []
        for idx, source in enumerate(sources, 1):
            citation = f"[{idx}] {source.get('book', 'Unknown')}"
            if source.get('chapter'):
                citation += f" - {source['chapter']}"
            citations.append(citation)
        
        # Combine answer with citations
        formatted = answer_text + "\n\n" + "المصادر:\n"
        formatted += "\n".join(citations)
        
        return formatted

    def generate_summary(
        self,
        full_response: Dict,
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of the answer.
        
        Args:
            full_response: Full response dictionary
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        answer = full_response.get("answer", "")
        
        # Simple truncation (can be enhanced with actual summarization)
        if len(answer) > max_length:
            summary = answer[:max_length].rsplit(' ', 1)[0] + "..."
        else:
            summary = answer
        
        return summary

    def validate_answer(
        self,
        answer: str,
        min_length: int = 50,
        max_length: int = 5000
    ) -> bool:
        """
        Validate generated answer.
        
        Args:
            answer: Answer text to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            
        Returns:
            True if valid, False otherwise
        """
        if not answer:
            return False
        
        return min_length <= len(answer) <= max_length

    def export_response(
        self,
        response: Dict,
        format: str = "json"
    ) -> str:
        """
        Export response in specified format.
        
        Args:
            response: Response dictionary
            format: Output format ('json', 'text', 'markdown')
            
        Returns:
            Formatted response string
        """
        if format == "json":
            return json.dumps(response, ensure_ascii=False, indent=2)
        
        elif format == "text":
            lines = []
            lines.append("الإجابة:")
            lines.append(response.get("answer", ""))
            lines.append(f"\nدرجة الثقة: {response.get('confidence', 0):.2%}")
            lines.append(f"عدد المصادر: {response.get('citation_count', 0)}")
            return "\n".join(lines)
        
        elif format == "markdown":
            lines = []
            lines.append("# الإجابة")
            lines.append(response.get("answer", ""))
            lines.append(f"**درجة الثقة:** {response.get('confidence', 0):.2%}")
            lines.append("## المصادر")
            for source in response.get("sources", []):
                lines.append(f"- {source.get('book')}: {source.get('chapter')}")
            return "\n".join(lines)
        
        return str(response)
