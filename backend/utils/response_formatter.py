"""
Response formatter utilities.
Handles formatting of chatbot responses with citations and metadata.
"""

from typing import Dict, List, Optional


def format_with_citations(
    answer: str,
    sources: List[Dict],
    include_confidence: bool = True,
    confidence: float = 0.0
) -> Dict:
    """
    Format answer with citations.
    
    Args:
        answer: Answer text
        sources: List of source documents
        include_confidence: Whether to include confidence score
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "answer": answer,
        "sources": sources,
        "formatted_answer": answer
    }
    
    if include_confidence:
        response["confidence"] = confidence
    
    # Add citations to answer
    if sources:
        citations_text = "\n\nمصادر:\n"
        for idx, source in enumerate(sources, 1):
            citation = f"[{idx}] {source.get('book', 'Unknown')}"
            if source.get('chapter'):
                citation += f" - {source['chapter']}"
            if source.get('madhhab'):
                citation += f" ({source['madhhab']})"
            citations_text += f"{citation}\n"
        
        response["formatted_answer"] = answer + citations_text
    
    return response


def create_error_response(error_code: str, message: str) -> Dict:
    """
    Create standardized error response.
    
    Args:
        error_code: Error code identifier
        message: Error message
        
    Returns:
        Error response dictionary
    """
    return {
        "error": {
            "code": error_code,
            "message": message
        },
        "success": False
    }


def create_success_response(data: Dict, message: Optional[str] = None) -> Dict:
    """
    Create standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        Success response dictionary
    """
    response = {
        "success": True,
        "data": data
    }
    if message:
        response["message"] = message
    return response
