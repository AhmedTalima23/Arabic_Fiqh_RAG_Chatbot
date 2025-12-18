"""
Arabic query processing module.
Handles normalization, expansion, and preprocessing of Arabic queries.
"""

import re
from typing import List, Set
from pathlib import Path

try:
    from PyArabic.araby import (
        remove_diacritics,
        remove_small_alef,
        remove_hamza,
        normalize_alef,
        normalize_lam_alef,
    )
except ImportError:
    pass

from loguru import logger


class ArabicQueryProcessor:
    """Process and normalize Arabic queries."""

    def __init__(self):
        """Initialize query processor."""
        # Common Arabic stop words
        self.stop_words = {
            "في", "من", "على", "مع", "هو", "هي", "هم", "نحن", "أنت",
            "ما", "الذي", "التي", "اللذان", "اللتان", "وال", "أو", "و",
            "أن", "إن", "كان", "كانت", "يكون", "تكون", "هناك", "هنا"
        }
        
        # Arabic diacritical marks
        self.diacritics = re.compile(
            r'[\u064B-\u0652\u0640]'  # Fatha, Damma, Kasra, Sukun, etc.
        )

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text for better retrieval.
        
        Args:
            text: Arabic text to normalize
            
        Returns:
            Normalized text
        """
        # Remove diacritics
        text = remove_diacritics(text)
        
        # Normalize alef variations
        text = normalize_alef(text)
        text = normalize_lam_alef(text)
        text = remove_small_alef(text)
        
        # Remove hamza for better matching
        text = remove_hamza(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def remove_stop_words(self, text: str) -> str:
        """
        Remove Arabic stop words from text.
        
        Args:
            text: Arabic text
            
        Returns:
            Text without stop words
        """
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered)

    def process(self, query: str, remove_stops: bool = False) -> str:
        """
        Full processing pipeline for Arabic query.
        
        Args:
            query: Raw Arabic query
            remove_stops: Whether to remove stop words
            
        Returns:
            Processed query
        """
        # Normalize
        query = self.normalize(query)
        
        # Optionally remove stop words
        if remove_stops:
            query = self.remove_stop_words(query)
        
        return query

    def expand_query(self, query: str, synonyms_map: dict = None) -> List[str]:
        """
        Expand query with synonyms for better retrieval.
        
        Args:
            query: Original query
            synonyms_map: Dictionary mapping words to synonyms
            
        Returns:
            List of expanded queries
        """
        if synonyms_map is None:
            synonyms_map = self._get_fiqh_synonyms()
        
        expanded = [query]
        words = query.split()
        
        for word in words:
            if word in synonyms_map:
                for synonym in synonyms_map[word]:
                    expanded.append(query.replace(word, synonym))
        
        return expanded

    def _get_fiqh_synonyms(self) -> dict:
        """
        Get common Fiqh synonyms for query expansion.
        
        Returns:
            Dictionary of word synonyms
        """
        return {
            "حكم": ["أحكام", "فتوى", "رأي", "الحكم"],
            "الربا": ["الفائدة", "الزيادة"],
            "الزكاة": ["الزكاة", "الصدقة"],
            "الحج": ["الحج", "المناسك"],
            "البيع": ["البيع", "التبادل", "المعاملة"],
            "الصلاة": ["الصلاة", "الصلوات", "الشعيرة"],
        }

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be enhanced)
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return words[:max_keywords]
