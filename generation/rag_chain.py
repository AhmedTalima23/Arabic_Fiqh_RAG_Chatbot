"""
RAG (Retrieval-Augmented Generation) chain combining retriever and LLM.
Manages the pipeline of document retrieval -> prompt construction -> generation.
"""

import os
from typing import Dict, List, Optional, Tuple

import yaml
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from loguru import logger

from retrieval.retriever import ArabicRetriever
from retrieval.query_processing import ArabicQueryProcessor


class FiqhRAGChain:
    """RAG chain for Arabic Fiqh question answering."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RAG chain."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.retriever = ArabicRetriever(config_path)
        self.query_processor = ArabicQueryProcessor()
        
        # Prompt template for Fiqh context
        self.prompt_template = self._create_prompt_template()
        
        logger.info("RAG chain initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create prompt template for Fiqh question answering.
        
        Returns:
            PromptTemplate object
        """
        template = """أنت خبير متخصص في الفقه الإسلامي والشريعة الإسلامية.
باستخدام المعلومات التالية من مصادر موثوقة، أجب على السؤال بشكل دقيق ومفصل.

المصادر:
{context}

السؤال: {question}

الإجابة:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def retrieve_context(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Arabic query
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, average_confidence)
        """
        # Process query
        processed_query = self.query_processor.process(query)
        
        # Retrieve documents
        docs, confidence = self.retriever.retrieve_with_threshold(processed_query, top_k)
        
        logger.info(f"Retrieved {len(docs)} documents with avg confidence {confidence:.2f}")
        
        return docs, confidence

    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            source = doc.get("book", "Unknown")
            chapter = doc.get("chapter", "")
            madhhab = doc.get("madhhab", "")
            
            # Format source attribution
            source_str = f"[{source}"
            if chapter:
                source_str += f" - {chapter}"
            if madhhab:
                source_str += f" ({madhhab})"
            source_str += "]"
            
            # Get text content
            text = doc.get("text", "")
            
            context_parts.append(f"{idx}. {text}\n{source_str}")
        
        return "\n\n".join(context_parts)

    def generate_answer(
        self, query: str, top_k: Optional[int] = None, llm=None
    ) -> Dict:
        """
        Generate answer using retrieved context.
        
        Args:
            query: Arabic query
            top_k: Number of documents to use
            llm: Language model (uses configured model if None)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve context
        docs, confidence = self.retrieve_context(query, top_k)
        
        if not docs:
            return {
                "answer": "آسف، لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك.",
                "sources": [],
                "confidence": 0.0,
                "retrieval_count": 0
            }
        
        # Format context
        context = self.format_context(docs)
        
        # For now, return a structured response
        # In production, this would use the actual LLM
        return {
            "query": query,
            "context": context,
            "sources": [{
                "text": doc.get("text"),
                "book": doc.get("book"),
                "chapter": doc.get("chapter"),
                "madhhab": doc.get("madhhab"),
                "relevance": doc.get("similarity_score", 0.0)
            } for doc in docs],
            "confidence": float(confidence),
            "retrieval_count": len(docs)
        }
