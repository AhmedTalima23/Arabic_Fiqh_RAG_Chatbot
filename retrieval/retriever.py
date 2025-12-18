"""
Retriever module for semantic search using FAISS or Chroma.
Handles Arabic query processing and vector similarity search.
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml
from loguru import logger


class ArabicRetriever:
    """Semantic search retriever for Arabic Fiqh text chunks."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize retriever with embeddings and index."""
        self.config = self._load_config(config_path)
        self.embedding_config = self.config.get("embeddings", {})
        self.retrieval_config = self.config.get("retrieval", {})
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_config['model_name']}")
        device = self.embedding_config.get("device", "cpu")
        self.embedding_model = SentenceTransformer(
            self.embedding_config["model_name"],
            device=device
        )
        
        # Load FAISS index
        self.index = self._load_index()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"Retriever initialized with {len(self.metadata)} documents")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_index(self) -> faiss.IndexFlatL2:
        """Load FAISS index from disk."""
        index_path = self.embedding_config["index_path"]
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        return faiss.read_index(index_path)

    def _load_metadata(self) -> List[Dict]:
        """Load chunk metadata."""
        metadata_path = self.config["data"]["metadata_file"]
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found at {metadata_path}")
            return []
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Arabic query string
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of retrieved documents with metadata and similarity scores
        """
        if top_k is None:
            top_k = self.retrieval_config.get("top_k", 3)
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for distance, idx in zip(distances[0], indices):
            if idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                doc["similarity_score"] = 1 / (1 + float(distance))  # Convert L2 to similarity
                results.append(doc)
        
        return results

    def retrieve_with_threshold(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve documents and filter by confidence threshold.
        
        Returns:
            Tuple of (documents, avg_confidence)
        """
        threshold = self.retrieval_config.get("score_threshold", 0.5)
        results = self.retrieve(query, top_k)
        
        filtered = [r for r in results if r["similarity_score"] >= threshold]
        avg_score = np.mean([r["similarity_score"] for r in filtered]) if filtered else 0.0
        
        return filtered, avg_score

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add new documents to the index.
        
        Args:
            embeddings: Document embeddings (shape: [n_docs, embedding_dim])
            metadata: List of metadata dicts for each document
        """
        embeddings = embeddings.astype(np.float32)
        
        if len(self.index) == 0:
            # Create new index
            self.index.add(embeddings)
        else:
            # Add to existing index
            self.index.add(embeddings)
        
        # Update metadata
        self.metadata.extend(metadata)
        
        # Save index and metadata
        self._save_index()
        self._save_metadata()
        
        logger.info(f"Added {len(embeddings)} documents to index")

    def _save_index(self):
        """Save FAISS index to disk."""
        index_path = self.embedding_config["index_path"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_path = self.config["data"]["metadata_file"]
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def search_by_metadata(
        self, field: str, value: str
    ) -> List[Dict]:
        """
        Search documents by metadata field (e.g., book name, madhhab).
        
        Args:
            field: Metadata field name
            value: Value to search for
            
        Returns:
            List of matching documents
        """
        return [doc for doc in self.metadata if doc.get(field) == value]
