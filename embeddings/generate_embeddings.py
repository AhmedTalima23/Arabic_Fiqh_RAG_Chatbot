"""
Generate embeddings for Fiqh text chunks.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger
from tqdm import tqdm


def load_chunks(
    chunks_metadata_file: str
) -> Tuple[List[str], List[dict]]:
    """
    Load text chunks and metadata.
    
    Args:
        chunks_metadata_file: Path to chunks metadata JSON
        
    Returns:
        Tuple of (chunk_texts, metadata)
    """
    with open(chunks_metadata_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    texts = [chunk.get("text", "") for chunk in chunks_data]
    return texts, chunks_data


def create_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/AraBERT-base-v2",
    batch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:
    """
    Create embeddings for texts.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the embedding model
        batch_size: Batch size for encoding
        device: Device to use (cuda/cpu)
        
    Returns:
        Embeddings array
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    logger.info(f"Creating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings.astype(np.float32)


def create_faiss_index(
    embeddings: np.ndarray,
    index_path: str
) -> faiss.IndexFlatL2:
    """
    Create and save FAISS index.
    
    Args:
        embeddings: Embeddings array
        index_path: Path to save index
        
    Returns:
        FAISS index
    """
    embedding_dim = embeddings.shape[1]
    logger.info(f"Creating FAISS index (dimension: {embedding_dim})...")
    
    # Create index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    
    # Save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info(f"Index saved to {index_path}")
    
    return index


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    chunks_metadata_file = os.path.join(
        config["data"]["chunks_dir"],
        "chunks_metadata.json"
    )
    
    if not os.path.exists(chunks_metadata_file):
        logger.error(f"Chunks file not found: {chunks_metadata_file}")
        logger.info("Please run: python scripts/preprocess_books.py")
    else:
        # Load chunks
        logger.info("Loading chunks...")
        texts, metadata = load_chunks(chunks_metadata_file)
        logger.info(f"Loaded {len(texts)} chunks")
        
        # Create embeddings
        embeddings = create_embeddings(
            texts,
            model_name=config["embeddings"]["model_name"],
            device=config["embeddings"]["device"]
        )
        
        # Create FAISS index
        create_faiss_index(
            embeddings,
            config["embeddings"]["index_path"]
        )
        
        # Save metadata for retrieval
        metadata_file = config["data"]["metadata_file"]
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
        logger.info("Embedding generation complete!")
