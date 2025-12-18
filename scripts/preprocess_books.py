"""
Preprocess Fiqh books: PDF extraction, text cleaning, and chunking.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import yaml
from loguru import logger
from tqdm import tqdm


def preprocess_texts(
    raw_dir: str,
    cleaned_dir: str,
    config_path: str = "config.yaml"
) -> List[Dict]:
    """
    Preprocess raw texts from raw_dir and save cleaned versions.
    
    Args:
        raw_dir: Directory containing raw texts/PDFs
        cleaned_dir: Output directory for cleaned texts
        config_path: Path to configuration file
        
    Returns:
        List of document metadata
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    os.makedirs(cleaned_dir, exist_ok=True)
    
    metadata = []
    
    # Iterate through raw files
    raw_path = Path(raw_dir)
    for file_path in tqdm(raw_path.glob("*.txt")):
        try:
            logger.info(f"Processing {file_path.name}...")
            
            # Read raw text
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Basic cleaning
            text = clean_text(text)
            
            # Save cleaned version
            cleaned_path = Path(cleaned_dir) / file_path.name
            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Create metadata
            meta = {
                "filename": file_path.name,
                "book": file_path.stem,
                "text_length": len(text),
                "cleaned": True
            }
            metadata.append(meta)
            
            logger.info(f"Saved cleaned version: {cleaned_path}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    return metadata


def clean_text(text: str) -> str:
    """
    Clean Arabic text.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove page numbers and headers (basic approach)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if len(line.strip()) > 5 and not line.isdigit():
            cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)


def chunk_texts(
    cleaned_dir: str,
    chunks_dir: str,
    chunk_size: int = 512,
    overlap: int = 50,
    config_path: str = "config.yaml"
):
    """
    Chunk cleaned texts for embedding.
    
    Args:
        cleaned_dir: Directory with cleaned texts
        chunks_dir: Output directory for chunks
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        config_path: Configuration file path
    """
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunks_metadata = []
    
    for file_path in tqdm(Path(cleaned_dir).glob("*.txt")):
        try:
            logger.info(f"Chunking {file_path.name}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Create chunks
            chunks = create_chunks(text, chunk_size, overlap)
            
            # Save chunks
            for idx, chunk in enumerate(chunks):
                chunk_meta = {
                    "book": file_path.stem,
                    "chunk_id": idx,
                    "text": chunk,
                    "length": len(chunk),
                    "chapter": None,  # Can be extracted from structure
                    "madhhab": None   # Can be set manually
                }
                chunks_metadata.append(chunk_meta)
            
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error chunking {file_path.name}: {e}")
    
    # Save all chunks metadata
    metadata_file = os.path.join(chunks_dir, "chunks_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks_metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created {len(chunks_metadata)} total chunks")
    return chunks_metadata


def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Create overlapping chunks from text.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


if __name__ == "__main__":
    # Default configuration
    config = {
        "raw_books_dir": "data/raw_books",
        "cleaned_books_dir": "data/cleaned_books",
        "chunks_dir": "data/chunks",
        "chunk_size": 512,
        "overlap": 50
    }
    
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Clean texts
    logger.info("Step 1: Cleaning texts...")
    metadata = preprocess_texts(
        config["raw_books_dir"],
        config["cleaned_books_dir"]
    )
    
    # Step 2: Chunk texts
    logger.info("Step 2: Chunking texts...")
    chunks_metadata = chunk_texts(
        config["cleaned_books_dir"],
        config["chunks_dir"],
        chunk_size=config["chunk_size"],
        overlap=config["overlap"]
    )
    
    logger.info("Preprocessing complete!")
