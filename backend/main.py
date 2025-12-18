"""
FastAPI backend server for Arabic Fiqh RAG Chatbot.
Exposes RESTful endpoints for chatbot interaction.
"""

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from generation.rag_chain import FiqhRAGChain
from generation.answer_generation import AnswerGenerator


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="Arabic Fiqh question")
    top_k: int = Field(3, description="Number of sources to retrieve", ge=1, le=10)
    include_sources: bool = Field(True, description="Include source documents in response")


class Citation(BaseModel):
    """Citation model."""
    text: str
    book: str
    chapter: Optional[str] = None
    madhhab: Optional[str] = None
    relevance: float


class ChatResponse(BaseModel):
    """Chat response model."""
    query: str
    answer: str
    confidence: float
    sources: List[Citation] = []
    retrieval_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    embeddings_available: bool


# Initialize components on startup
rag_chain = None
answer_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    global rag_chain, answer_generator
    logger.info("Starting up chatbot...")
    try:
        rag_chain = FiqhRAGChain("config.yaml")
        answer_generator = AnswerGenerator()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down chatbot...")


# Create FastAPI app
app = FastAPI(
    title="Arabic Fiqh RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for Islamic Jurisprudence",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return HealthResponse(
        status="healthy" if rag_chain else "initializing",
        model="FiqhRAGChain v1",
        embeddings_available=rag_chain is not None
    )


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Ask a Fiqh question.
    
    Args:
        request: Chat request with query and parameters
        
    Returns:
        Chat response with answer and sources
    """
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Generate answer
        result = rag_chain.generate_answer(
            query=request.query,
            top_k=request.top_k
        )
        
        # Format sources
        sources = []
        if request.include_sources:
            sources = [
                Citation(
                    text=src.get("text", "")[:200],  # Truncate text
                    book=src.get("book", "Unknown"),
                    chapter=src.get("chapter"),
                    madhhab=src.get("madhhab"),
                    relevance=src.get("relevance", 0.0)
                )
                for src in result.get("sources", [])
            ]
        
        return ChatResponse(
            query=request.query,
            answer=result.get("context", ""),
            confidence=result.get("confidence", 0.0),
            sources=sources,
            retrieval_count=result.get("retrieval_count", 0)
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/sources")
async def list_sources():
    """
    List available Fiqh sources.
    
    Returns:
        List of available sources with metadata
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        metadata = rag_chain.retriever.metadata
        
        # Extract unique sources
        sources = {}
        for doc in metadata:
            book = doc.get("book", "Unknown")
            if book not in sources:
                sources[book] = {
                    "book": book,
                    "madhhab": doc.get("madhhab"),
                    "author": doc.get("author"),
                    "document_count": 0
                }
            sources[book]["document_count"] += 1
        
        return list(sources.values())
    
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get chatbot statistics.
    
    Returns:
        Statistics about the chatbot state
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "total_documents": len(rag_chain.retriever.metadata),
        "embedding_model": rag_chain.config["embeddings"]["model_name"],
        "index_type": rag_chain.config["embeddings"]["index_type"],
        "retrieval_top_k": rag_chain.config["retrieval"]["top_k"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
