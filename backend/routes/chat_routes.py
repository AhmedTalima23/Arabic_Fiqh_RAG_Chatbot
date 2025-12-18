"""
Chat routes for FastAPI backend.
Handles chat-related endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class Message(BaseModel):
    """Message model."""
    role: str  # "user" or "assistant"
    content: str


class ConversationRequest(BaseModel):
    """Conversation request model."""
    messages: list[Message]
    context_length: int = 3


@router.post("/message")
async def send_message(request: ConversationRequest):
    """
    Send a message in conversation context.
    
    Args:
        request: Conversation request with message history
        
    Returns:
        Assistant response
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get last user message
        last_user_msg = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        
        if not last_user_msg:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Process message (would use RAG chain here)
        logger.info(f"Processing conversation message: {last_user_msg}")
        
        return {
            "role": "assistant",
            "content": "برجاء الانتظار، يتم تطبيق هذه الالقابات..."
        }
    
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
