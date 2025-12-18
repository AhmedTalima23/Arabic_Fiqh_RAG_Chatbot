"""
Quick demo script for testing RAG pipeline locally.
"""

import sys
from loguru import logger

from generation.rag_chain import FiqhRAGChain
from generation.answer_generation import AnswerGenerator


def main():
    """
    Run interactive demo.
    """
    logger.info("Initializing RAG Chatbot...")
    
    try:
        # Initialize components
        rag_chain = FiqhRAGChain("config.yaml")
        answer_gen = AnswerGenerator()
        
        logger.info("Chatbot ready! Enter your questions in Arabic.")
        print("\n=== Arabic Fiqh RAG Chatbot Demo ===")
        print("الرجاء إدخال أسئلتك باللغة العربية (\"quit\" للخروج)")
        print("="*40 + "\n")
        
        while True:
            # Get user input
            query = input("\nسؤالك: ").strip()
            
            if query.lower() in ["quit", "exit", "خروج"]:
                logger.info("Exiting demo...")
                break
            
            if not query:
                continue
            
            try:
                # Generate answer
                logger.info(f"Processing query: {query}")
                response = rag_chain.generate_answer(query, top_k=3)
                
                # Display response
                print("\nالإجابة:")
                print(response.get("context", ""))
                print(f"\nدرجة الثقة: {response.get('confidence', 0):.2%}")
                print(f"عدد المصادر: {response.get('retrieval_count', 0)}")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nخطأ: {e}")
    
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
