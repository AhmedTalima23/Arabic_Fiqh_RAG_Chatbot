# Arabic Fiqh RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot specialized in Islamic Jurisprudence (Fiqh) using Arabic LLMs. This system combines semantic search over classical and contemporary Fiqh texts with generative AI to provide accurate, contextually-grounded answers to Islamic legal questions with proper citations.

## Features

- **Arabic-Native Processing**: Built for Arabic text with proper normalization and tokenization
- **Semantic Search**: FAISS/Chroma-based vector retrieval over chunked Fiqh texts
- **Multi-Madhhab Support**: Includes perspectives from different Islamic schools of thought
- **Citation Management**: Automatic source attribution and reference tracking
- **FastAPI Backend**: RESTful API for easy integration
- **RAG Pipeline**: LangChain integration for retriever + generator combination
- **Extensible Design**: Modular structure for adding new sources or LLMs

## Project Structure

```
Arabic_Fiqh_RAG_Chatbot/
│
├── data/
│   ├── raw_books/                  # Original Fiqh PDFs, texts, or scanned sources
│   ├── cleaned_books/              # Preprocessed plain text versions
│   ├── chunks/                     # Chunked text files for embedding
│   └── metadata.json                # Book metadata: name, chapter, scholar, madhhab
│
├── embeddings/
│   ├── generate_embeddings.py      # Script to create embeddings from chunks
│   ├── embeddings_index.faiss      # FAISS vector store for retrieval
│   └── embeddings_index.chroma     # Optional Chroma DB version
│
├── models/
│   ├── arabic_llm/                 # Pretrained Arabic LLM checkpoint (e.g., Jais, Noor)
│   └── tokenizer/                  # Arabic tokenizer if required
│
├── retrieval/
│   ├── retriever.py                # Semantic search using FAISS/Chroma
│   └── query_processing.py         # Arabic query normalization and processing
│
├── generation/
│   ├── rag_chain.py                # LangChain RAG pipeline
│   └── answer_generation.py        # Answer generation with citations
│
├── backend/
│   ├── main.py                     # FastAPI server
│   ├── routes/
│   │   └── chat_routes.py          # Chat endpoints
│   └── utils/
│       └── response_formatter.py   # Response formatting with citations
│
├── notebooks/
│   ├── preprocessing.ipynb         # Data cleaning & chunking experiments
│   └── embeddings_demo.ipynb       # Embedding & retrieval quality testing
│
├── tests/
│   ├── test_retrieval.py           # Retrieval tests
│   ├── test_generation.py          # Generation tests
│   └── test_end_to_end.py          # Full pipeline tests
│
├── scripts/
│   ├── preprocess_books.py         # Batch preprocessing
│   └── run_rag_demo.py             # Demo script
│
├── requirements.txt                # Dependencies
├── config.yaml                     # Configuration
└── .gitignore                      # Git ignore rules
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AhmedTalima23/Arabic_Fiqh_RAG_Chatbot.git
cd Arabic_Fiqh_RAG_Chatbot

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` with your settings:
- Model paths
- Chunk sizes
- Embedding dimensions
- API keys (if using cloud LLMs)

### Data Preparation

1. Place raw Fiqh PDFs/texts in `data/raw_books/`
2. Run preprocessing:
   ```bash
   python scripts/preprocess_books.py
   ```
3. Generate embeddings:
   ```bash
   python embeddings/generate_embeddings.py
   ```

### Start the Chatbot

```bash
# Local demo
python scripts/run_rag_demo.py

# API server
python -m uvicorn backend.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## API Endpoints

### POST `/ask`

Ask a Fiqh question:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما حكم الربا في الإسلام؟",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    {
      "text": "...",
      "book": "...",
      "chapter": "...",
      "madhhab": "..."
    }
  ],
  "confidence": 0.85
}
```

## Technologies

- **LLM Framework**: LangChain
- **Vector Store**: FAISS / Chroma
- **API**: FastAPI, Uvicorn
- **Arabic NLP**: CAMeL Tools, Farasa, or Arabic Transformers
- **Arabic LLM Options**: Jais, Noor, AraBART, Qwen Arabic
- **Embedding Models**: Arabic-BERT, AraBERTv2, Multilingual-E5

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Sources

1. Place texts in `data/raw_books/`
2. Update `data/metadata.json` with source info
3. Run preprocessing pipeline
4. Regenerate embeddings

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use this project in your research, please cite:

```bibtex
@software{arabic_fiqh_rag,
  title={Arabic Fiqh RAG Chatbot},
  author={Your Name},
  year={2025},
  url={https://github.com/AhmedTalima23/Arabic_Fiqh_RAG_Chatbot}
}
```

## Acknowledgments

- Islamic Fiqh scholars and classical texts contributors
- Arabic NLP community
- Open-source ML/AI community

## Contact & Support

For issues, questions, or suggestions, please open a GitHub issue or contact the maintainers.

---

**Note**: This project is for educational and informational purposes. For critical religious rulings, always consult qualified Islamic scholars.
