# Development Guide

## Setup Development Environment

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/AhmedTalima23/Arabic_Fiqh_RAG_Chatbot.git
cd Arabic_Fiqh_RAG_Chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .
```

## Project Structure

```
Arabic_Fiqh_RAG_Chatbot/
├── data/              # Training and source data
├── embeddings/        # Embedding generation and indexing
├── models/            # Model checkpoints
├── retrieval/         # Semantic search and query processing
├── generation/        # LLM-based answer generation
├── backend/           # FastAPI server
├── notebooks/         # Jupyter notebooks for experiments
├── tests/             # Unit and integration tests
├── scripts/           # Utility scripts
├── config.yaml        # Configuration file
└── requirements.txt   # Python dependencies
```

## Workflow

### 1. Data Preparation

```bash
# Place Fiqh texts in data/raw_books/
# Update data/metadata.json with source information

python scripts/preprocess_books.py      # Clean and chunk texts
python embeddings/generate_embeddings.py # Create embeddings
```

### 2. Local Testing

```bash
# Run interactive demo
python scripts/run_rag_demo.py

# Run unit tests
pytest tests/ -v --cov

# Run linting
flake8 .
black --check .
```

### 3. API Testing

```bash
# Start the API server
python -m uvicorn backend.main:app --reload

# Visit http://localhost:8000/docs for interactive documentation

# Example API call
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما حكم الربا؟",
    "top_k": 3
  }'
```

## Configuration

Edit `config.yaml` to customize:

- **Embedding model**: Change `embeddings.model_name`
- **Chunk size**: Adjust `data.chunk_size`
- **Retrieval parameters**: Modify `retrieval` section
- **API settings**: Configure `api` section

## Key Modules

### Retrieval (`retrieval/`)

- `retriever.py`: FAISS-based semantic search
- `query_processing.py`: Arabic NLP and normalization

### Generation (`generation/`)

- `rag_chain.py`: RAG pipeline orchestration
- `answer_generation.py`: Answer formatting with citations

### Backend (`backend/`)

- `main.py`: FastAPI application
- `routes/chat_routes.py`: Chat endpoints
- `utils/response_formatter.py`: Response formatting

## Adding Features

### Adding a New Endpoint

1. Create function in `backend/routes/`
2. Add request/response models
3. Write tests in `tests/`
4. Update API documentation

### Improving Retrieval

1. Experiment in `notebooks/embeddings_demo.ipynb`
2. Test with `tests/test_retrieval.py`
3. Update configuration and reindex

### Extending Generation

1. Modify prompts in `generation/rag_chain.py`
2. Add formatting logic in `generation/answer_generation.py`
3. Test responses with various queries

## Performance Optimization

### Embeddings

- Use GPU acceleration: Set `embeddings.device: cuda`
- Adjust batch size for memory constraints
- Consider quantization for faster retrieval

### Retrieval

- Use HNSW index for faster search: `index_type: hnsw`
- Implement query expansion for better recall
- Add re-ranking for precision improvement

### Generation

- Implement prompt caching
- Use model quantization
- Add response streaming

## Debugging

### Enable Debug Logging

```python
from loguru import logger
logger.enable("debug")
```

### Common Issues

**Issue**: FAISS index not found
```
Solution: Run embeddings/generate_embeddings.py
```

**Issue**: Arabic text encoding errors
```
Solution: Ensure UTF-8 encoding: open(file, encoding='utf-8')
```

**Issue**: Out of memory during embedding
```
Solution: Reduce batch_size in config.yaml
```

## Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create GitHub release with tag
4. Build and publish to PyPI (if applicable)

## Documentation

- API docs: Auto-generated at `/docs`
- Code docs: Docstrings in all modules
- User guide: README.md
- Developer guide: This file

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Arabic NLP Tools](https://github.com/aalhour/awesome-arabic-nlp)
