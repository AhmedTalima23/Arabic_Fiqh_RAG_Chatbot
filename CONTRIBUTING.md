# Contributing to Arabic Fiqh RAG Chatbot

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/Arabic_Fiqh_RAG_Chatbot.git
cd Arabic_Fiqh_RAG_Chatbot
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 4. Test Locally

```bash
pytest tests/
flake8 .
black .
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add descriptive commit message"
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

- Provide clear description of changes
- Reference any related issues
- Include test results

## Code Style

- Use Black for formatting: `black .`
- Use Flake8 for linting: `flake8 .`
- Follow PEP 8 conventions
- Write docstrings for all functions and classes

## Testing

- Write unit tests for new features
- Ensure all tests pass: `pytest tests/ -v`
- Maintain >80% code coverage

## Adding Fiqh Sources

When adding new Fiqh sources:

1. Place texts in `data/raw_books/`
2. Update `data/metadata.json` with:
   - Book name
   - Author
   - Madhab (Islamic school)
   - Period (Classical/Medieval/Modern)
   - Source type

3. Run preprocessing:
   ```bash
   python scripts/preprocess_books.py
   python embeddings/generate_embeddings.py
   ```

## Issues and Discussion

- Report bugs via GitHub Issues
- Discuss major changes in Discussions
- Be respectful in all communications

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have any questions.

Thank you for contributing!
