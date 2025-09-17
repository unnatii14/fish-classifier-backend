# Contributing to Fish Species Classifier API

Thank you for your interest in contributing to the Fish Species Classifier API! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/unnatii14/fish-classifier-backend/issues) page
- Provide detailed information about the bug or feature request
- Include steps to reproduce for bugs
- Add relevant labels and assignees

### Pull Requests
1. **Fork** the repository
2. **Create** a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make** your changes following our coding standards
4. **Test** your changes thoroughly
5. **Commit** with clear, descriptive messages
   ```bash
   git commit -m "feat: add new fish species classification"
   ```
6. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create** a Pull Request with detailed description

## 📋 Development Setup

### Prerequisites
- Python 3.10+
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fish-classifier-backend.git
cd fish-classifier-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main_minimal:app --reload
```

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Maximum line length: 88 characters (Black formatter)

### Code Quality
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Use type hints where appropriate
- Handle exceptions gracefully

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Test specific endpoint
python -c "
import requests
response = requests.get('http://localhost:8000/health')
print(response.json())
"
```

### Adding Tests
- Add tests for new features in `tests/` directory
- Test both success and error cases
- Mock external dependencies
- Aim for >80% code coverage

## 🚀 Deployment Testing

### Local Testing
```bash
# Test minimal version
uvicorn main_minimal:app --host 0.0.0.0 --port 8000

# Test full version (if model available)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Railway Testing
- Test deployment on Railway before submitting PR
- Ensure all endpoints work correctly
- Verify health checks pass

## 📚 Documentation

### API Documentation
- Update OpenAPI/Swagger documentation
- Add examples for new endpoints
- Document request/response schemas

### README Updates
- Update feature lists for new functionality
- Add usage examples
- Update deployment instructions if needed

## 🐛 Bug Reports

When reporting bugs, include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs
- Screenshots if applicable

## 💡 Feature Requests

For new features, provide:
- Clear description of the feature
- Use case and benefits
- Proposed implementation approach
- Breaking changes (if any)

## 🔒 Security

- Report security vulnerabilities privately
- Don't include sensitive data in issues/PRs
- Follow secure coding practices
- Validate all inputs

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private matters

## 🏆 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Added to GitHub contributors list

Thank you for contributing to make this project better! 🚀