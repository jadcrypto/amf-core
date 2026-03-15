# Testing Conventions

## 🎯 Testing Philosophy

- **Test behavior, not implementation** — Focus on what the code does, not how
- **Fast feedback** — Tests should run quickly, especially on limited hardware
- **Deterministic** — Tests must produce consistent results every run
- **Independent** — Each test should be self-contained with no shared state

## 📊 Coverage Requirements

| Category          | Minimum Coverage |
|-------------------|------------------|
| Core logic        | 80%              |
| Utility functions | 90%              |
| API endpoints     | 75%              |
| UI components     | 60%              |
| Integration tests | Key paths only   |

## 🧪 Test Frameworks

### Python

- **Unit Testing**: `pytest` (preferred) or `unittest`
- **Mocking**: `unittest.mock` or `pytest-mock`
- **Coverage**: `pytest-cov`

### JavaScript/TypeScript

- **Unit Testing**: `jest` or `vitest`
- **E2E Testing**: `playwright` (preferred) or `cypress`

## 📁 Test File Structure

```
project/
├── src/
│   ├── module_a/
│   │   ├── __init__.py
│   │   └── processor.py
│   └── module_b/
│       └── handler.py
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── unit/
    │   ├── test_processor.py
    │   └── test_handler.py
    └── integration/
        └── test_pipeline.py
```

## ✍️ Test Naming Convention

```python
# Format: test_<function_name>_<scenario>_<expected_result>

def test_process_data_with_valid_input_returns_normalized_dict():
    ...

def test_process_data_with_empty_input_raises_value_error():
    ...

def test_connect_to_ollama_when_server_down_returns_graceful_error():
    ...
```

## 🏗️ Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange — Set up test data and conditions
    input_data = {"key": "value"}
    expected = {"key": "VALUE"}

    # Act — Execute the code under test
    result = process_data(input_data)

    # Assert — Verify the outcome
    assert result == expected
```

## 🔌 Fixtures and Mocking

### Shared Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def sample_config():
    """Provide a minimal test configuration."""
    return {
        "model": "qwen2.5:0.5b",
        "num_ctx": 2048,
        "low_vram": True,
    }

@pytest.fixture
def mock_ollama_response():
    """Simulate an Ollama API response."""
    return {
        "response": "Test response",
        "done": True,
    }
```

### Mocking External Services

```python
from unittest.mock import patch

def test_query_model_mocks_ollama_api(mock_ollama_response):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_ollama_response
        result = query_model("test prompt")
        assert result == "Test response"
```

## ⚡ Performance Testing

> [!IMPORTANT]
> Given hardware constraints, include memory-aware tests:

```python
import tracemalloc

def test_model_loading_stays_within_memory_budget():
    tracemalloc.start()
    load_model("qwen2.5:0.5b")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 500 * 1024 * 1024  # 500MB limit
```

## 🚫 Testing Anti-Patterns

- ❌ Tests that depend on execution order
- ❌ Tests that require network access (mock external APIs)
- ❌ Tests with hardcoded file paths — use `tmp_path` fixture
- ❌ Tests that take >5 seconds individually
- ❌ Asserting on exact floating-point values — use `pytest.approx()`
- ❌ Testing private implementation details

## 🔄 CI Integration

```yaml
# Run tests before every merge
test:
  script:
    - pip install -r requirements-test.txt
    - pytest tests/ -v --cov=src --cov-report=term-missing
  rules:
    - All tests must pass
    - Coverage must not decrease
```

## ✅ Pre-Commit Checklist

- [ ] All new code has corresponding tests
- [ ] All tests pass locally
- [ ] No test relies on external services without mocking
- [ ] Memory-intensive tests include resource assertions
- [ ] Test names clearly describe the scenario being tested
