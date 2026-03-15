# Code Style Guidelines

## 🎯 General Principles

- **Readability over cleverness** — Write code that is easy to understand and maintain
- **Consistency** — Follow established patterns throughout the codebase
- **Simplicity** — Prefer simple, direct solutions; avoid over-engineering

## 📝 Naming Conventions

### Python

- **Files/Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Variables**: `snake_case` — descriptive and meaningful

### JavaScript/TypeScript

- **Files**: `kebab-case.js` or `PascalCase.tsx` (for components)
- **Classes/Components**: `PascalCase`
- **Functions/Variables**: `camelCase`
- **Constants**: `UPPER_SNAKE_CASE`

## 📐 Formatting

- **Indentation**: 4 spaces (Python), 2 spaces (JS/TS/HTML/CSS)
- **Max line length**: 100 characters (soft limit), 120 characters (hard limit)
- **Trailing commas**: Always use in multi-line structures
- **Quotes**: Double quotes `"` for Python, single quotes `'` for JS/TS
- **Semicolons**: Required in JS/TS

## 🏗️ Code Structure

### Functions

- Maximum **30 lines** per function (excluding docstrings)
- Single Responsibility — each function does one thing well
- **Type hints** required for all Python function signatures
- **JSDoc** comments for all public JS/TS functions

### Classes

- Maximum **200 lines** per class
- Follow SOLID principles
- Constructor should only initialize — no heavy logic

### Modules/Files

- Maximum **400 lines** per file
- Group related functionality together
- Clear separation of concerns

## 📚 Documentation

- **Docstrings**: Required for all public functions, classes, and modules
- **Inline comments**: Only for non-obvious logic — code should be self-documenting
- **README**: Every major module should have a README explaining its purpose

### Python Docstring Format

```python
def process_data(input_data: dict, mode: str = "default") -> dict:
    """Process input data according to the specified mode.

    Args:
        input_data: Raw data dictionary to be processed.
        mode: Processing mode — 'default', 'strict', or 'lenient'.

    Returns:
        Processed data dictionary with normalized values.

    Raises:
        ValueError: If mode is not recognized.
    """
```

## 🚫 Anti-Patterns to Avoid

- ❌ God classes/functions that do too many things
- ❌ Deep nesting (max 3 levels)
- ❌ Magic numbers — use named constants
- ❌ Commented-out code — delete it; use version control
- ❌ Global mutable state
- ❌ Wildcard imports (`from module import *`)

## 🔄 Import Ordering

```python
# 1. Standard library imports
import os
import sys

# 2. Third-party imports
import requests
import numpy as np

# 3. Local application imports
from .utils import helper
from .models import DataModel
```

## ⚡ Performance Considerations

> [!IMPORTANT]
> Given the low-VRAM constraint of this project:
>
> - Avoid loading large models or datasets into memory unnecessarily
> - Use generators/iterators for large data processing
> - Profile memory usage before adding new dependencies
> - Prefer lazy loading and on-demand initialization
