# Security Requirements

## 🎯 Security Philosophy

- **Defense in depth** — Multiple layers of protection, never rely on a single control
- **Least privilege** — Components should only access what they absolutely need
- **Fail secure** — On error, default to the most restrictive behavior
- **Zero trust** — Validate all inputs regardless of source

## 🔑 Secrets Management

### Rules

- ❌ **NEVER** commit secrets, API keys, or credentials to version control
- ❌ **NEVER** hardcode secrets in source code
- ❌ **NEVER** log secrets or include them in error messages
- ✅ Use environment variables or `.env` files for local development
- ✅ Use a secrets manager (e.g., Azure Key Vault, AWS Secrets Manager) for production

### Implementation

```python
# ✅ CORRECT — Load from environment
import os
API_KEY = os.environ.get("OLLAMA_API_KEY")
if not API_KEY:
    raise EnvironmentError("OLLAMA_API_KEY is not set")

# ❌ WRONG — Hardcoded secret
API_KEY = "sk-abc123xyz789"
```

### .gitignore Requirements

```gitignore
# Always ignore these
.env
.env.*
*.pem
*.key
secrets/
config/local.yaml
```

## 🛡️ Input Validation

### All User Inputs Must Be Validated

```python
import re
from typing import Optional

def validate_prompt(prompt: str, max_length: int = 4096) -> Optional[str]:
    """Sanitize and validate user prompt before sending to model.

    Args:
        prompt: Raw user input string.
        max_length: Maximum allowed prompt length.

    Returns:
        Sanitized prompt string, or None if invalid.
    """
    if not prompt or not isinstance(prompt, str):
        return None
    
    # Strip control characters
    prompt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)
    
    # Enforce length limit
    if len(prompt) > max_length:
        return None
    
    return prompt.strip()
```

### Validation Checklist

- [ ] All API inputs are type-checked and length-limited
- [ ] File paths are canonicalized and restricted to allowed directories
- [ ] SQL queries use parameterized statements (never string concatenation)
- [ ] JSON/XML payloads are schema-validated before processing

## 🌐 Network Security

### API Communication

- ✅ Use HTTPS for all external API calls
- ✅ Verify SSL/TLS certificates
- ✅ Set reasonable timeouts on all network requests
- ✅ Implement rate limiting for exposed endpoints

```python
import requests

def call_api(url: str, data: dict, timeout: int = 30) -> dict:
    """Make a secure API call with proper timeout and verification."""
    response = requests.post(
        url,
        json=data,
        timeout=timeout,
        verify=True,  # Always verify SSL
    )
    response.raise_for_status()
    return response.json()
```

### Ollama Local Security
>
> [!IMPORTANT]
> Even though Ollama runs locally, apply these safeguards:

- Bind Ollama to `localhost` only — never expose to `0.0.0.0`
- Validate model responses before using them in system operations
- Sanitize any LLM output used in downstream commands (prevent prompt injection)

## 🛑 Prompt Injection Prevention

```python
def sanitize_llm_output(output: str) -> str:
    """Remove potentially dangerous patterns from LLM output.

    Prevents prompt injection and command injection attacks.
    """
    # Remove shell-dangerous characters if output will be used in commands
    dangerous_patterns = [';', '&&', '||', '`', '$(',  '|', '>', '<']
    for pattern in dangerous_patterns:
        output = output.replace(pattern, '')
    
    return output.strip()


def execute_with_llm_output(output: str) -> None:
    """Never directly execute LLM output as system commands."""
    sanitized = sanitize_llm_output(output)
    # Always validate against an allowlist of expected actions
    ALLOWED_ACTIONS = {"summarize", "translate", "analyze"}
    if sanitized.lower() not in ALLOWED_ACTIONS:
        raise ValueError(f"Unauthorized action: {sanitized}")
```

## 📂 File System Security

- ✅ Validate and sanitize all file paths
- ✅ Use path canonicalization to prevent directory traversal
- ✅ Restrict file operations to designated directories
- ❌ Never allow user input to directly construct file paths

```python
from pathlib import Path

ALLOWED_ROOT = Path("d:/AI_NEW_GEN/data")

def safe_read_file(user_path: str) -> str:
    """Read a file only if it's within the allowed directory."""
    resolved = (ALLOWED_ROOT / user_path).resolve()
    if not str(resolved).startswith(str(ALLOWED_ROOT.resolve())):
        raise PermissionError("Access denied: path traversal detected")
    return resolved.read_text(encoding="utf-8")
```

## 🔒 Dependency Security

- ✅ Pin dependency versions in `requirements.txt` / `package.json`
- ✅ Audit dependencies regularly with `pip audit` / `npm audit`
- ✅ Minimize the number of third-party dependencies
- ✅ Review changelogs before upgrading major versions

```bash
# Regular security audits
pip install pip-audit
pip-audit

# For Node.js projects
npm audit
```

## 📋 Security Checklist (Pre-Deployment)

- [ ] All secrets are loaded from environment variables, not hardcoded
- [ ] `.env` and secret files are in `.gitignore`
- [ ] All user inputs are validated and sanitized
- [ ] LLM outputs are sanitized before any system use
- [ ] File paths are restricted and canonicalized
- [ ] Network requests use HTTPS with proper timeouts
- [ ] Dependencies are pinned and audited
- [ ] Ollama is bound to localhost only
- [ ] Error messages do not leak sensitive information
- [ ] Logging excludes secrets, tokens, and PII

## ⚠️ Incident Response

> [!CAUTION]
> If a security vulnerability is discovered:
>
> 1. **Do NOT** commit a fix that reveals the vulnerability details in the commit message
> 2. **Rotate** all potentially compromised credentials immediately
> 3. **Document** the vulnerability privately and notify the team
> 4. **Fix** the vulnerability and deploy the patch
> 5. **Post-mortem** — analyze the root cause and update these guidelines
