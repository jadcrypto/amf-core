# AI_NEW_GEN - Project Instructions

## 📋 Project Overview

This project is an AI-powered system designed to run on **resource-constrained hardware** (low VRAM ~771MB–1.1GB). It leverages local LLM inference via **Ollama** with optimized lightweight models.

## 🧠 Core Model Configuration

- **Selected Model**: `qwen2.5:0.5b` — the lightest stable option
- **RAM Requirement**: ~400MB
- **Context Window**: `num_ctx 2048` (reduced to conserve memory)
- **Mode**: Low VRAM mode enabled
- **System Prompt Strategy**: Chain-of-Thought reasoning to compensate for small parameter size

## 🏗️ Architecture Principles

1. **Resource Efficiency First** — Every feature must be viable on low-VRAM hardware
2. **Graceful Degradation** — System must never crash due to memory pressure; degrade functionality instead
3. **Modular Design** — Components should be loosely coupled and independently testable
4. **Local-First AI** — Prioritize Ollama/local inference; cloud APIs are optional fallbacks

## 📁 Project Structure

```
AI_NEW_GEN/
├── .claude/
│   ├── CLAUDE.md              # This file — main project instructions
│   └── rules/
│       ├── code-style.md      # Code style guidelines
│       ├── testing.md         # Testing conventions
│       └── security.md        # Security requirements
├── conv.txt                   # Historical optimization notes
└── ...                        # Application source code
```

## 🔧 Development Environment

- **OS**: Windows
- **AI Backend**: Ollama (local)
- **Primary Model**: `qwen2.5:0.5b`
- **Hardware Constraint**: Limited GPU VRAM (~1GB)

## ⚡ Quick Commands

```bash
# Pull the lightweight model
ollama run qwen2.5:0.5b

# Export model config for editing
ollama show qwen2.5:0.5b --modelfile > my_model_config

# Create custom model from Modelfile
ollama create my-optimized-model -f ./Modelfile
```

## 📜 Rules

All contributors must follow the rules defined in `.claude/rules/`:

- **[code-style.md](rules/code-style.md)** — Naming, formatting, and structural conventions
- **[testing.md](rules/testing.md)** — Test coverage, frameworks, and CI expectations
- **[security.md](rules/security.md)** — Secrets management, input validation, and security hardening

## 🚨 Critical Constraints

> [!CAUTION]
> This system runs on hardware with **severely limited VRAM**. Never introduce dependencies or features that require >1GB GPU memory without explicit approval.

> [!IMPORTANT]
> Always test with `qwen2.5:0.5b` as the baseline model. Larger models may work on other hardware but must not be assumed.
