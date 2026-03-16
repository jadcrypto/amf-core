# Changelog

All notable changes to `amf-core` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-03-16

### Added
- `AMFEngine` — new direct molecular inference engine
  - `predict(prompt)` — next-token prediction via deep layer inference
  - `generate(prompt, n)` — auto-regressive sentence generation
  - `set_layer(n)` / `list_layers()` — runtime layer switching
  - `info()` — engine diagnostics
  - Context manager support (`with amf.engine(...) as eng:`)
- `amf.engine()` factory function in public API
- Full dequantization support: F32, F16, Q8_0, Q4_K_M
- Sovereign Filter: RegEx-based output sanitization (human words only)
- Validated on Qwen2.5-7B @ Kaggle: RAM < 500 MB ✅

### Changed
- `amf.py` public API restructured — `amf.engine()` is now the primary entry point
- `__version__` bumped to `0.2.0`
- README rewritten with full bilingual documentation (EN/AR)
- `pyproject.toml` keywords and classifiers expanded

### Removed
- Internal prototype identifiers from engine code

---

## [0.1.1] — 2026-03-14

### Added
- First public release on PyPI
- `amf.load_universal()` and `amf.fragment()` API
- GGUF parser, weight analyzer, DNA tagging
- Functional / per-layer / hybrid / per-component fragmentation strategies
- Molecular Engine with LRU cell caching
- CLI: `amf fragment`, `amf chat`, `amf info`
- Tokenizer with GGUF metadata extraction
- Intent analyzer (6 categories, Arabic + English keywords)

---

## [0.1.0] — 2026-03-10

### Added
- Initial internal release
- Core architecture: ModelLoader, SortingAlgorithm, CellManifest
- Basic inference pipeline (NumPy forward pass)
