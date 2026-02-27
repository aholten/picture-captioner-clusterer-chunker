# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-26

### Added

- `caption.py` CLI with `run`, `estimate`, and `stats` subcommands
- Append-only `captions.jsonl` journal with resume, retry, and thread-safe writes
- Image loader with HEIC support, EXIF stripping, and RGB normalization
- Backend abstraction (`CaptionBackend` ABC) with pluggable backends:
  - `mock` — fake captions for testing and validation
  - `local` — local HuggingFace models (Qwen2.5-VL) via transformers + bitsandbytes
  - `openai` — OpenAI API (gpt-4o-mini, gpt-4o)
  - `xai` — xAI API (grok-2-vision) via OpenAI SDK
  - `anthropic` — Anthropic API (claude-haiku-4-5)
  - `gemini` — Google Gemini API (gemini-1.5-flash, gemini-2.0-flash)
- `cluster_report.py` for semantic clustering via sentence-transformers + UMAP + HDBSCAN
- `--dry-run` mode to validate entire photo library without inference
- `--restart-every N` for batch restarts (GPU memory management)
- `run_loop.sh` production wrapper
- `config.env`-based configuration via pydantic-settings
- Structured JSON logging to `caption.log`
- Concurrent API requests via `--max-workers`
- Exponential backoff with tenacity for all API backends
- GitHub Actions CI (lint + format + tests)
- Ruff for linting and formatting
- justfile for common dev commands
