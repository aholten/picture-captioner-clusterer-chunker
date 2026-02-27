# The Picture Captioner-Clusterer-Chunker

Generate per-photo captions for a large photo library using pluggable vision-language backends,
then cluster photos by semantic similarity.

## Pipeline

```
caption.py  ──→  captions.jsonl  ──→  cluster_report.py  ──→  cluster_report.csv / .md
```

1. **`caption.py`** walks your photo library, generates a one-sentence caption per image, and
   writes results to an append-only `captions.jsonl` journal. Resumable, parallelizable, and
   backend-agnostic.
2. **`cluster_report.py`** reads the journal, embeds captions with a sentence transformer,
   clusters with UMAP + HDBSCAN, and outputs a CSV + markdown report.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and enter the project
git clone <repo-url>
cd picture-captioner-clusterer-chunker

# Create config.env with your photo library path (required)
echo "PHOTOS_DIR=~/icloud_photos" > config.env

# Optional: add API keys for cloud backends
echo "OPENAI_API_KEY=sk-..." >> config.env
```

## Quick Start

### 1. Validate your photo library (no model needed)

```bash
uv run python caption.py run --backend mock --dry-run
```

Opens every image to check for corrupt files, without loading any model or calling any API.

### 2. Run captioning

**Local GPU (RTX 3070):**
```bash
uv run --extra local python caption.py run \
    --backend local \
    --model Qwen/Qwen2.5-VL-3B-Instruct
```

**OpenAI API:**
```bash
uv run --extra api python caption.py run \
    --backend openai \
    --model gpt-4o-mini \
    --max-workers 8
```

### 3. Check progress

```bash
uv run python caption.py stats
```

### 4. Estimate cost (API backends)

```bash
uv run python caption.py estimate --backend openai --model gpt-4o-mini
```

### 5. Generate cluster report

```bash
uv run --extra report python cluster_report.py
```

Outputs `cluster_export.csv` and `cluster_report.md`.

## Full Production Run

For large libraries (44k+ photos), use `run_loop.sh` which restarts the process every 5,000
images to clear GPU memory:

```bash
bash run_loop.sh local Qwen/Qwen2.5-VL-3B-Instruct
```

Safe to interrupt at any time — re-run to resume.

## Backends

| Backend | `--backend` | Extra Deps | Notes |
|---|---|---|---|
| Mock | `mock` | None | Fake captions for testing |
| Local HF | `local` | `--extra local` | Qwen2.5-VL, bitsandbytes 4-bit |
| OpenAI | `openai` | `--extra api` | gpt-4o-mini, gpt-4o |
| xAI | `xai` | `--extra api` | grok-2-vision |
| Anthropic | `anthropic` | `--extra api` | claude-haiku-4-5 |
| Gemini | `gemini` | `--extra api` | gemini-1.5-flash, gemini-2.0-flash |

See [docs/api_providers.md](docs/api_providers.md) for pricing and privacy details.
See [docs/self_hosted_models.md](docs/self_hosted_models.md) for GPU requirements.
See [docs/adding_a_backend.md](docs/adding_a_backend.md) to add a new backend.

## Resumability

`captions.jsonl` is an append-only journal. Each record is fsynced to disk immediately.
On restart, already-processed photos are skipped automatically.

```bash
# Retry only API errors from a previous run
uv run --extra api python caption.py run --backend openai --retry-status error_api

# Full restart from scratch
uv run python caption.py run --backend mock --force
```

## Testing

```bash
uv run --extra dev pytest tests/ -v
```

## Project Structure

```
├── caption.py              CLI orchestrator (typer)
├── cluster_report.py       Cluster report generator
├── config.py               pydantic-settings config (reads config.env)
├── journal.py              Append-only captions.jsonl journal
├── image_loader.py         HEIC/EXIF/RGB image preprocessing
├── run_loop.sh             Production run wrapper
├── backends/
│   ├── __init__.py         Backend registry + factory
│   ├── base.py             CaptionBackend ABC
│   ├── mock.py             Mock backend (testing)
│   ├── local_hf.py         Local HuggingFace backend
│   ├── openai_api.py       OpenAI + xAI backends
│   ├── anthropic_api.py    Anthropic backend
│   └── gemini_api.py       Google Gemini backend
├── tests/
│   ├── conftest.py         Shared fixtures
│   ├── test_journal.py     Journal unit tests
│   └── test_caption.py     Backend + image loader tests
└── docs/
    ├── api_providers.md    Pricing + privacy reference
    ├── self_hosted_models.md  GPU requirements
    ├── adding_a_backend.md    How to add a new backend
    └── caption_samples.md     Sample output comparison
```
