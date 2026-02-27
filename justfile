# Default: run lint + tests
default: check

# Run tests
test *args:
    uv run --extra dev pytest tests/ {{args}}

# Lint with ruff
lint:
    uv run --extra dev ruff check .

# Format with ruff
fmt:
    uv run --extra dev ruff format .

# Check formatting without changing files
fmt-check:
    uv run --extra dev ruff format --check .

# Lint + format check + tests
check:
    uv run --extra dev ruff check .
    uv run --extra dev ruff format --check .
    uv run --extra dev pytest tests/

# Auto-fix lint issues + format
fix:
    uv run --extra dev ruff check --fix .
    uv run --extra dev ruff format .

# Run captioning with mock backend (validation only)
dry-run *args:
    uv run python caption.py run --backend mock --dry-run {{args}}

# Show captions.jsonl stats
stats:
    uv run python caption.py stats

# Estimate cost for an API backend
estimate backend="openai" model="gpt-4o-mini":
    uv run python caption.py estimate --backend {{backend}} --model {{model}}
