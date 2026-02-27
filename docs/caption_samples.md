# Caption Samples

Side-by-side comparison of captions from each backend on representative test photos.
Run your own comparison with `--limit 5` before committing to a full run.

## Test Photos

| # | Description | Filename |
|---|---|---|
| 1 | Nature / landscape | `landscape.jpg` |
| 2 | Group portrait | `group.jpg` |
| 3 | Food / restaurant | `food.jpg` |
| 4 | Screenshot / text | `screenshot.png` |
| 5 | Pet / animal | `pet.jpg` |

## Sample Captions

*Run the following to generate your own comparison:*

```bash
# Local model
uv run --extra local python caption.py run --backend local \
    --model Qwen/Qwen2.5-VL-3B-Instruct --limit 5

# OpenAI
uv run --extra api python caption.py run --backend openai \
    --model gpt-4o-mini --limit 5

# Anthropic
uv run --extra api python caption.py run --backend anthropic \
    --model claude-haiku-4-5 --limit 5

# Gemini
uv run --extra api python caption.py run --backend gemini \
    --model gemini-2.0-flash --limit 5
```

| Photo | Local (Qwen2.5-VL-3B) | OpenAI (gpt-4o-mini) | Anthropic (claude-haiku) | Gemini (2.0-flash) |
|---|---|---|---|---|
| 1. Landscape | *(run to fill)* | *(run to fill)* | *(run to fill)* | *(run to fill)* |
| 2. Group | *(run to fill)* | *(run to fill)* | *(run to fill)* | *(run to fill)* |
| 3. Food | *(run to fill)* | *(run to fill)* | *(run to fill)* | *(run to fill)* |
| 4. Screenshot | *(run to fill)* | *(run to fill)* | *(run to fill)* | *(run to fill)* |
| 5. Pet | *(run to fill)* | *(run to fill)* | *(run to fill)* | *(run to fill)* |

## Notes

- Local models tend to be terser but adequate for clustering
- API models produce richer descriptions with more detail
- For clustering purposes, even short captions significantly outperform vocabulary-based labels
