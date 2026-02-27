# Self-Hosted Model Reference

Vision-language models suitable for local inference on consumer GPUs.

## Model Comparison

| Model | Released | FP16 VRAM | INT4 VRAM | RTX 3070 (8GB)? | Speed Est. | HF ID |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL-3B** | Jan 2025 | 7-8 GB | **3-4 GB** | Yes (int4) | ~1-2 img/s | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **Qwen2.5-VL-7B** | Jan 2025 | 17 GB | **6-8 GB** | Tight | ~0.5-1 img/s | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Moondream2 | Jun 2025 | 5-10 GB | 4 GB | Yes | ~1-2 img/s | `vikhyatk/moondream2` |
| Florence-2-large | Jun 2024 | 1.2-2 GB | <1 GB | Yes | **5-10 img/s** | `microsoft/Florence-2-large` |
| SmolVLM2-2.2B | Feb 2025 | ~4 GB | <1 GB | Yes | **5-20 img/s** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| InternVL2-8B | Jul 2024 | 16 GB | 5-6 GB | Marginal | ~1 img/s | `OpenGVLab/InternVL2-8B` |

## Recommendations

- **RTX 3070 (8 GB)**: `Qwen/Qwen2.5-VL-3B-Instruct` in INT4 — best quality/VRAM tradeoff.
  44k photos at ~1-2 img/s = 6-12 hours overnight, fully resumable.
- **RTX 3090/4090 (24 GB)**: `Qwen/Qwen2.5-VL-7B-Instruct` in FP16 for better quality.

## Usage

```bash
uv run --extra local python caption.py run \
    --backend local \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --photos-dir ~/icloud_photos
```

The local backend automatically loads in 4-bit quantization via bitsandbytes.
