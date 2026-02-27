# API Provider Reference

Pricing and privacy comparison for cloud vision-language APIs (as of early 2025).

## Cost Estimates (44,000 photos)

| Provider | Model | Est. Cost | Data Retention | ZDR Available? |
|---|---|---|---|---|
| Google Gemini | gemini-1.5-flash | ~$3–6 | 3 years default | Yes |
| Google Gemini | gemini-2.0-flash | ~$6–10 | 3 years default | Yes |
| OpenAI | gpt-4o-mini | ~$6 | 30 days | Yes |
| OpenAI | gpt-4o | ~$19 | 30 days | Yes |
| Anthropic | claude-haiku-4-5 | ~$70 | Never trained on | N/A (safe by default) |
| xAI | grok-2-vision-1212 | ~$44–132 | Private chats not trained | N/A |

## Privacy Notes

All API backends upload raw image bytes to external servers. The `image_loader.py` module strips
EXIF metadata (GPS, device serial numbers) before encoding to JPEG, but the image content itself
is transmitted.

**Recommendations:**
- **Safest**: Anthropic API — data never used for training by default
- **Cheapest + safe**: OpenAI `gpt-4o-mini` API with Zero Data Retention (~$6 total)
- **Avoid**: ChatGPT web UI, Gemini without ZDR enabled
- Always use the **API**, never the web interface, for personal photos

## API Key Setup

Add keys to `config.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
XAI_API_KEY=xai-...
```

Only the key for your chosen backend is required. Missing keys for unused backends are ignored.
