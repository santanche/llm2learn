# Multimodal: simple image generation with Janus-Pro-1B

Generates simple, non-realistic images (drawings, cartoons, flat illustrations) of
scenarios and characters for virtual patients and interactive digital narratives,
using [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B), a small
unified multimodal model from DeepSeek.

## Why not Ollama?

Ollama does not serve Janus (nor its image-generation mode), so the model runs through
Hugging Face `transformers`, which supports Janus natively
([`deepseek-community/Janus-Pro-1B`](https://huggingface.co/deepseek-community/Janus-Pro-1B)).
Ollama is still used, optionally, to expand a short description into a richer visual
prompt with a local LLM (`--enhance`).

## Setup

```bash
cd multimodal
TMPDIR=~/tmp_pip_cache uv sync
```

Notes:
- PyTorch is the CPU build: Janus-Pro-1B needs ~4 GB in bf16, more than the local GPU has.
  On a machine with a larger GPU, drop the `[tool.uv.sources]` block in `pyproject.toml`.
- `transformers` is pinned to `<5`: Janus image generation is broken in 5.x
  (`_prepare_static_cache()` signature mismatch).
- The first run downloads the model (~4 GB) to `~/.cache/huggingface`.

## Usage

```bash
uv run janus_image.py "an elderly woman with glasses sitting in a doctor's office"
uv run janus_image.py "a nurse smiling" --style cartoon -n 2 --seed 42
uv run janus_image.py "hospital waiting room" --enhance --ollama-model llama3.2:3b
```

| Option | Description |
|---|---|
| `--style` | `drawing` (default), `cartoon`, `flat`, `sketch`, `watercolor`, `pixel`, `none` |
| `-n` | number of images (generated in one batch) |
| `--seed` | fixed seed for reproducible images |
| `--guidance` | classifier-free guidance scale (default 5); higher follows the prompt more closely |
| `--enhance` | rewrite the prompt with an Ollama model first |
| `--out` | output directory (default `output/`) |
| `--float32` | use float32 (~8 GB RAM) instead of bf16 |

Images are 384x384 PNGs. On a laptop CPU (i7-1255U, bf16) one image takes about 13 minutes;
a GPU with 6 GB or more brings this down to seconds.

From Python (for example, inside a virtual patient agent):

```python
from janus_image import JanusImageGenerator

generator = JanusImageGenerator()        # load once
images = generator.generate("a child with a fever lying in bed, cartoon illustration")
images[0].save("patient.png")
```
