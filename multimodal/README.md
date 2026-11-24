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

## Batch generation from a CSV

`janus_batch.py` generates one image per line of a CSV with the columns `id,prompt`,
recording each finished image in a progress CSV (`id,prompt,filename,seed,seconds`).
Press any key to stop at any moment (the image in progress is discarded); running the
same command again resumes with the ids not yet in the progress CSV.

```bash
uv run janus_batch.py prompts/health_prompts.csv --out output/health
```

| Option | Description |
|---|---|
| `--out` | output directory (default `output/batch/`) |
| `--progress` | progress CSV (default `<out>/progress.csv`) |
| `--seed` | seed used for every prompt (default 42), so that directives can be compared |
| `--random-seed` | do not fix the seed |
| `--guidance`, `--temperature`, `--device`, `--float32` | as in `janus_image.py` |

`prompts/health_prompts.csv` is an essay of 130 prompts in the health context:

| ids | group |
|---|---|
| 1–23 | directive study: the same physician with different style directives (colorful drawing, clean lines, white/transparent background, line art, cartoon, flat vector, watercolor, pixel art, sticker, framing, negative directives) |
| 24–33 | directive study: the same emergency room with different styles and viewpoints |
| 34–56 | health scenarios: hospital exterior, reception, waiting room, triage, office, ward, ICU, operating room, pharmacy, lab, ambulance, home visit, rural post, etc. |
| 57–68 | health professionals |
| 69–94 | patients in clinical conditions (chest pain, dyspnea, fever, asthma, stroke, jaundice, edema, etc.) |
| 95–98 | the same patient with different expressions (character consistency) |
| 99–106 | clinical interactions (consultation, blood pressure, CPR, telemedicine, vaccination) |
| 107–130 | objects, icons, diagrams, pain scale, body outline, text in images, infographics |
