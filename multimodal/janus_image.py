"""
Image generation from a text prompt with Janus-Pro-1B (a small multimodal model).

Aimed at simple, non-realistic pictures (drawings, cartoons, flat illustrations)
of scenarios and characters for virtual patients and interactive narratives.

Janus is not served by Ollama, so it runs through Hugging Face transformers.
Ollama is used optionally (--enhance) to have a small local LLM expand a short
scene description into a richer visual prompt before generating the image.

Usage examples:
    uv run janus_image.py "an elderly man with a cane in a doctor's office"
    uv run janus_image.py "a nurse smiling" --style cartoon -n 2 --seed 42
    uv run janus_image.py "hospital waiting room" --enhance --ollama-model llama3.2:3b
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import torch
from transformers import JanusForConditionalGeneration, JanusProcessor

MODEL_ID = "deepseek-community/Janus-Pro-1B"   # transformers-native conversion of deepseek-ai/Janus-Pro-1B

# Style suffixes that steer Janus away from photorealism
STYLES = {
    "drawing":    "simple colored drawing, clean lines, white background",
    "cartoon":    "cartoon illustration, flat colors, bold outlines, friendly style",
    "flat":       "flat vector illustration, minimalist, soft pastel colors",
    "sketch":     "pencil sketch, black and white, hand-drawn",
    "watercolor": "watercolor painting, soft colors, storybook illustration",
    "pixel":      "pixel art, 16-bit video game style",
    "none":       "",
}


def enhance_prompt(description, model="llama3.2:3b", model_url="http://localhost:11434/api/generate"):
    """Use a local LLM (Ollama) to turn a short description into a visual prompt"""
    instruction = (
        "Rewrite the following description as a single short prompt for an image generator. "
        "Describe only what is visible: characters, clothing, expressions, objects, setting. "
        "Use at most 40 words, no preamble, no quotes.\n\n"
        f"Description: {description}\n\nPrompt:"
    )
    try:
        response = requests.post(model_url, json={
            "model": model,
            "prompt": instruction,
            "stream": False,
            "options": {"temperature": 0.3}
        }, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip().strip('"')
    except requests.RequestException as e:
        print(f"Ollama unavailable ({e}); using the original description.")
        return description


def pick_device():
    """Use the GPU only if it has enough memory for the model (~4 GB in bf16)"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total >= 6:
            return "cuda"
    return "cpu"


class JanusImageGenerator:
    """Loads Janus-Pro-1B once and generates images from text prompts"""

    def __init__(self, model_id=MODEL_ID, device=None, dtype=torch.bfloat16):
        self.device = device or pick_device()
        self.dtype = dtype
        print(f"Loading {model_id} on {self.device} ({dtype})...")
        self.processor = JanusProcessor.from_pretrained(model_id)
        self.model = JanusForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype
        ).to(self.device).eval()

    def generate(self, prompt, num_images=1, guidance_scale=5.0, temperature=1.0, seed=None):
        """Returns a list of PIL images (384x384)"""
        if seed is not None:
            torch.manual_seed(seed)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=chat, generation_mode="image", return_tensors="pt"
        ).to(self.device, dtype=self.dtype)

        tokens = self.model.generate(
            **inputs,
            generation_mode="image",
            do_sample=True,
            temperature=temperature,
            guidance_scale=guidance_scale,
            num_return_sequences=num_images,
        )
        decoded = self.model.decode_image_tokens(tokens)
        images = self.processor.postprocess(list(decoded.float()), return_tensors="PIL.Image.Image")
        return images["pixel_values"]


def slugify(text, max_len=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len]


def main():
    parser = argparse.ArgumentParser(description="Generate simple images with Janus-Pro-1B")
    parser.add_argument("prompt", help="description of the scene or character")
    parser.add_argument("--style", choices=STYLES, default="drawing", help="visual style (default: drawing)")
    parser.add_argument("-n", "--num-images", type=int, default=1)
    parser.add_argument("--seed", type=int, help="random seed for reproducible images")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG scale; higher follows the prompt more closely")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default="output", help="output directory (default: output/)")
    parser.add_argument("--enhance", action="store_true", help="expand the prompt with an Ollama model first")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="force a device (default: auto)")
    parser.add_argument("--float32", action="store_true", help="use float32 (needs ~8 GB RAM; may be faster on some CPUs)")
    args = parser.parse_args()

    description = args.prompt
    if args.enhance:
        description = enhance_prompt(description, model=args.ollama_model)
        print(f"Enhanced prompt: {description}")

    style = STYLES[args.style]
    prompt = f"{description}, {style}" if style else description
    print(f"Prompt: {prompt}")

    generator = JanusImageGenerator(
        device=args.device,
        dtype=torch.float32 if args.float32 else torch.bfloat16,
    )

    start = time.time()
    images = generator.generate(
        prompt,
        num_images=args.num_images,
        guidance_scale=args.guidance,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"Generated {len(images)} image(s) in {time.time() - start:.0f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i, image in enumerate(images):
        path = out_dir / f"{stamp}-{slugify(args.prompt)}-{i}.png"
        image.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
