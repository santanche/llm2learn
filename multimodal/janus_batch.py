"""
Batch image generation with Janus-Pro-1B from a CSV of prompts, with resumable progress.

The input CSV has one prompt per line, with the columns `id` and `prompt`:

    id,prompt
    1,"a physician with a stethoscope, colorful drawing, clean lines"
    2,"a hospital waiting room, flat vector illustration"

Each generated image is saved in the output directory and recorded in a progress CSV
(id, prompt, filename, seed, seconds). Press any key to stop at any moment: the image
in progress is discarded, and the next run resumes with the prompts that are not yet
in the progress CSV.

Usage examples:
    uv run janus_batch.py prompts/health_prompts.csv
    uv run janus_batch.py prompts/health_prompts.csv --out output/health --seed 7
    uv run janus_batch.py prompts/health_prompts.csv --random-seed
"""

import _thread
import argparse
import csv
import os
import select
import sys
import threading
import time
from pathlib import Path

import torch

from janus_image import JanusImageGenerator, slugify

PROGRESS_FIELDS = ["id", "prompt", "filename", "seed", "seconds"]


class KeyListener:
    """Watches the terminal in a background thread; any key press interrupts the main thread"""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.stop = threading.Event()

    def __enter__(self):
        if self.enabled:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)   # read single keys without Enter; Ctrl+C still works
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()
        return self

    def _listen(self):
        while not self.stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.read(1)
                _thread.interrupt_main()   # raises KeyboardInterrupt in the generation loop
                return

    def __exit__(self, *exc):
        if self.enabled:
            import termios
            self.stop.set()
            self.thread.join()
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        return False


def read_prompts(path):
    """Returns a list of (id, prompt) from the input CSV"""
    with open(path, newline="", encoding="utf-8") as f:
        return [(int(row["id"]), row["prompt"].strip())
                for row in csv.DictReader(f) if row.get("prompt", "").strip()]


def read_done_ids(path):
    """Returns the ids already recorded in the progress CSV"""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["id"]) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Generate images with Janus-Pro-1B from a CSV of prompts")
    parser.add_argument("input", help="CSV with the columns id,prompt")
    parser.add_argument("--out", default="output/batch", help="output directory (default: output/batch/)")
    parser.add_argument("--progress", help="progress CSV (default: <out>/progress.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed used for every prompt, so that directives can be compared (default: 42)")
    parser.add_argument("--random-seed", action="store_true", help="do not fix the seed")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG scale; higher follows the prompt more closely")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="force a device (default: auto)")
    parser.add_argument("--float32", action="store_true", help="use float32 (needs ~8 GB RAM)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress) if args.progress else out_dir / "progress.csv"
    seed = None if args.random_seed else args.seed

    prompts = read_prompts(args.input)
    done = read_done_ids(progress_path)
    pending = [(pid, prompt) for pid, prompt in prompts if pid not in done]
    print(f"{len(prompts)} prompts, {len(prompts) - len(pending)} already generated, {len(pending)} pending.")
    if not pending:
        return

    new_file = not progress_path.exists() or progress_path.stat().st_size == 0
    with KeyListener() as keys, open(progress_path, "a", newline="", encoding="utf-8") as progress:
        if keys.enabled:
            print("Press any key to stop (the image in progress is discarded; run again to resume).")
        writer = csv.writer(progress)
        if new_file:
            writer.writerow(PROGRESS_FIELDS)
            progress.flush()

        current = pending[0][0]
        try:
            generator = JanusImageGenerator(
                device=args.device,
                dtype=torch.float32 if args.float32 else torch.bfloat16,
            )
            for n, (pid, prompt) in enumerate(pending, 1):
                current = pid
                print(f"\n[{n}/{len(pending)}] #{pid}: {prompt}")
                start = time.time()
                image = generator.generate(
                    prompt,
                    guidance_scale=args.guidance,
                    temperature=args.temperature,
                    seed=seed,
                )[0]
                seconds = time.time() - start

                # write to a temporary file first, so an interruption never leaves a truncated PNG
                filename = f"{pid:03d}-{slugify(prompt)}.png"
                tmp_path = out_dir / f".{filename}.tmp"
                image.save(tmp_path, format="PNG")
                os.replace(tmp_path, out_dir / filename)

                writer.writerow([pid, prompt, filename, "" if seed is None else seed, f"{seconds:.0f}"])
                progress.flush()
                print(f"Saved {out_dir / filename} ({seconds:.0f}s)")
        except KeyboardInterrupt:
            print(f"\nStopped. Run again to resume from #{current}.")
            return

    print("\nAll prompts generated.")


if __name__ == "__main__":
    main()
