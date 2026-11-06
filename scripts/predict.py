#!/usr/bin/env python3
"""Predict where the camera should move next, from one image and intent.

  python scripts/predict.py --image room.jpg --intent "Find my bike"

Writes JSON to stdout, or to --out. Progress goes to stderr. Repeat --intent to run
several on the same image, which gives a JSON list. See `lime_camera.pose` for the
pose convention.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from lime_camera import LIME
from lime_camera.models.lime import DTYPES

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "lime_v1"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=WEIGHTS,
                    help="Checkpoint folder. Default: weights/lime_v1, where "
                         "scripts/download_weights.sh puts the released one.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--intent", action="append", required=True,
                    help="What the camera should do. Repeat for several.")
    ap.add_argument("--samples", type=int, default=5, help="Pose samples K.")
    ap.add_argument("--steps", type=int, default=10, help="Euler steps per sample.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Fixes the sampling noise. Default: random.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    ap.add_argument("--attn", default="sdpa",
                    help="sdpa (default, no extra deps) or flash_attention_2")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON here.")
    args = ap.parse_args()

    if not args.image.is_file():
        ap.error(f"no such image: {args.image}")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[%(name)s] %(message)s")
    try:
        model = LIME.from_pretrained(args.checkpoint, args.dtype, args.attn)
    except FileNotFoundError as e:
        ap.error(str(e))

    results = []
    for intent in args.intent:
        prediction = model.predict(
            args.image, intent, n_samples=args.samples, n_steps=args.steps,
            seed=args.seed, max_new_tokens=args.max_new_tokens)
        move = np.linalg.norm(prediction.pose9_mean[:3])
        spread = np.linalg.norm(prediction.pose9_std[:3])
        print(f"{intent!r}: move {move:.2f} m, spread {spread:.2f} m\n"
              f"  {prediction.observation_gain}", file=sys.stderr)
        results.append({"image": str(args.image), **prediction.to_dict()})

    text = json.dumps(results[0] if len(results) == 1 else results, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
