from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation in an isolated process.")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--image-npy")
    parser.add_argument("--object-name")
    parser.add_argument("--mask-npy")
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    return parser.parse_args()


_MODEL_CACHE: dict[str, object] = {}


def _prompt_candidates(object_name: str) -> list[str]:
    name = str(object_name).strip()
    lowered = name.lower()
    aliases = {
        "gray bin": ["gray bin", "bin", "storage bin"],
        "grey bin": ["grey bin", "bin", "storage bin"],
        "bin": ["bin", "storage bin"],
    }
    candidates = aliases.get(lowered, [name])
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _load_model():
    if "model" in _MODEL_CACHE and "processor_cls" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor_cls"]

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = Path(sam3.__file__).resolve().parent
    bpe_path = sam3_root.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["processor_cls"] = Sam3Processor
    return model, Sam3Processor


def run_inference(
    image_npy: str,
    object_name: str,
    mask_npy: str,
    confidence_threshold: float,
) -> int:
    import torch
    from PIL import Image

    image = np.load(image_npy)
    image = np.asarray(image, dtype=np.uint8)
    pil_image = Image.fromarray(image)

    model, Sam3Processor = _load_model()
    for prompt in _prompt_candidates(object_name):
        processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        inference_state = processor.set_image(pil_image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

        scores = inference_state["scores"]
        if scores.shape[0] < 1:
            print(f"SAM3 found no mask for prompt: {prompt}", flush=True)
            continue

        segmap_all = inference_state["masks"]
        idx = torch.argmax(scores).item()
        score = float(scores[idx].detach().cpu().item())
        mask = segmap_all[idx].detach().cpu().numpy()
        print(f"SAM3 selected prompt: {prompt} score={score:.4f}", flush=True)
        mask = np.squeeze(mask).astype(np.uint8)
        np.save(mask_npy, mask)
        return 0

    np.save(mask_npy, np.zeros(image.shape[:2], dtype=np.uint8))
    return 2


def run_server(default_confidence_threshold: float) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        if request.get("cmd") == "shutdown":
            print(json.dumps({"id": request_id, "returncode": 0}), flush=True)
            return 0

        try:
            with contextlib.redirect_stdout(sys.stderr):
                returncode = run_inference(
                    image_npy=request["image_npy"],
                    object_name=request["object_name"],
                    mask_npy=request["mask_npy"],
                    confidence_threshold=float(
                        request.get("confidence_threshold", default_confidence_threshold)
                    ),
                )
            response = {"id": request_id, "returncode": returncode}
        except Exception as exc:
            response = {
                "id": request_id,
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response), flush=True)
    return 0


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    args = parse_args()
    if args.server:
        return run_server(args.confidence_threshold)

    missing = [
        name
        for name, value in (
            ("--image-npy", args.image_npy),
            ("--object-name", args.object_name),
            ("--mask-npy", args.mask_npy),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments outside --server: {', '.join(missing)}")
    return run_inference(
        image_npy=args.image_npy,
        object_name=args.object_name,
        mask_npy=args.mask_npy,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
