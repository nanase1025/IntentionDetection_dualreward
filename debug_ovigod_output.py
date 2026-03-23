#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script: check what the model actually outputs for OV-IGOD affordance queries.
Run this in your conda env to see raw model outputs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_ovigod_output.py \
        --checkpoint finetuning/work_dirs/intention_grpo_iou_lt_06/global_step_828/actor/huggingface
"""

import argparse
import json
from PIL import Image
from rex_omni import RexOmniWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backend", type=str, default="vllm")
    args = parser.parse_args()

    rex = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    image_path = "/home/hairong/hairong/code/ov-igod-dataset/sunrgbd_jpgs/7.jpg"
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    print("\n" + "=" * 80)
    print("TEST 1: Short category (normal detection)")
    print("=" * 80)
    r1 = rex.inference(images=image, task="detection", categories=["bed"])
    print(f"Prompt:     {r1[0]['prompt']}")
    print(f"Raw output: {repr(r1[0]['raw_output'])}")
    print(f"Extracted:  {r1[0]['extracted_predictions']}")

    print("\n" + "=" * 80)
    print("TEST 2: Long affordance (OV-IGOD style)")
    print("=" * 80)
    aff = "I long for a comfortable place to rest my tired body and get a good night's sleep"
    r2 = rex.inference(images=image, task="detection", categories=[aff])
    print(f"Prompt:     {r2[0]['prompt']}")
    print(f"Raw output: {repr(r2[0]['raw_output'])}")
    print(f"Extracted:  {r2[0]['extracted_predictions']}")

    print("\n" + "=" * 80)
    print("TEST 3: Medium affordance (intention-dataset style)")
    print("=" * 80)
    r3 = rex.inference(images=image, task="detection", categories=["a comfortable place to rest"])
    print(f"Prompt:     {r3[0]['prompt']}")
    print(f"Raw output: {repr(r3[0]['raw_output'])}")
    print(f"Extracted:  {r3[0]['extracted_predictions']}")

    print("\n" + "=" * 80)
    print("TEST 4: Multiple short categories at once")
    print("=" * 80)
    r4 = rex.inference(images=image, task="detection", categories=["bed", "dresser", "lamp"])
    print(f"Prompt:     {r4[0]['prompt']}")
    print(f"Raw output: {repr(r4[0]['raw_output'])}")
    print(f"Extracted:  {r4[0]['extracted_predictions']}")


if __name__ == "__main__":
    main()
