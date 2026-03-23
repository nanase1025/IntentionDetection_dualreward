#!/usr/bin/env python3
"""Test if the SFT checkpoint can be loaded correctly"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_path = "/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft/checkpoint-2523"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("Processor loaded successfully!")

print("\nLoading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
print("Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

