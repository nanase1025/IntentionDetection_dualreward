#!/usr/bin/env python3
"""
Convert GRPO (VERL/FSDP) checkpoint to HuggingFace format

Usage:
    python convert_grpo_to_hf.py \
        --grpo_checkpoint work_dirs/ovigod_grpo/global_step_50 \
        --output_dir work_dirs/ovigod_grpo_hf/checkpoint-50
"""

import argparse
import os
import shutil
import torch
from pathlib import Path


def convert_grpo_to_hf(grpo_checkpoint_dir, output_dir):
    """
    Convert GRPO checkpoint to HuggingFace format
    
    GRPO structure:
        global_step_50/
        ├── dataloader.pt
        └── actor/
            ├── model_world_size_1_rank_0.pt
            ├── optim_world_size_1_rank_0.pt  
            ├── extra_state_world_size_1_rank_0.pt
            └── huggingface/  ← Already in HF format!
    
    HuggingFace structure:
        checkpoint-50/
        ├── pytorch_model.bin (or safetensors)
        ├── config.json
        ├── tokenizer files
        └── preprocessor_config.json
    """
    grpo_checkpoint_dir = Path(grpo_checkpoint_dir)
    output_dir = Path(output_dir)
    
    print(f"Converting GRPO checkpoint: {grpo_checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input exists
    if not grpo_checkpoint_dir.exists():
        raise FileNotFoundError(f"GRPO checkpoint not found: {grpo_checkpoint_dir}")
    
    actor_dir = grpo_checkpoint_dir / "actor"
    if not actor_dir.exists():
        raise FileNotFoundError(f"Actor directory not found: {actor_dir}")
    
    hf_dir = actor_dir / "huggingface"
    if not hf_dir.exists():
        raise FileNotFoundError(f"HuggingFace directory not found: {hf_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Step 1: Copying HuggingFace config files...")
    # Copy all HuggingFace config files
    hf_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "chat_template.json",
    ]
    
    for file in hf_files:
        src = hf_dir / file
        if src.exists():
            shutil.copy2(src, output_dir / file)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠ Skipped {file} (not found)")
    
    print("\n🔄 Step 2: Converting FSDP model to HuggingFace format...")
    # Load FSDP model weights
    model_files = list(actor_dir.glob("model_world_size_*.pt"))
    if not model_files:
        raise FileNotFoundError("No FSDP model files found")
    
    print(f"  Found {len(model_files)} model shard(s)")
    
    # For single GPU (world_size=1), we can directly use the model
    if len(model_files) == 1:
        print("  Loading single-GPU FSDP checkpoint...")
        model_state = torch.load(model_files[0], map_location='cpu', weights_only=False)
        
        # FSDP saves state dict, we need to extract model weights
        # The state dict might have different keys depending on FSDP configuration
        if isinstance(model_state, dict):
            # Check if it's already in standard format
            if any(k.startswith('model.') for k in model_state.keys()):
                # Remove 'model.' prefix if exists
                clean_state = {}
                for k, v in model_state.items():
                    if k.startswith('model.'):
                        clean_state[k[6:]] = v  # Remove 'model.' prefix
                    else:
                        clean_state[k] = v
                model_state = clean_state
            
            print(f"  Model has {len(model_state)} parameters")
            print(f"  Sample keys: {list(model_state.keys())[:3]}")
            
            # Save in PyTorch format
            output_model_path = output_dir / "pytorch_model.bin"
            print(f"  Saving to {output_model_path}...")
            torch.save(model_state, output_model_path)
            print("  ✓ Model saved successfully")
        else:
            raise ValueError(f"Unexpected model state type: {type(model_state)}")
    else:
        # Multiple GPUs - need to merge shards
        print(f"  ⚠ Multi-GPU checkpoint detected ({len(model_files)} shards)")
        print("  Note: This script currently supports single-GPU checkpoints only")
        print("  For multi-GPU checkpoints, you need to merge shards first")
        raise NotImplementedError("Multi-GPU checkpoint merging not yet implemented")
    
    print("\n✅ Conversion complete!")
    print(f"\n📁 Output checkpoint: {output_dir}")
    print("\n💡 You can now use this checkpoint with:")
    print(f"   python evaluate_ovigod_ap.py --checkpoint {output_dir} --backend vllm")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert GRPO checkpoint to HuggingFace format")
    parser.add_argument(
        "--grpo_checkpoint",
        type=str,
        required=True,
        help="Path to GRPO checkpoint directory (e.g., work_dirs/ovigod_grpo/global_step_50)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace format checkpoint"
    )
    
    args = parser.parse_args()
    
    try:
        convert_grpo_to_hf(args.grpo_checkpoint, args.output_dir)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

