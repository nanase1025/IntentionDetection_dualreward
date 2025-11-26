#!/bin/bash

echo "Killing all GRPO/Ray/vLLM related processes..."

# Kill Ray processes
pkill -9 -f "ray::" || true
pkill -9 -f "ray start" || true
pkill -9 -f "ray.util" || true

# Kill Python processes related to VERL/GRPO
pkill -9 -f "verl.trainer.main" || true
pkill -9 -f "vllm" || true

# Kill any remaining Python processes on GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true

# Clean up Ray temp files
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true

# Wait a moment for cleanup
sleep 3

echo "Cleanup complete. Checking remaining GPU processes..."
nvidia-smi

echo "Done! You can now restart training."
