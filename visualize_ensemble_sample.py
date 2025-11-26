#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize ensemble process for a single sample

Usage:
    python visualize_ensemble_sample.py \
        --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-627 \
        --image_path data/ov-igod-dataset/sunrgbd_jpgs/1.jpg \
        --category "I want to sit" \
        --n_samples 10
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rex_omni import RexOmniWrapper


def visualize_ensemble(
    image_path,
    categories,
    model,
    n_samples=10,
    iou_threshold=0.5,
    vote_threshold=0.4,
    temperature=1.0,
    output_path="ensemble_visualization.png"
):
    """
    Visualize the ensemble process
    
    Shows:
    1. All n predictions (different colors/transparency)
    2. Clusters after IoU-based grouping
    3. Final predictions after voting
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Generate n predictions
    print(f"Generating {n_samples} predictions with temperature={temperature}...")
    all_predictions = []
    for i in range(n_samples):
        old_temp = model.temperature
        model.temperature = temperature
        
        results = model.inference(
            images=image, 
            task="detection", 
            categories=categories,
        )
        
        model.temperature = old_temp
        all_predictions.append(results[0]["extracted_predictions"])
        print(f"  Sample {i+1}/{n_samples}: {sum(len(v) for v in all_predictions[i].values())} boxes")
    
    # For visualization, focus on the first category
    category = categories[0]
    
    # Collect all boxes for this category
    all_boxes = []
    for pred in all_predictions:
        if category in pred:
            for box_pred in pred[category]:
                coords = box_pred['coords']
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    all_boxes.append(coords)
    
    print(f"\nTotal boxes for '{category}': {len(all_boxes)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: All raw predictions
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title(f"Panel 1: All {len(all_boxes)} Raw Predictions\n(different samples in different colors)", fontsize=14)
    ax1.axis('off')
    
    # Assign colors to each sample
    colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))
    box_idx = 0
    for sample_idx, pred in enumerate(all_predictions):
        if category in pred:
            for box_pred in pred[category]:
                coords = box_pred['coords']
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    x0, y0, x1, y1 = coords
                    rect = patches.Rectangle(
                        (x0, y0), x1-x0, y1-y0,
                        linewidth=2,
                        edgecolor=colors[sample_idx],
                        facecolor='none',
                        alpha=0.6
                    )
                    ax1.add_patch(rect)
                    box_idx += 1
    
    # Panel 2: Clustered boxes
    from evaluate_ovigod_ap_ensemble import cluster_boxes_by_iou, compute_iou
    
    clusters = cluster_boxes_by_iou(all_boxes, iou_threshold)
    
    ax2 = axes[1]
    ax2.imshow(image)
    ax2.set_title(f"Panel 2: {len(clusters)} Clusters (IoU>{iou_threshold})\n(each cluster = one color)", fontsize=14)
    ax2.axis('off')
    
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
    for cluster_idx, cluster in enumerate(clusters):
        for box_idx in cluster:
            x0, y0, x1, y1 = all_boxes[box_idx]
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=2,
                edgecolor=cluster_colors[cluster_idx],
                facecolor='none',
                alpha=0.7
            )
            ax2.add_patch(rect)
        
        # Draw cluster center (mean box)
        cluster_boxes = [all_boxes[i] for i in cluster]
        mean_box = np.mean(cluster_boxes, axis=0)
        x0, y0, x1, y1 = mean_box
        rect = patches.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            linewidth=4,
            edgecolor=cluster_colors[cluster_idx],
            facecolor='none',
            linestyle='--'
        )
        ax2.add_patch(rect)
        
        # Add vote count text
        vote_count = len(cluster)
        vote_ratio = vote_count / n_samples
        ax2.text(
            x0, y0-5, 
            f"{vote_count}/{n_samples} ({vote_ratio:.1%})",
            color=cluster_colors[cluster_idx],
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Panel 3: Final predictions after voting
    ax3 = axes[2]
    ax3.imshow(image)
    ax3.set_title(f"Panel 3: Final Predictions (vote>{vote_threshold})\n(thick boxes = high confidence)", fontsize=14)
    ax3.axis('off')
    
    for cluster_idx, cluster in enumerate(clusters):
        vote_count = len(cluster)
        vote_ratio = vote_count / n_samples
        
        if vote_ratio >= vote_threshold:
            # Compute mean box
            cluster_boxes = [all_boxes[i] for i in cluster]
            mean_box = np.mean(cluster_boxes, axis=0)
            x0, y0, x1, y1 = mean_box
            
            # Box thickness based on confidence
            linewidth = 2 + 6 * vote_ratio
            
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=linewidth,
                edgecolor='lime',
                facecolor='none',
                alpha=0.9
            )
            ax3.add_patch(rect)
            
            # Add confidence text
            ax3.text(
                x0, y0-5,
                f"conf={vote_ratio:.2f}",
                color='lime',
                fontsize=12,
                weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total raw boxes:      {len(all_boxes)}")
    print(f"  Number of clusters:   {len(clusters)}")
    
    filtered_count = sum(1 for c in clusters if len(c)/n_samples >= vote_threshold)
    print(f"  After voting:         {filtered_count} boxes")
    print(f"  Filtering ratio:      {filtered_count/len(clusters)*100:.1f}%")
    
    print(f"\nCluster details:")
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True), 1):
        vote_ratio = len(cluster) / n_samples
        status = "✓ KEPT" if vote_ratio >= vote_threshold else "✗ FILTERED"
        print(f"  Cluster {i:2d}: {len(cluster):2d}/{n_samples} votes ({vote_ratio:5.1%}) {status}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ensemble process for a single sample")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--image_path", type=str, required=True, help="Path to test image")
    parser.add_argument("--category", type=str, required=True, help="Category to detect")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--vote_threshold", type=float, default=0.4, help="Vote threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for clustering")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output", type=str, default="ensemble_visualization.png", help="Output path")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"])
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
        max_tokens=2048,
    )
    
    visualize_ensemble(
        image_path=args.image_path,
        categories=[args.category],
        model=model,
        n_samples=args.n_samples,
        iou_threshold=args.iou_threshold,
        vote_threshold=args.vote_threshold,
        temperature=args.temperature,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

