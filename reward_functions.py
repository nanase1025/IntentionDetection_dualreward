"""
Dual Reward Functions for Intention Detection GRPO
Combines geometric accuracy (IoU) with semantic relevance (CLIP)
"""

import torch
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import os


class RewardComputer:
    """Computes rewards for intention detection predictions"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        """
        Args:
            clip_model_name: HuggingFace model name for CLIP
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        Args:
            box1, box2: [x0, y0, x1, y1] format
        Returns:
            iou: float
        """
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter = 0
            
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        
        return float(inter) / union if union > 0 else 0.0
    
    def compute_clip_score(self, image, bbox, text_query):
        """
        Compute CLIP similarity between cropped image region and text
        Args:
            image: PIL Image
            bbox: [x0, y0, x1, y1] predicted bounding box
            text_query: str, intention query
        Returns:
            score: float, CLIP similarity score
        """
        # Crop the predicted region
        try:
            cropped = image.crop((int(bbox[0]), int(bbox[1]), 
                                 int(bbox[2]), int(bbox[3])))
        except Exception as e:
            print(f"Warning: Failed to crop image with bbox {bbox}: {e}")
            return 0.0
        
        # Compute CLIP score
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[text_query],
                images=cropped,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.clip_model(**inputs)
            # Get similarity score (logits_per_image)
            score = outputs.logits_per_image[0, 0].item()
            
        return score


def iou_reward(completions, solutions, **kwargs):
    """
    IoU-based reward function
    Args:
        completions: list of model outputs
        solutions: list of ground truth bboxes [x0, y0, x1, y1]
        **kwargs: additional arguments (not used)
    Returns:
        rewards: list of float (1.0 if IoU > 0.5, else 0.0)
    """
    reward_computer = kwargs.get('reward_computer', None)
    if reward_computer is None:
        reward_computer = RewardComputer()
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    for content, gt_bbox in zip(contents, solutions):
        reward = 0.0
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    pred_bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                                int(bbox_match.group(3)), int(bbox_match.group(4))]
                    iou = reward_computer.compute_iou(pred_bbox, gt_bbox)
                    if iou > 0.5:
                        reward = 1.0
        except Exception as e:
            pass
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_debug.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"GT BBox: {gt_bbox}\n")
    
    return rewards


def clip_reward(completions, solutions, **kwargs):
    """
    CLIP-based semantic reward function
    Args:
        completions: list of model outputs
        solutions: list of dict with 'bbox' and 'intention_query'
        **kwargs: must include 'images' (list of PIL Images) and 'intention_queries'
    Returns:
        rewards: list of float (1.0 if CLIP score > threshold, else 0.0)
    """
    reward_computer = kwargs.get('reward_computer', None)
    if reward_computer is None:
        reward_computer = RewardComputer()
    
    images = kwargs.get('images', None)
    intention_queries = kwargs.get('intention_queries', None)
    clip_threshold = kwargs.get('clip_threshold', 20.0)  # CLIP score threshold
    
    if images is None or intention_queries is None:
        raise ValueError("Must provide 'images' and 'intention_queries' in kwargs for clip_reward")
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    for content, solution, image, query in zip(contents, solutions, images, intention_queries):
        reward = 0.0
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    pred_bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                                int(bbox_match.group(3)), int(bbox_match.group(4))]
                    
                    # Compute CLIP score
                    clip_score = reward_computer.compute_clip_score(image, pred_bbox, query)
                    
                    if clip_score > clip_threshold:
                        reward = 1.0
        except Exception as e:
            pass
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_debug.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} CLIP reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Query: {query}\n")
                f.write(f"CLIP Score: {clip_score if 'clip_score' in locals() else 'N/A'}\n")
    
    return rewards


def dual_reward(completions, solutions, **kwargs):
    """
    Combined IoU + CLIP reward
    Args:
        completions: list of model outputs
        solutions: list of ground truth bboxes
        **kwargs: must include 'images' and 'intention_queries'
    Returns:
        rewards: list of float (weighted combination of IoU and CLIP)
    """
    alpha = kwargs.get('alpha', 0.5)  # Weight for IoU reward
    beta = kwargs.get('beta', 0.5)    # Weight for CLIP reward
    
    # Compute both rewards
    iou_rewards = iou_reward(completions, solutions, **kwargs)
    clip_rewards = clip_reward(completions, solutions, **kwargs)
    
    # Combine rewards
    # Method 1: Weighted sum
    combined_rewards = [alpha * r_iou + beta * r_clip 
                       for r_iou, r_clip in zip(iou_rewards, clip_rewards)]
    
    # Alternative methods (uncomment to use):
    
    # Method 2: Product (both must be good)
    # combined_rewards = [r_iou * r_clip 
    #                    for r_iou, r_clip in zip(iou_rewards, clip_rewards)]
    
    # Method 3: Conditional (IoU first, then CLIP)
    # combined_rewards = [r_iou + r_clip if r_iou > 0 else 0.0
    #                    for r_iou, r_clip in zip(iou_rewards, clip_rewards)]
    
    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_path = os.getenv("LOG_PATH", "grpo_debug.log")
        with open(log_path, "a") as f:
            f.write(f"------------- {current_time} Dual Reward -------------\n")
            for i, (r_iou, r_clip, r_combined) in enumerate(zip(iou_rewards, clip_rewards, combined_rewards)):
                f.write(f"Sample {i}: IoU={r_iou:.2f}, CLIP={r_clip:.2f}, Combined={r_combined:.2f}\n")
    
    return combined_rewards


# Reward function registry
reward_funcs_registry = {
    "iou": iou_reward,
    "clip": clip_reward,
    "dual": dual_reward,
}
