import base64
import io
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask


class BaseRewardFunction(ABC):
    """Base reward function abstract class"""

    @abstractmethod
    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """Compute reward score"""
        pass

    @abstractmethod
    def get_reward_name(self) -> str:
        """Get reward function name"""
        pass


class BoxIoURewardFunction(BaseRewardFunction):
    """Bounding box IoU reward function"""

    def get_reward_name(self) -> str:
        return "box_iou"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """Parse ground truth string, extract image dimensions and annotation information"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息
        objects = {}

        for class_name, class_data in answer.items():
            if "boxes" in class_data:
                objects[class_name] = class_data["boxes"]

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的对象"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, boxes_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有边界框坐标
                box_pattern = r"<(\d+)>"
                all_coords = re.findall(box_pattern, boxes_str)

                # 将坐标分组为4个一组 (x1, y1, x2, y2)
                for i in range(0, len(all_coords), 4):
                    if i + 3 < len(all_coords):
                        try:
                            x1, y1, x2, y2 = [
                                int(coord) for coord in all_coords[i : i + 4]
                            ]

                            # 将归一化坐标转换为实际像素坐标
                            x1_abs = x1 / 1000.0 * width
                            y1_abs = y1 / 1000.0 * height
                            x2_abs = x2 / 1000.0 * width
                            y2_abs = y2 / 1000.0 * height

                            # 确保有效的边界框 (x1 < x2, y1 < y2)
                            x1_final = min(x1_abs, x2_abs)
                            y1_final = min(y1_abs, y2_abs)
                            x2_final = max(x1_abs, x2_abs)
                            y2_final = max(y1_abs, y2_abs)

                            # 跳过退化的边界框（零面积）
                            if x1_final < x2_final and y1_final < y2_final:
                                objects[class_name].append(
                                    [x1_final, y1_final, x2_final, y2_final]
                                )
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的交并比"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算边界框IoU奖励分数"""
        # 解析ground truth获取图像尺寸
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            self._log_debug(f"Failed to parse prediction: {predict}")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有边界框
        all_gt_boxes = [
            (box, class_name)
            for class_name, boxes in gt_objects.items()
            for box in boxes
        ]
        all_pred_boxes = [
            (box, class_name)
            for class_name, boxes in pred_objects.items()
            for box in boxes
        ]

        num_gt = len(all_gt_boxes)
        num_pred = len(all_pred_boxes)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Full Rejection and pred is None. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0

        if num_gt == 0 and num_pred != 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(
                        f"Full Rejection and pred is not None. GT: {ground_truth}\n"
                    )
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0
        # 计算Recall
        total_recall_score = 0.0
        for gt_box, gt_class in all_gt_boxes:
            best_iou = 0.0
            for pred_box, pred_class in all_pred_boxes:
                if gt_class == pred_class:
                    iou = self.calculate_iou(gt_box, pred_box)
                    best_iou = max(best_iou, iou)
            total_recall_score += best_iou

        recall = total_recall_score / num_gt if num_gt > 0 else 0.0

        # 计算Precision
        total_precision_score = 0.0
        for pred_box, pred_class in all_pred_boxes:
            best_iou = 0.0
            for gt_box, gt_class in all_gt_boxes:
                if pred_class == gt_class:
                    iou = self.calculate_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
            total_precision_score += best_iou

        precision = total_precision_score / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(f"Prediction: {predict}\n")
                f.write(f"GT: {json.dumps(ground_truth['answer'])}\n")
                # f.write(f"Solution: {ground_truth['answer']}\n\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/box_iou_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="box_iou",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for box_iou: {e}")

        return f1_score


class PointInBoxRewardFunction(BaseRewardFunction):
    """点是否在框内的奖励函数"""

    def get_reward_name(self) -> str:
        return "point_in_box"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """解析ground truth字符串，提取图像尺寸和点框对应信息"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息 - 每个类别同时有points和boxes
        objects = {}

        for class_name, class_data in answer.items():
            if "points" in class_data and "boxes" in class_data:
                points = class_data["points"]
                boxes = class_data["boxes"]
                # 确保points和boxes长度一致
                if len(points) == len(boxes):
                    objects[class_name] = {"points": points, "boxes": boxes}

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的点和类别"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, points_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有点坐标
                point_pattern = r"<(\d+)>"
                all_coords = re.findall(point_pattern, points_str)

                # 将坐标分组为2个一组 (x, y)
                for i in range(0, len(all_coords), 2):
                    if i + 1 < len(all_coords):
                        try:
                            x, y = [int(coord) for coord in all_coords[i : i + 2]]

                            # 将归一化坐标转换为实际像素坐标
                            x_abs = x / 1000.0 * width
                            y_abs = y / 1000.0 * height

                            objects[class_name].append([x_abs, y_abs])
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def is_point_in_box(self, point: List[float], box: List[float]) -> bool:
        """判断点是否在边界框内"""
        x, y = point
        x0, y0, x1, y1 = box

        # 确保边界框坐标顺序正确
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        return x_min <= x <= x_max and y_min <= y <= y_max

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算点是否在框内的奖励分数"""
        # 解析ground truth获取图像尺寸和点框对应信息
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            self._log_debug(f"Failed to parse ground truth: {ground_truth}")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有ground truth的点框对和类别信息
        all_gt_point_box_pairs = []
        for class_name, class_data in gt_objects.items():
            points = class_data["points"]
            boxes = class_data["boxes"]
            for point, box in zip(points, boxes):
                all_gt_point_box_pairs.append((point, box, class_name))

        # 收集所有预测的点
        all_pred_points = []
        for class_name, points in pred_objects.items():
            for point in points:
                all_pred_points.append((point, class_name))

        num_gt = len(all_gt_point_box_pairs)
        num_pred = len(all_pred_points)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt and pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0
        if num_gt == 0 or num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 0.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt or pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        # 贪心匹配：对于每个GT框，寻找一个点和他匹配
        # 一旦匹配上，该GT框和预测点就不要再参与匹配
        matched_gt_indices = set()
        matched_pred_indices = set()

        # 为每个GT框寻找最佳匹配的预测点
        for gt_idx, (gt_point, gt_box, gt_class) in enumerate(all_gt_point_box_pairs):
            if gt_idx in matched_gt_indices:
                continue

            best_match_score = 0.0
            best_match_idx = -1

            for pred_idx, (pred_point, pred_class) in enumerate(all_pred_points):
                if pred_idx in matched_pred_indices:
                    continue  # 这个预测点已经被匹配了

                # 判断预测点是否在GT框内
                if self.is_point_in_box(pred_point, gt_box):
                    # 如果类别一致，reward为1，否则为0
                    match_score = 1.0 if gt_class == pred_class else 0.0
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = pred_idx

            # 如果找到匹配，标记为已匹配
            if best_match_idx >= 0:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_match_idx)

        # 计算Recall: 匹配成功的GT框数量 / 总GT数量
        recall = len(matched_gt_indices) / num_gt if num_gt > 0 else 0.0

        # 计算Precision: 匹配成功的预测点数量 / 总预测数量
        precision = len(matched_pred_indices) / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(
                    f"Matched GT: {len(matched_gt_indices)}, Total GT: {num_gt}, Matched Pred: {len(matched_pred_indices)}, Total Pred: {num_pred}\n\n"
                )
                f.write(f"Prediction: {predict}\n")
                f.write(f"GT: {json.dumps(ground_truth['answer'])}\n")
                # f.write(f"Solution: {ground_truth['answer']}\n\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/point_in_box_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="point_in_box",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for point_in_box: {e}")

        return f1_score


class PointInMaskRewardFunction(BaseRewardFunction):
    """点是否在mask内的奖励函数"""

    def get_reward_name(self) -> str:
        return "point_in_mask"

    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """解析ground truth字符串，提取图像尺寸和点mask对应信息"""
        answer = ground_truth["answer"]

        # 使用resized_image_size进行坐标转换
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size

        # 提取对象信息 - 每个类别同时有points和masks
        objects = {}

        for class_name, class_data in answer.items():
            if "points" in class_data and "masks" in class_data:
                points = class_data["points"]
                masks = class_data["masks"]
                # 确保points和masks长度一致
                if len(points) == len(masks):
                    objects[class_name] = {"points": points, "masks": masks}

        return {"dims": (width, height), "objects": objects, "raw_data": answer}

    def parse_detection_output(
        self, text: str, width: int, height: int
    ) -> Optional[Dict]:
        """解析模型输出字符串，提取检测到的点和类别"""
        try:
            text = text.replace("\n", "").strip()

            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )

            matches = re.findall(pattern, text)

            for class_name, points_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []

                # 查找所有点坐标
                point_pattern = r"<(\d+)>"
                all_coords = re.findall(point_pattern, points_str)

                # 将坐标分组为2个一组 (x, y)
                for i in range(0, len(all_coords), 2):
                    if i + 1 < len(all_coords):
                        try:
                            x, y = [int(coord) for coord in all_coords[i : i + 2]]

                            # 将归一化坐标转换为实际像素坐标
                            x_abs = x / 1000.0 * width
                            y_abs = y / 1000.0 * height

                            objects[class_name].append([x_abs, y_abs])
                        except (ValueError, IndexError):
                            continue

            return {"objects": objects}
        except Exception:
            return None

    def is_point_in_mask(
        self, point: List[float], mask: Union[dict, list], height: int, width: int
    ) -> bool:
        """判断点是否在mask内"""
        try:
            x, y = point

            # 确保坐标在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                return False

            # 解码mask
            if isinstance(mask, dict) and "counts" in mask:
                # RLE format

                binary_mask = coco_mask.decode(mask)
            elif isinstance(mask, list):
                # Already decoded mask
                binary_mask = np.array(mask)
            else:
                return False

            # 检查点是否在mask内
            if (
                binary_mask is not None
                and binary_mask.shape[0] == height
                and binary_mask.shape[1] == width
            ):
                return bool(binary_mask[int(y), int(x)])

            return False
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                self._log_debug(f"Failed to check point in mask: {e}")
            return False

    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """计算点是否在mask内的奖励分数"""
        # 解析ground truth获取图像尺寸和点mask对应信息
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"Fail to parse ground truth: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        width, height = gt_data["dims"]

        # 使用ground truth的尺寸解析预测
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            self._log_debug(f"Failed to parse prediction: {predict}")
            return 0.0

        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]

        # 收集所有ground truth的点mask对和类别信息
        all_gt_point_mask_pairs = []
        for class_name, class_data in gt_objects.items():
            points = class_data["points"]
            masks = class_data["masks"]
            for point, mask in zip(points, masks):
                all_gt_point_mask_pairs.append((point, mask, class_name))

        # 收集所有预测的点
        all_pred_points = []
        for class_name, points in pred_objects.items():
            for point in points:
                all_pred_points.append((point, class_name))

        num_gt = len(all_gt_point_mask_pairs)
        num_pred = len(all_pred_points)

        if num_gt == 0 and num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 1.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt and pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 1.0
        if num_gt == 0 or num_pred == 0:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- Reward: {ground_truth['reward_name']}: 0.0 | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                    )
                    f.write(f"No gt or pred. GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0

        # 贪心匹配：对于每个GT mask，寻找一个点和他匹配
        # 一旦匹配上，该GT mask和预测点就不要再参与匹配
        matched_gt_indices = set()
        matched_pred_indices = set()

        # 为每个GT mask寻找最佳匹配的预测点
        for gt_idx, (gt_point, gt_mask, gt_class) in enumerate(all_gt_point_mask_pairs):
            if gt_idx in matched_gt_indices:
                continue

            best_match_score = 0.0
            best_match_idx = -1

            for pred_idx, (pred_point, pred_class) in enumerate(all_pred_points):
                if pred_idx in matched_pred_indices:
                    continue  # 这个预测点已经被匹配了

                # 判断预测点是否在GT mask内
                if self.is_point_in_mask(pred_point, gt_mask, height, width):
                    # 如果类别一致，reward为1，否则为0
                    match_score = 1.0 if gt_class == pred_class else 0.0
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = pred_idx

            # 如果找到匹配，标记为已匹配
            if best_match_idx >= 0:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_match_idx)

        # 计算Recall: 匹配成功的GT mask数量 / 总GT数量
        recall = len(matched_gt_indices) / num_gt if num_gt > 0 else 0.0

        # 计算Precision: 匹配成功的预测点数量 / 总预测数量
        precision = len(matched_pred_indices) / num_pred if num_pred > 0 else 0.0

        # 计算F1分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"------------- Reward: {ground_truth['reward_name']}: {f1_score} | Dataset: {ground_truth['dataset_name']} -------------\n\n"
                )
                f.write(f"Precision: {precision}, Recall: {recall}\n\n")
                f.write(
                    f"Matched GT: {len(matched_gt_indices)}, Total GT: {num_gt}, Matched Pred: {len(matched_pred_indices)}, Total Pred: {num_pred}\n\n"
                )
                f.write(f"Prediction: {predict}\n")

        # 可视化功能
        visualize_path = os.getenv("LOG_VISUALIZE_PATH")
        if visualize_path:
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = ground_truth.get("dataset_name", "unknown")
                save_path = (
                    f"{visualize_path}/point_in_mask_{dataset_name}_{current_time}.png"
                )

                if ensure_dir_exists(save_path):
                    create_visualization(
                        gt_data=gt_data,
                        pred_data=pred_data,
                        reward_score=f1_score,
                        reward_name="point_in_mask",
                        dataset_name=dataset_name,
                        save_path=save_path,
                    )
            except Exception as e:
                print(f"Visualization failed for point_in_mask: {e}")

        return f1_score

    def _log_debug(self, message: str):
        """记录调试信息"""
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- {current_time} {self.get_reward_name()} reward -------------\n"
                    )
                    f.write(f"{message}\n\n")


class RewardFunctionFactory:
    """奖励函数工厂类"""

    _reward_functions = {}

    @classmethod
    def register(cls, reward_name: str, reward_class: type):
        """注册奖励函数"""
        cls._reward_functions[reward_name] = reward_class

    @classmethod
    def get_reward_function(cls, reward_name: str) -> Optional[BaseRewardFunction]:
        """获取奖励函数实例"""
        if reward_name in cls._reward_functions:
            return cls._reward_functions[reward_name]()
        return None

    @classmethod
    def get_available_rewards(cls) -> List[str]:
        """获取所有可用的奖励函数名称"""
        return list(cls._reward_functions.keys())


# 注册奖励函数
RewardFunctionFactory.register("box_iou", BoxIoURewardFunction)
RewardFunctionFactory.register("point_in_box", PointInBoxRewardFunction)
RewardFunctionFactory.register("point_in_mask", PointInMaskRewardFunction)


class DualRewardFunction(BaseRewardFunction):
    """
    Dual Reward Function: Combines IoU (position) and CLIP (semantic) rewards
    
    This reward function evaluates predictions based on two criteria:
    1. IoU: Position accuracy (how well the bbox overlaps with ground truth)
    2. CLIP: Semantic relevance (how well the predicted region matches the intention query)
    """
    
    _clip_model = None  # Class variable to share CLIP model across instances
    _clip_processor = None
    
    def __init__(self):
        """Initialize Dual Reward Function and load CLIP model if needed"""
        super().__init__()
        
        # Configuration (can be adjusted via environment variables)
        self.alpha = float(os.getenv("DUAL_REWARD_ALPHA", "0.5"))  # IoU weight
        self.beta = float(os.getenv("DUAL_REWARD_BETA", "0.5"))    # CLIP weight
        self.iou_threshold = float(os.getenv("DUAL_REWARD_IOU_THRESHOLD", "0.5"))
        # FG-CLIP v1 Long Text Mode threshold: 15.0 (empirically tested on COCO outdoor)
        # Balanced accuracy ~62% at this threshold
        self.clip_threshold = float(os.getenv("DUAL_REWARD_CLIP_THRESHOLD", "15.0"))
        
        # Load CLIP model (shared across instances)
        if DualRewardFunction._clip_model is None:
            self._load_clip_model()
    
    def _load_clip_model(self):
        """Load FG-CLIP v1 model (only once, shared across instances)"""
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                AutoModelForCausalLM,  # FG-CLIP needs this instead of AutoModel
            )
            
            # Use FG-CLIP v1 by default
            clip_model_name = os.getenv("CLIP_MODEL_NAME", "qihoo360/fg-clip-large")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Loading FG-CLIP v1 model: {clip_model_name} on {device}")
            
            # Load FG-CLIP v1 model (requires AutoModelForCausalLM due to custom config)
            DualRewardFunction._clip_model = AutoModelForCausalLM.from_pretrained(
                clip_model_name, 
                trust_remote_code=True
            ).to(device)
            
            DualRewardFunction._clip_processor = {
                'tokenizer': AutoTokenizer.from_pretrained(clip_model_name),
                'image_processor': AutoImageProcessor.from_pretrained(clip_model_name),
                'is_fgclip_v1': True
            }
            
            DualRewardFunction._clip_model.eval()
            
            if os.getenv("DEBUG_MODE") == "true":
                print(f"✅ FG-CLIP v1 model loaded successfully")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to load FG-CLIP v1 model: {e}")
            print("   Falling back to pure IoU reward")
            DualRewardFunction._clip_model = None
            DualRewardFunction._clip_processor = None
    
    def get_reward_name(self) -> str:
        return "dual"
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def compute_clip_score(self, image_data: str, bbox: List[float], query: str, width: int, height: int) -> float:
        """
        Compute FG-CLIP v1 similarity score between cropped image region and query text
        
        Args:
            image_data: Base64 encoded image string
            bbox: Predicted bounding box [x0, y0, x1, y1]
            query: Intention query text
            width: Image width
            height: Image height
            
        Returns:
            Similarity score (continuous value)
            - FG-CLIP v1 Long Text Mode: logit_scale.exp() * cosine_similarity
            - Typical range for v1: ~5-25, threshold=15.0 for binary reward
        """
        if DualRewardFunction._clip_model is None or DualRewardFunction._clip_processor is None:
            return -1.0  # Return negative to indicate failure
        
        try:
            import torch
            
            # Decode image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((width, height))
            
            # Crop predicted region
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            
            if x0 >= x1 or y0 >= y1:
                return -1.0  # Invalid bbox
            
            cropped = img.crop((x0, y0, x1, y1))
            
            # Check if using FG-CLIP v1
            is_fgclip_v1 = DualRewardFunction._clip_processor.get('is_fgclip_v1', False)
            
            with torch.no_grad():
                if is_fgclip_v1:
                    # FG-CLIP v1 pathway (Long Text Mode)
                    tokenizer = DualRewardFunction._clip_processor['tokenizer']
                    image_processor = DualRewardFunction._clip_processor['image_processor']
                    device = next(DualRewardFunction._clip_model.parameters()).device
                    
                    # FG-CLIP v1 uses fixed resolution (336x336 for large model)
                    image_size = int(os.getenv("FGCLIP_IMAGE_SIZE", "336"))
                    cropped_resized = cropped.resize((image_size, image_size))
                    
                    # Process image with FG-CLIP v1 API
                    image_input = image_processor.preprocess(
                        cropped_resized, 
                        return_tensors='pt'
                    )['pixel_values'].to(device)
                    
                    # Long caption mode: max_length=248, walk_short_pos=False
                    use_long_text = os.getenv("FGCLIP_USE_LONG_TEXT", "true").lower() == "true"
                    if use_long_text:
                        max_length = 248
                        walk_short_pos = False
                    else:
                        max_length = 77
                        walk_short_pos = True
                    
                    caption_input = torch.tensor(
                        tokenizer([query], max_length=max_length, padding="max_length", truncation=True).input_ids,
                        dtype=torch.long,
                        device=device
                    )
                    
                    # Get features
                    image_feature = DualRewardFunction._clip_model.get_image_features(image_input)
                    text_feature = DualRewardFunction._clip_model.get_text_features(
                        caption_input, 
                        walk_short_pos=walk_short_pos
                    )
                    
                    # Normalize
                    image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
                    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
                    
                    # Compute similarity with logit_scale (FG-CLIP v1)
                    logits_per_image = image_feature @ text_feature.T
                    logit_scale = DualRewardFunction._clip_model.logit_scale
                    similarity = (logit_scale.exp() * logits_per_image).squeeze().item()
                    
                else:
                    # Fallback: Should not reach here with current config
                    return -1.0
                
            return similarity
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Warning: CLIP/FG-CLIP computation failed: {e}")
                import traceback
                traceback.print_exc()
            return -1.0
    
    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """Parse ground truth (reuse BoxIoU logic)"""
        answer = ground_truth["answer"]
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size
        
        objects = {}
        for class_name, class_data in answer.items():
            if "boxes" in class_data:
                objects[class_name] = class_data["boxes"]
        
        # Extract image data and intention query
        image_data = ground_truth.get("image", None)
        intention_query = ground_truth.get("intention_query", "")
        
        return {
            "dims": (width, height),
            "objects": objects,
            "raw_data": answer,
            "image": image_data,
            "intention_query": intention_query
        }
    
    def parse_detection_output(self, text: str, width: int, height: int) -> Optional[Dict]:
        """Parse model output (reuse BoxIoU logic)"""
        try:
            text = text.replace("\n", "").strip()
            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )
            
            matches = re.findall(pattern, text)
            
            for class_name, boxes_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []
                
                box_pattern = r"<(\d+)>"
                all_coords = re.findall(box_pattern, boxes_str)
                
                for i in range(0, len(all_coords), 4):
                    if i + 3 < len(all_coords):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in all_coords[i:i+4]]
                            x1_abs = x1 / 1000.0 * width
                            y1_abs = y1 / 1000.0 * height
                            x2_abs = x2 / 1000.0 * width
                            y2_abs = y2 / 1000.0 * height
                            
                            x1_final = min(x1_abs, x2_abs)
                            y1_final = min(y1_abs, y2_abs)
                            x2_final = max(x1_abs, x2_abs)
                            y2_final = max(y1_abs, y2_abs)
                            
                            if x1_final < x2_final and y1_final < y2_final:
                                objects[class_name].append([x1_final, y1_final, x2_final, y2_final])
                        except (ValueError, IndexError):
                            continue
            
            return {"objects": objects}
        except Exception:
            return None
    
    def compute_reward(self, predict: str, ground_truth: str) -> float:
        """
        Compute Dual Reward = alpha * IoU_F1 + beta * CLIP_binary_reward
        
        IoU uses F1 score (continuous 0-1) based on Precision and Recall
        CLIP uses binary reward (1.0 if score > threshold, else 0.0)
        
        Returns:
            Combined reward score (continuous, range depends on alpha/beta)
        """
        # Parse ground truth
        gt_data = self.parse_ground_truth(ground_truth)
        if gt_data is None:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- Dual Reward: Failed to parse GT -------------\n")
                    f.write(f"GT: {ground_truth}\n")
                    f.write(f"Prediction: {predict}\n\n")
            return 0.0
        
        width, height = gt_data["dims"]
        
        # Parse prediction
        pred_data = self.parse_detection_output(predict, width, height)
        if pred_data is None:
            return 0.0
        
        gt_objects = gt_data["objects"]
        pred_objects = pred_data["objects"]
        
        # Collect all boxes
        all_gt_boxes = [(box, class_name) for class_name, boxes in gt_objects.items() for box in boxes]
        all_pred_boxes = [(box, class_name) for class_name, boxes in pred_objects.items() for box in boxes]
        
        num_gt = len(all_gt_boxes)
        num_pred = len(all_pred_boxes)
        
        # Handle edge cases
        if num_gt == 0 and num_pred == 0:
            return 1.0
        if num_gt == 0 and num_pred != 0:
            return 0.0
        if num_pred == 0:
            return 0.0
        
        # ===== Part 1: IoU Reward (F1 Score - same as BoxIoURewardFunction) =====
        # 计算 Recall
        total_recall_score = 0.0
        for gt_box, gt_class in all_gt_boxes:
            best_iou = 0.0
            for pred_box, pred_class in all_pred_boxes:
                if gt_class == pred_class:
                    iou = self.compute_iou(gt_box, pred_box)
                    best_iou = max(best_iou, iou)
            total_recall_score += best_iou
        
        recall = total_recall_score / num_gt if num_gt > 0 else 0.0
        
        # 计算 Precision
        total_precision_score = 0.0
        for pred_box, pred_class in all_pred_boxes:
            best_iou = 0.0
            for gt_box, gt_class in all_gt_boxes:
                if pred_class == gt_class:
                    iou = self.compute_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
            total_precision_score += best_iou
        
        precision = total_precision_score / num_pred if num_pred > 0 else 0.0
        
        # 计算 F1 分数（连续值：0.0-1.0）
        if precision + recall == 0:
            iou_reward = 0.0
        else:
            iou_reward = 2 * (precision * recall) / (precision + recall)
        
        # ===== Part 2: CLIP Reward =====
        clip_reward = 0.0
        avg_clip_score = -1.0  # Initialize to -1 (failure/invalid)
        
        if DualRewardFunction._clip_model is not None and gt_data.get("image") and gt_data.get("intention_query"):
            # Compute CLIP cosine similarity for the best matching prediction
            total_clip_score = 0.0
            valid_preds = 0
            
            for pred_box, pred_class in all_pred_boxes:
                clip_score = self.compute_clip_score(
                    gt_data["image"],
                    pred_box,
                    gt_data["intention_query"],
                    width,
                    height
                )
                if clip_score > -1.0:  # Valid score (cosine similarity range: -1 to 1)
                    total_clip_score += clip_score
                    valid_preds += 1
            
            avg_clip_score = total_clip_score / valid_preds if valid_preds > 0 else -1.0
            clip_reward = 1.0 if avg_clip_score > self.clip_threshold else 0.0
        
        # ===== Combine Rewards =====
        print(f"IoU F1: {iou_reward:.3f} (P={precision:.3f}, R={recall:.3f}), CLIP: {avg_clip_score:.3f} (reward={clip_reward:.1f})")
        dual_reward = self.alpha * iou_reward + self.beta * clip_reward
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- Dual Reward: {dual_reward:.3f} | Dataset: {ground_truth.get('dataset_name', 'unknown')} -------------\n")
                f.write(f"IoU Reward (F1): {iou_reward:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})\n")
                f.write(f"CLIP Reward: {clip_reward:.1f} (avg score: {avg_clip_score:.3f}, threshold: {self.clip_threshold})\n")
                f.write(f"Weights: alpha={self.alpha}, beta={self.beta}\n")
                f.write(f"Prediction: {predict}\n")
                f.write(f"GT: {json.dumps(ground_truth['answer'])}\n\n")
        
        return dual_reward


# Register dual reward function
RewardFunctionFactory.register("dual", DualRewardFunction)


class VLMSemanticRewardFunction(BaseRewardFunction):
    """
    VLM Semantic Reward Function: Uses InternVL3.5 1B for captioning + BGE M3 for similarity
    
    Pipeline:
    1. Crop bbox region from image
    2. Use InternVL3.5 1B to generate caption for the region
    3. Use BGE M3 to compute similarity between caption and intention query
    4. Reward = 1 if similarity > 0.5, else 0
    """
    
    _vlm_model = None  # InternVL3.5 1B for captioning
    _vlm_tokenizer = None
    _bge_model = None  # BGE M3 for semantic similarity
    
    def __init__(self):
        """Initialize VLM Semantic Reward Function"""
        super().__init__()
        
        # Configuration
        self.similarity_threshold = float(os.getenv("VLM_SIMILARITY_THRESHOLD", "0.5"))
        
        # Load models (shared across instances)
        if VLMSemanticRewardFunction._vlm_model is None:
            self._load_models()
    
    def _load_models(self):
        """Load InternVL3.5 1B and BGE M3 models"""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from FlagEmbedding import BGEM3FlagModel
            
            # Force set CUDA device to GPU 2 (user's default)
            # This overrides Ray's worker environment
            gpu_id = os.getenv("VLM_GPU_ID", "2")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            print(f"🔧 Forcing CUDA_VISIBLE_DEVICES={gpu_id} for VLM reward")
            print(f"🔍 torch.cuda.is_available() = {torch.cuda.is_available()}")
            
            if not torch.cuda.is_available():
                print("⚠️  ERROR: CUDA not available even after setting CUDA_VISIBLE_DEVICES")
                print(f"   Make sure GPU {gpu_id} exists on this machine")
                VLMSemanticRewardFunction._vlm_model = None
                VLMSemanticRewardFunction._bge_model = None
                return
            
            device = "cuda:0"  # After setting CUDA_VISIBLE_DEVICES, it's always cuda:0
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"📍 Loading VLM and BGE models on GPU {gpu_id} (mapped to {device})")
            
            # Load InternVL3.5 1B for captioning
            vlm_model_name = os.getenv("VLM_MODEL_NAME", "OpenGVLab/InternVL3_5-1B")
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Loading VLM model: {vlm_model_name}")
            
            VLMSemanticRewardFunction._vlm_tokenizer = AutoTokenizer.from_pretrained(
                vlm_model_name, trust_remote_code=True, use_fast=False
            )
            VLMSemanticRewardFunction._vlm_model = AutoModel.from_pretrained(
                vlm_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,  # Use cuda:0 explicitly
                attn_implementation="eager"  # Disable FlashAttention2
            ).eval()
            
            if os.getenv("DEBUG_MODE") == "true":
                print("✅ VLM model loaded successfully")
            
            # Load BGE M3 for semantic similarity using FlagEmbedding
            bge_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Loading BGE model: {bge_model_name}")
            
            VLMSemanticRewardFunction._bge_model = BGEM3FlagModel(
                bge_model_name,
                use_fp16=True,
                device=device  # Explicitly use cuda:0
            )
            
            if os.getenv("DEBUG_MODE") == "true":
                print("✅ BGE model loaded successfully")
                
        except Exception as e:
            import traceback
            print(f"⚠️  Warning: Failed to load models: {e}")
            if os.getenv("DEBUG_MODE") == "true":
                traceback.print_exc()
            print("   VLM Semantic Reward will not work")
            VLMSemanticRewardFunction._vlm_model = None
            VLMSemanticRewardFunction._bge_model = None
    
    def get_reward_name(self) -> str:
        return "vlm_semantic"
    
    @staticmethod
    def build_transform(input_size):
        """Build image transformation for InternVL"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio for dynamic preprocessing"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamic preprocessing for InternVL"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = VLMSemanticRewardFunction.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    @staticmethod
    def load_image_for_internvl(image, input_size=448, max_num=6):
        """Load and preprocess image for InternVL"""
        import torch
        
        transform = VLMSemanticRewardFunction.build_transform(input_size=input_size)
        images = VLMSemanticRewardFunction.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def generate_caption(self, image_data: str, bbox: List[float], width: int, height: int) -> str:
        """
        Generate caption for the bbox region using InternVL3.5 1B
        
        Args:
            image_data: Base64 encoded image string
            bbox: Bounding box [x0, y0, x1, y1]
            width: Image width
            height: Image height
            
        Returns:
            Caption string
        """
        if VLMSemanticRewardFunction._vlm_model is None:
            return ""
        
        try:
            import torch
            
            # Decode and crop image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((width, height))
            
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            
            if x0 >= x1 or y0 >= y1:
                return ""  # Invalid bbox
            
            cropped = img.crop((x0, y0, x1, y1))
            
            # Preprocess image for InternVL
            pixel_values = self.load_image_for_internvl(cropped, input_size=448, max_num=6)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Generate caption using InternVL
            question = "<image>\nDescribe what you see in this image in one sentence."
            
            generation_config = dict(
                max_new_tokens=100,
                do_sample=False,
            )
            
            with torch.no_grad():
                response = VLMSemanticRewardFunction._vlm_model.chat(
                    VLMSemanticRewardFunction._vlm_tokenizer,
                    pixel_values,
                    question,
                    generation_config
                )
            
            return response.strip()
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Warning: Caption generation failed: {e}")
            return ""
    
    def compute_similarity(self, caption: str, query: str) -> float:
        """
        Compute semantic similarity between caption and query using BGE M3
        
        Args:
            caption: Generated caption
            query: Intention query
            
        Returns:
            Cosine similarity score [0, 1]
        """
        if VLMSemanticRewardFunction._bge_model is None or not caption or not query:
            return 0.0
        
        try:
            import numpy as np
            
            # Encode using BGE M3
            caption_embedding = VLMSemanticRewardFunction._bge_model.encode(
                [caption],
                batch_size=1,
                max_length=512
            )['dense_vecs']
            
            query_embedding = VLMSemanticRewardFunction._bge_model.encode(
                [query],
                batch_size=1,
                max_length=512
            )['dense_vecs']
            
            # Compute cosine similarity
            similarity_matrix = caption_embedding @ query_embedding.T
            similarity = float(similarity_matrix[0, 0])
            
            # BGE M3 similarity is already in range [0, 1] approximately
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Warning: Similarity computation failed: {e}")
            return 0.0
    
    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """Parse ground truth (reuse DualReward logic)"""
        answer = ground_truth["answer"]
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size
        
        objects = {}
        for class_name, class_data in answer.items():
            if "boxes" in class_data:
                objects[class_name] = class_data["boxes"]
        
        # Extract image data and intention query
        image_data = ground_truth.get("image", None)
        intention_query = ground_truth.get("intention_query", "")
        
        return {
            "dims": (width, height),
            "objects": objects,
            "raw_data": answer,
            "image": image_data,
            "intention_query": intention_query
        }
    
    def parse_detection_output(self, text: str, width: int, height: int) -> Optional[Dict]:
        """Parse model output (reuse DualReward logic)"""
        try:
            text = text.replace("\n", "").strip()
            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )
            
            matches = re.findall(pattern, text)
            
            for class_name, boxes_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []
                
                box_pattern = r"<(\d+)>"
                all_coords = re.findall(box_pattern, boxes_str)
                
                for i in range(0, len(all_coords), 4):
                    if i + 3 < len(all_coords):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in all_coords[i:i+4]]
                            x1_abs = x1 / 1000.0 * width
                            y1_abs = y1 / 1000.0 * height
                            x2_abs = x2 / 1000.0 * width
                            y2_abs = y2 / 1000.0 * height
                            
                            x1_final = min(x1_abs, x2_abs)
                            y1_final = min(y1_abs, y2_abs)
                            x2_final = max(x1_abs, x2_abs)
                            y2_final = max(y1_abs, y2_abs)
                            
                            if x1_final < x2_final and y1_final < y2_final:
                                objects[class_name].append([x1_final, y1_final, x2_final, y2_final])
                        except (ValueError, IndexError):
                            continue
            
            if not objects:
                return None
            
            return {"objects": objects}
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Error parsing detection output: {e}")
            return None
    
    def compute_reward(self, predict: str, ground_truth: Dict) -> Dict[str, float]:
        """
        Compute VLM Semantic Reward
        
        Returns:
            Dictionary with reward scores for WandB reporting
        """
        gt_parsed = self.parse_ground_truth(ground_truth)
        if gt_parsed is None:
            return {"overall": 0.0, "vlm_semantic": 0.0, "similarity": 0.0}
        
        pred_parsed = self.parse_detection_output(predict, gt_parsed["dims"][0], gt_parsed["dims"][1])
        if pred_parsed is None:
            return {"overall": 0.0, "vlm_semantic": 0.0, "similarity": 0.0}
        
        width, height = gt_parsed["dims"]
        image_data = gt_parsed["image"]
        intention_query = gt_parsed["intention_query"]
        
        if not image_data or not intention_query:
            return {"overall": 0.0, "vlm_semantic": 0.0, "similarity": 0.0}
        
        # Collect all predicted boxes
        all_pred_boxes = []
        for class_boxes in pred_parsed["objects"].values():
            all_pred_boxes.extend(class_boxes)
        
        if not all_pred_boxes:
            return {"overall": 0.0, "vlm_semantic": 0.0, "similarity": 0.0}
        
        # Process each predicted box
        max_similarity = 0.0
        best_caption = ""
        
        for bbox in all_pred_boxes:
            # Generate caption for this bbox
            caption = self.generate_caption(image_data, bbox, width, height)
            
            if caption:
                # Compute similarity
                similarity = self.compute_similarity(caption, intention_query)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_caption = caption
        
        # Binary reward based on threshold
        reward = 1.0 if max_similarity > self.similarity_threshold else 0.0
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- VLM Semantic Reward: {reward:.1f} -------------\n")
                    f.write(f"Max Similarity: {max_similarity:.3f} (threshold: {self.similarity_threshold})\n")
                    f.write(f"Best Caption: {best_caption}\n")
                    f.write(f"Intention Query: {intention_query}\n")
                    f.write(f"Prediction: {predict}\n\n")
        
        return {
            "overall": reward,
            "vlm_semantic": reward,
            "similarity": max_similarity
        }


# Register VLM semantic reward function
RewardFunctionFactory.register("vlm_semantic", VLMSemanticRewardFunction)


class IoUVLMRewardFunction(BaseRewardFunction):
    """
    IoU + VLM Semantic Hybrid Reward Function
    
    Combines:
    1. IoU Reward: Position accuracy (binary 0 or 1)
    2. VLM Semantic Reward: Semantic understanding via InternVL + BGE M3 (binary 0 or 1)
    
    Final Reward = alpha * IoU_reward + beta * VLM_reward
    Default: alpha=0.5, beta=0.5 (equal weight)
    """
    
    # Shared VLM and BGE models (same as VLMSemanticRewardFunction)
    _vlm_model = None
    _vlm_tokenizer = None
    _bge_model = None
    
    def __init__(self):
        """Initialize IoU + VLM Hybrid Reward Function"""
        super().__init__()
        
        # Configuration (can be adjusted via environment variables)
        self.alpha = float(os.getenv("IOU_VLM_ALPHA", "0.5"))  # IoU weight
        self.beta = float(os.getenv("IOU_VLM_BETA", "0.5"))    # VLM weight
        self.iou_threshold = float(os.getenv("IOU_VLM_IOU_THRESHOLD", "0.5"))
        self.vlm_threshold = float(os.getenv("IOU_VLM_VLM_THRESHOLD", "0.5"))
        
        # Load VLM models (shared with VLMSemanticRewardFunction)
        if IoUVLMRewardFunction._vlm_model is None:
            self._load_models()
    
    def _load_models(self):
        """Load InternVL3.5 1B and BGE M3 models"""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from FlagEmbedding import BGEM3FlagModel
            
            # Force set CUDA device to GPU 2 (user's default)
            # This overrides Ray's worker environment
            gpu_id = os.getenv("VLM_GPU_ID", "2")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            print(f"🔧 Forcing CUDA_VISIBLE_DEVICES={gpu_id} for IoU+VLM reward")
            print(f"🔍 torch.cuda.is_available() = {torch.cuda.is_available()}")
            
            if not torch.cuda.is_available():
                print("⚠️  ERROR: CUDA not available even after setting CUDA_VISIBLE_DEVICES")
                print(f"   Make sure GPU {gpu_id} exists on this machine")
                print("   IoU+VLM will fall back to pure IoU")
                IoUVLMRewardFunction._vlm_model = None
                IoUVLMRewardFunction._bge_model = None
                return
            
            device = "cuda:0"  # After setting CUDA_VISIBLE_DEVICES, it's always cuda:0
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"📍 Loading VLM and BGE models on GPU {gpu_id} (mapped to {device})")
            
            # Load InternVL3.5 1B
            vlm_model_name = os.getenv("VLM_MODEL_NAME", "OpenGVLab/InternVL3_5-1B")
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Loading VLM model: {vlm_model_name}")
            
            IoUVLMRewardFunction._vlm_tokenizer = AutoTokenizer.from_pretrained(
                vlm_model_name, trust_remote_code=True, use_fast=False
            )
            IoUVLMRewardFunction._vlm_model = AutoModel.from_pretrained(
                vlm_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,  # Use cuda:0 explicitly
                attn_implementation="eager"  # Disable FlashAttention2
            ).eval()
            
            if os.getenv("DEBUG_MODE") == "true":
                print("✅ VLM model loaded successfully")
            
            # Load BGE M3
            bge_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Loading BGE model: {bge_model_name}")
            
            IoUVLMRewardFunction._bge_model = BGEM3FlagModel(
                bge_model_name,
                use_fp16=True,
                device=device  # Explicitly use cuda:0
            )
            
            if os.getenv("DEBUG_MODE") == "true":
                print("✅ BGE model loaded successfully")
                
        except Exception as e:
            import traceback
            print(f"⚠️  Warning: Failed to load models: {e}")
            if os.getenv("DEBUG_MODE") == "true":
                traceback.print_exc()
            print("   IoU+VLM Reward will fall back to pure IoU")
            IoUVLMRewardFunction._vlm_model = None
            IoUVLMRewardFunction._bge_model = None
    
    def get_reward_name(self) -> str:
        return "iou_vlm"
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def build_transform(input_size):
        """Build image transformation for InternVL"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio for dynamic preprocessing"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        """Dynamic preprocessing for InternVL"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = IoUVLMRewardFunction.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    @staticmethod
    def load_image_for_internvl(image, input_size=448, max_num=6):
        """Load and preprocess image for InternVL"""
        import torch
        
        transform = IoUVLMRewardFunction.build_transform(input_size=input_size)
        images = IoUVLMRewardFunction.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def generate_caption(self, image_data: str, bbox: List[float], width: int, height: int) -> str:
        """Generate caption for the bbox region using InternVL3.5 1B"""
        if IoUVLMRewardFunction._vlm_model is None:
            return ""
        
        try:
            import torch
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((width, height))
            
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            
            if x0 >= x1 or y0 >= y1:
                return ""
            
            cropped = img.crop((x0, y0, x1, y1))
            
            pixel_values = self.load_image_for_internvl(cropped, input_size=448, max_num=6)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            question = "<image>\nDescribe what you see in this image in one sentence."
            
            generation_config = dict(
                max_new_tokens=100,
                do_sample=False,
            )
            
            with torch.no_grad():
                response = IoUVLMRewardFunction._vlm_model.chat(
                    IoUVLMRewardFunction._vlm_tokenizer,
                    pixel_values,
                    question,
                    generation_config
                )
            
            return response.strip()
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Warning: Caption generation failed: {e}")
            return ""
    
    def compute_similarity(self, caption: str, query: str) -> float:
        """Compute semantic similarity between caption and query using BGE M3"""
        if IoUVLMRewardFunction._bge_model is None or not caption or not query:
            return 0.0
        
        try:
            import numpy as np
            
            caption_embedding = IoUVLMRewardFunction._bge_model.encode(
                [caption],
                batch_size=1,
                max_length=512
            )['dense_vecs']
            
            query_embedding = IoUVLMRewardFunction._bge_model.encode(
                [query],
                batch_size=1,
                max_length=512
            )['dense_vecs']
            
            similarity_matrix = caption_embedding @ query_embedding.T
            similarity = float(similarity_matrix[0, 0])
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Warning: Similarity computation failed: {e}")
            return 0.0
    
    def parse_ground_truth(self, ground_truth: str) -> Optional[Dict]:
        """Parse ground truth"""
        answer = ground_truth["answer"]
        resized_size = ground_truth["resized_image_size"]
        width, height = resized_size
        
        objects = {}
        for class_name, class_data in answer.items():
            if "boxes" in class_data:
                objects[class_name] = class_data["boxes"]
        
        image_data = ground_truth.get("image", None)
        intention_query = ground_truth.get("intention_query", "")
        
        return {
            "dims": (width, height),
            "objects": objects,
            "raw_data": answer,
            "image": image_data,
            "intention_query": intention_query
        }
    
    def parse_detection_output(self, text: str, width: int, height: int) -> Optional[Dict]:
        """Parse model output"""
        try:
            text = text.replace("\n", "").strip()
            objects = {}
            pattern = (
                r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
                r"<\|box_start\|>(.*?)<\|box_end\|>"
            )
            
            matches = re.findall(pattern, text)
            
            for class_name, boxes_str in matches:
                class_name = class_name.strip()
                if class_name not in objects:
                    objects[class_name] = []
                
                box_pattern = r"<(\d+)>"
                all_coords = re.findall(box_pattern, boxes_str)
                
                for i in range(0, len(all_coords), 4):
                    if i + 3 < len(all_coords):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in all_coords[i:i+4]]
                            x1_abs = x1 / 1000.0 * width
                            y1_abs = y1 / 1000.0 * height
                            x2_abs = x2 / 1000.0 * width
                            y2_abs = y2 / 1000.0 * height
                            
                            x1_final = min(x1_abs, x2_abs)
                            y1_final = min(y1_abs, y2_abs)
                            x2_final = max(x1_abs, x2_abs)
                            y2_final = max(y1_abs, y2_abs)
                            
                            if x1_final < x2_final and y1_final < y2_final:
                                objects[class_name].append([x1_final, y1_final, x2_final, y2_final])
                        except (ValueError, IndexError):
                            continue
            
            if not objects:
                return None
            
            return {"objects": objects}
            
        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Error parsing detection output: {e}")
            return None
    
    def compute_reward(self, predict: str, ground_truth: Dict) -> Dict[str, float]:
        """
        Compute IoU + VLM Hybrid Reward
        
        Returns:
            Dictionary with reward scores for WandB reporting
        """
        gt_parsed = self.parse_ground_truth(ground_truth)
        if gt_parsed is None:
            return {"overall": 0.0, "iou_vlm": 0.0, "iou": 0.0, "vlm": 0.0}
        
        pred_parsed = self.parse_detection_output(predict, gt_parsed["dims"][0], gt_parsed["dims"][1])
        if pred_parsed is None:
            return {"overall": 0.0, "iou_vlm": 0.0, "iou": 0.0, "vlm": 0.0}
        
        width, height = gt_parsed["dims"]
        image_data = gt_parsed["image"]
        intention_query = gt_parsed["intention_query"]
        
        # Compute IoU Reward
        all_ious = []
        for gt_class, gt_boxes in gt_parsed["objects"].items():
            if gt_class in pred_parsed["objects"]:
                pred_boxes = pred_parsed["objects"][gt_class]
                for gt_box in gt_boxes:
                    max_iou_for_gt = 0.0
                    for pred_box in pred_boxes:
                        iou = self.compute_iou(pred_box, gt_box)
                        max_iou_for_gt = max(max_iou_for_gt, iou)
                    all_ious.append(max_iou_for_gt)
        
        avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
        iou_reward = 1.0 if avg_iou > self.iou_threshold else 0.0
        
        # Compute VLM Semantic Reward
        vlm_reward = 0.0
        max_similarity = 0.0
        best_caption = ""
        
        if image_data and intention_query and IoUVLMRewardFunction._vlm_model is not None:
            all_pred_boxes = []
            for class_boxes in pred_parsed["objects"].values():
                all_pred_boxes.extend(class_boxes)
            
            if all_pred_boxes:
                for bbox in all_pred_boxes:
                    caption = self.generate_caption(image_data, bbox, width, height)
                    if caption:
                        similarity = self.compute_similarity(caption, intention_query)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_caption = caption
                
                vlm_reward = 1.0 if max_similarity > self.vlm_threshold else 0.0
        
        # Combined reward
        hybrid_reward = self.alpha * iou_reward + self.beta * vlm_reward
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- IoU+VLM Hybrid Reward: {hybrid_reward:.3f} -------------\n")
                    f.write(f"IoU Reward: {iou_reward:.1f} (avg IoU: {avg_iou:.3f}, threshold: {self.iou_threshold})\n")
                    f.write(f"VLM Reward: {vlm_reward:.1f} (similarity: {max_similarity:.3f}, threshold: {self.vlm_threshold})\n")
                    f.write(f"Weights: alpha={self.alpha}, beta={self.beta}\n")
                    f.write(f"Best Caption: {best_caption}\n")
                    f.write(f"Intention Query: {intention_query}\n")
                    f.write(f"Prediction: {predict}\n\n")
        
        return {
            "overall": hybrid_reward,
            "iou_vlm": hybrid_reward,
            "iou": iou_reward,
            "vlm": vlm_reward,
            "avg_iou": avg_iou,
            "similarity": max_similarity
        }


# Register IoU + VLM hybrid reward function
RewardFunctionFactory.register("iou_vlm", IoUVLMRewardFunction)


def compute_score(
    predicts: List[str], ground_truths: List[str]
) -> List[Dict[str, float]]:
    """批量计算分数 - 兼容性函数"""
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        # 处理Qwen2.5VL-32B格式
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)

        reward_name = ground_truth["reward_name"]
        # 获取对应的奖励函数
        reward_func = RewardFunctionFactory.get_reward_function(reward_name)
        if reward_func is None:
            print(f"Warning: Unknown reward function '{reward_name}', using default")
            reward_func = RewardFunctionFactory.get_reward_function("box_iou")

        accuracy_score = reward_func.compute_reward(predict, ground_truth)
        scores.append(
            {
                "overall": accuracy_score,
                f"{reward_name}": accuracy_score,
            }
        )

    return scores


# 兼容性函数 - 保持向后兼容
def accuracy_reward(predict: str, ground_truth: str) -> float:
    """兼容性函数，使用默认的box_iou奖励"""
    reward_func = RewardFunctionFactory.get_reward_function("box_iou")
    return reward_func.compute_reward(predict, ground_truth)


def ensure_dir_exists(path: str) -> bool:
    """确保目录存在"""
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory: {e}")
        return False


def create_visualization(
    gt_data: Dict,
    pred_data: Dict,
    reward_score: float,
    reward_name: str,
    dataset_name: str,
    image_data: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    创建GT和预测结果的可视化对比图

    Args:
        gt_data: Ground truth数据
        pred_data: 预测数据
        reward_score: 奖励分数
        reward_name: 奖励函数名称
        dataset_name: 数据集名称
        image_data: Base64编码的图像数据（可选）
        save_path: 保存路径（可选）
    """
    try:
        # 获取图像尺寸
        width, height = gt_data.get("dims", (1000, 1000))

        # 创建子图
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(20, 10))

        # 如果有图像数据，解码并显示
        if image_data:
            try:
                # 解码base64图像
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((width, height))

                ax_gt.imshow(img)
                ax_pred.imshow(img)
            except Exception as e:
                print(f"Failed to decode image: {e}")
                # 创建空白背景
                ax_gt.set_xlim(0, width)
                ax_gt.set_ylim(height, 0)
                ax_pred.set_xlim(0, width)
                ax_pred.set_ylim(height, 0)
        else:
            # 创建空白背景
            ax_gt.set_xlim(0, width)
            ax_gt.set_ylim(height, 0)
            ax_pred.set_xlim(0, width)
            ax_pred.set_ylim(height, 0)

        # 设置标题
        ax_gt.set_title(f"Ground Truth\n{dataset_name}", fontsize=14, fontweight="bold")
        ax_pred.set_title(
            f"Prediction (Reward: {reward_score:.3f})\n{reward_name}",
            fontsize=14,
            fontweight="bold",
        )

        # 根据奖励类型进行可视化
        if reward_name == "box_iou":
            _visualize_box_iou(ax_gt, ax_pred, gt_data, pred_data)
        elif reward_name == "point_in_box":
            _visualize_point_in_box(ax_gt, ax_pred, gt_data, pred_data)
        elif reward_name == "point_in_mask":
            _visualize_point_in_mask(ax_gt, ax_pred, gt_data, pred_data, width, height)
        elif reward_name == "rejection":
            _visualize_rejection(ax_gt, ax_pred, gt_data, pred_data)

        ax_gt.axis("off")
        ax_pred.axis("off")
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"Visualization failed: {e}")
        if "fig" in locals():
            plt.close(fig)


def _visualize_box_iou(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Box IoU任务"""
    # 获取颜色映射
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT框
    for category, boxes in gt_objects.items():
        color = category_to_color[category]
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=0.3,
            )
            ax_gt.add_patch(rect)

            # 添加标签
            label = f"{category}_{i+1}" if len(boxes) > 1 else category
            ax_gt.text(
                x0,
                y0 - 10,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

    # 绘制预测框
    for category, boxes in pred_objects.items():
        color = category_to_color[category]
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=0.3,
            )
            ax_pred.add_patch(rect)

            # 添加标签
            label = f"{category}_{i+1}" if len(boxes) > 1 else category
            ax_pred.text(
                x0,
                y0 - 10,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_point_in_box(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Point in Box任务"""
    # GT: 显示框和点
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT（框和点）
    for category, obj_data in gt_objects.items():
        color = category_to_color[category]
        boxes = obj_data.get("boxes", [])
        points = obj_data.get("points", [])

        # 绘制框
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax_gt.add_patch(rect)

        # 绘制GT点
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="white",
                facecolor=color,
                alpha=0.8,
            )
            ax_gt.add_patch(circle)

        # 添加标签
        if boxes and points:
            label = f"{category} (GT)"
            ax_gt.text(
                boxes[0][0],
                boxes[0][1] - 15,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )

    # 绘制预测点
    for category, points in pred_objects.items():
        color = category_to_color[category]
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="yellow",
                facecolor=color,
                alpha=0.8,
            )
            ax_pred.add_patch(circle)

            # 添加标签
            label = f"{category}_pred_{i+1}" if len(points) > 1 else f"{category}_pred"
            ax_pred.text(
                x + 15,
                y,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_point_in_mask(ax_gt, ax_pred, gt_data, pred_data, width, height):
    """可视化Point in Mask任务"""
    gt_objects = gt_data.get("objects", {})
    pred_objects = pred_data.get("objects", {})

    all_categories = set(gt_objects.keys()) | set(pred_objects.keys())
    colors = plt.cm.tab20(range(len(all_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(all_categories)}

    # 绘制GT（框、掩码和点）
    for category, obj_data in gt_objects.items():
        color = category_to_color[category]
        boxes = obj_data.get("boxes", [])
        masks = obj_data.get("masks", [])
        points = obj_data.get("points", [])

        # 绘制框
        for box in boxes:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax_gt.add_patch(rect)

        # 绘制掩码
        for mask in masks:
            try:
                if isinstance(mask, dict) and "counts" in mask:
                    binary_mask = coco_mask.decode(mask)
                elif isinstance(mask, list):
                    binary_mask = np.array(mask)
                else:
                    continue

                # 创建掩码轮廓
                contours = _get_mask_contours(binary_mask)
                for contour in contours:
                    polygon = patches.Polygon(
                        contour,
                        closed=True,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )
                    ax_gt.add_patch(polygon)
            except Exception as e:
                print(f"Failed to visualize mask: {e}")

        # 绘制GT点
        for point in points:
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="white",
                facecolor=color,
                alpha=0.8,
            )
            ax_gt.add_patch(circle)

    # 绘制预测点
    for category, points in pred_objects.items():
        color = category_to_color[category]
        for i, point in enumerate(points):
            x, y = point
            circle = patches.Circle(
                (x, y),
                radius=8,
                linewidth=3,
                edgecolor="yellow",
                facecolor=color,
                alpha=0.8,
            )
            ax_pred.add_patch(circle)

            # 添加标签
            label = f"{category}_pred_{i+1}" if len(points) > 1 else f"{category}_pred"
            ax_pred.text(
                x + 15,
                y,
                label,
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
            )


def _visualize_rejection(ax_gt, ax_pred, gt_data, pred_data):
    """可视化Rejection任务"""
    # GT: 显示应该为空
    ax_gt.text(
        0.5,
        0.5,
        "Should be EMPTY\n(No objects)",
        transform=ax_gt.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="green",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    # 预测结果
    pred_objects = pred_data.get("objects", {})
    total_predictions = sum(len(coords) for coords in pred_objects.values())

    if total_predictions == 0:
        # 正确拒绝
        ax_pred.text(
            0.5,
            0.5,
            "CORRECT REJECTION\n(No predictions)",
            transform=ax_pred.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="green",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )
    else:
        # 产生幻觉
        ax_pred.text(
            0.5,
            0.5,
            f"HALLUCINATION\n({total_predictions} predictions)",
            transform=ax_pred.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
        )

        # 显示错误的预测
        colors = plt.cm.tab20(range(len(pred_objects)))
        color_idx = 0
        for category, coords_list in pred_objects.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1

            for i, coords in enumerate(coords_list):
                if len(coords) == 4:  # 边界框
                    x0, y0, x1, y1 = coords
                    rect = patches.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=3,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )
                    ax_pred.add_patch(rect)
                elif len(coords) == 2:  # 点
                    x, y = coords
                    circle = patches.Circle(
                        (x, y),
                        radius=8,
                        linewidth=3,
                        edgecolor="red",
                        facecolor=color,
                        alpha=0.8,
                    )
                    ax_pred.add_patch(circle)


def _get_mask_contours(binary_mask):
    """从二进制掩码中提取轮廓"""
    try:
        import cv2

        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 转换为matplotlib可用的格式
        result = []
        for contour in contours:
            if len(contour) > 2:
                points = contour.reshape(-1, 2)
                result.append(points)
        return result
    except ImportError:
        # 如果没有cv2，使用简单的边界
        y_coords, x_coords = np.where(binary_mask)
        if len(y_coords) > 0:
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            return [
                np.array(
                    [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                )
            ]
        return []
