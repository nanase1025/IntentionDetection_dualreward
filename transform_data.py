import json
import os
import re
from PIL import Image
from tqdm import tqdm

def parse_target_bboxes(target_string):
    """
    从target字符串中提取所有bbox坐标
    例如: "description<loc_1><loc_2><loc_304><loc_998><loc_215><loc_19><loc_430><loc_704>"
    返回: [[1, 2, 304, 998], [215, 19, 430, 704]]
    """
    # 提取所有 <loc_xxx> 格式的坐标
    coords = re.findall(r'<loc_(\d+)>', target_string)
    coords = [int(c) for c in coords]
    
    # 每4个坐标组成一个bbox
    bboxes = []
    for i in range(0, len(coords), 4):
        if i + 3 < len(coords):
            bboxes.append([coords[i], coords[i+1], coords[i+2], coords[i+3]])
    
    return bboxes

def convert_ovigod_to_rexomni_format(
    input_json_path,
    image_root_path,
    output_json_path,
    use_affordance=True  # True: 使用意图描述, False: 使用类别名
):
    """
    转换 OV-IGOD 数据集格式到 Rex-Omni 训练格式
    
    Args:
        input_json_path: 输入的 train.json 路径
        image_root_path: 图片根目录
        output_json_path: 输出的 JSON 文件路径（每行一个JSON对象）
        use_affordance: 是否使用 affordance 作为 phrase（否则使用 class_name）
    """
    # 读取原始数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备输出文件
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        for image_id, sample in tqdm(data.items(), desc="Converting"):
            image_path = os.path.join(image_root_path, f"{image_id}.jpg")
            
            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # 获取图片尺寸（用于验证坐标范围）
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Cannot open image {image_path}: {e}")
                continue
            
            # 转换 bboxes
            boxes_list = []
            for bbox_item in sample['bboxes']:
                # 决定使用 affordance 还是 class_name 作为 phrase
                if use_affordance:
                    phrase = bbox_item['affordance']
                else:
                    phrase = bbox_item['class_name']
                
                # 从 target 中解析出所有的 bbox（一个对象可能有多个部分）
                target_bboxes = parse_target_bboxes(bbox_item['target'])
                
                # 将每个bbox转换为 Rex-Omni 格式
                # 注意：target 中的坐标已经是 [0, 999] 归一化的
                # 但我们需要转回绝对坐标，因为 Rex-Omni 的转换脚本会自动归一化
                for norm_bbox in target_bboxes:
                    # 从 [0, 999] 转回绝对坐标
                    x0 = (norm_bbox[0] / 999.0) * img_width
                    y0 = (norm_bbox[1] / 999.0) * img_height
                    x1 = (norm_bbox[2] / 999.0) * img_width
                    y1 = (norm_bbox[3] / 999.0) * img_height
                    
                    boxes_list.append({
                        "bbox": [x0, y0, x1, y1],
                        "phrase": phrase
                    })
            
            # 构造输出格式
            output_item = {
                "image_name": f"{image_id}.jpg",
                "annotation": {
                    "boxes": boxes_list
                }
            }
            
            # 写入文件（每行一个JSON对象）
            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed! Output saved to: {output_json_path}")

# 使用示例
if __name__ == "__main__":
    convert_ovigod_to_rexomni_format(
        input_json_path="/workspace/hairong/data/ov-igod-dataset/train.json",
        image_root_path="/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs",
        output_json_path="/workspace/hairong/data/ov-igod-dataset/rexomni_train.json",
        use_affordance=True  # 使用意图描述作为检测目标
    )