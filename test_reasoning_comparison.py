#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比测试：原始Qwen2.5-VL vs 微调后的Rex-Omni
Comparison: Original Qwen2.5-VL vs Fine-tuned Rex-Omni
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def test_model(model_path, model_name):
    """测试单个模型"""
    
    print("="*80)
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print("="*80)
    
    try:
        # 加载模型
        print("\n加载模型中...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        print("✅ 模型加载成功\n")
        
        # Test 1: 纯文本任务（不使用图像）
        print("-" * 80)
        print("测试 1: 纯文本推理（无图像）")
        print("-" * 80)
        
        text_question = "What is 2 + 2?"
        
        print(f"问题: {text_question}")
        
        try:
            messages = [
                {"role": "user", "content": text_question}
            ]
            
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            print(f"✅ 回答: {output_text}")
            
        except Exception as e:
            print(f"❌ 纯文本测试失败: {str(e)[:200]}")
        
        # Test 2: 视觉问答（带图像）
        print("\n" + "-" * 80)
        print("测试 2: 视觉问答（带图像）")
        print("-" * 80)
        
        image_path = "/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs/1.jpg"
        
        try:
            image = Image.open(image_path).convert("RGB")
            visual_question = "What room is this?"
            
            print(f"问题: {visual_question}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": visual_question},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            print(f"✅ 回答: {output_text}")
            
        except FileNotFoundError:
            print("❌ 图片文件未找到")
        except Exception as e:
            print(f"❌ 视觉测试失败: {str(e)[:200]}")
        
        # Test 3: 检测任务（Rex-Omni的专长）
        print("\n" + "-" * 80)
        print("测试 3: 物体检测任务")
        print("-" * 80)
        
        try:
            detection_prompt = "Detect bed in this image. Output the bounding box coordinates in [x0, y0, x1, y1] format."
            
            print(f"问题: {detection_prompt}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": detection_prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            print(f"✅ 回答: {output_text[:500]}")  # 只显示前500字符
            
        except Exception as e:
            print(f"❌ 检测测试失败: {str(e)[:200]}")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           对比测试：官方 Rex-Omni vs 微调模型的能力                         ║
╚════════════════════════════════════════════════════════════════════════════╝

这个测试将对比：
1. 官方 IDEA-Research/Rex-Omni（预训练的检测模型）
2. 微调后的 Rex-Omni（在OV-IGOD affordance数据集上训练）

测试内容：
- 纯文本推理能力（基础模型是否保留）
- 视觉问答能力（基础模型是否保留）
- 物体检测能力（专长任务）
""")
    
    # Test 1: 官方 Rex-Omni
    print("\n" + "🔵" * 40)
    test_model("IDEA-Research/Rex-Omni", "官方 Rex-Omni（预训练）")
    
    # Test 2: 微调后的Rex-Omni
    print("\n" + "🟢" * 40)
    test_model("finetuning/work_dirs/ovigod_sft_5ep", "微调后的 Rex-Omni (OV-IGOD)")
    
    # 总结
    print("\n" + "="*80)
    print("📊 总结")
    print("="*80)
    print("""
如果观察到：

情况 A: 原始模型 ✅ 全通过，微调模型 ✅ 全通过
  → 完美！微调保留了所有能力
  → 可以直接加入thinking机制

情况 B: 原始模型 ✅ 全通过，微调模型 ❌ 纯文本失败，✅ 视觉任务成功
  → 部分灾难性遗忘
  → 需要混合训练（检测 + 对话数据）来恢复能力

情况 C: 原始模型 ✅ 全通过，微调模型 ❌ 多项失败
  → 严重的灾难性遗忘
  → 建议：
    1. 使用LoRA等轻量微调方法
    2. 混合多任务数据训练
    3. 降低学习率
    4. 使用replay buffer保留原始能力
    """)

