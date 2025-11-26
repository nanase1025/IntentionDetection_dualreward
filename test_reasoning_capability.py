#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Rex-Omni 是否保留了基础模型的语言推理能力
Test if Rex-Omni retains the base model's language reasoning capabilities
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def test_reasoning_capability(checkpoint_path):
    """
    直接使用transformers加载模型，测试推理能力
    不通过RexOmniWrapper，以避免wrapper的限制
    """
    
    print("="*80)
    print("测试 1: 直接对话能力（无图像）")
    print("="*80)
    
    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    
    # Test 1: 纯文本推理任务
    text_questions = [
        "Please solve this step by step: What is 15 + 27 × 3?",
        "Explain your reasoning: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Think step by step: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ]
    
    for i, question in enumerate(text_questions, 1):
        print(f"\n问题 {i}: {question}\n")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
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
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
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
        
        print(f"回答: {output_text}")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("测试 2: 视觉推理能力（带图像）")
    print("="*80)
    
    # Test 2: 视觉推理任务
    image_path = "/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs/1.jpg"
    try:
        image = Image.open(image_path).convert("RGB")
        
        visual_questions = [
            "Describe what you see in this image.",
            "What is the main purpose of this room? Explain your reasoning.",
            "If someone said 'I need something to sit on', what objects in this image would be suitable? Explain why.",
        ]
        
        for i, question in enumerate(visual_questions, 1):
            print(f"\n问题 {i}: {question}\n")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
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
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
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
            
            print(f"回答: {output_text}")
            print("-" * 80)
    
    except FileNotFoundError:
        print("图片文件未找到，跳过视觉测试")
    
    print("\n" + "="*80)
    print("测试 3: 是否可以通过prompt引导生成thinking格式")
    print("="*80)
    
    # Test 3: 测试是否可以通过prompt引导输出thinking格式
    thinking_question = """Look at this image and answer: What objects are suitable for 'I need something to sit on while working'?

Please answer in this format:
<think>
[Your reasoning process here]
</think>
<answer>
[Your final answer here]
</answer>"""
    
    print(f"\n问题: {thinking_question}\n")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides detailed reasoning."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": thinking_question},
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
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
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
        
        print(f"回答: {output_text}")
        
        # 检查是否生成了thinking格式
        has_thinking = "<think>" in output_text.lower() and "<answer>" in output_text.lower()
        print(f"\n是否生成了thinking格式: {'✅ 是' if has_thinking else '❌ 否'}")
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
如果模型能够：
1. ✅ 回答纯文本推理问题 → 保留了语言推理能力
2. ✅ 描述图像内容 → 保留了视觉理解能力
3. ✅ 进行视觉推理 → 保留了视觉+语言的推理能力
4. ✅ 可以生成<think>格式 → 可以通过prompt引导生成thinking

如果不能：
- ❌ 模型能力可能在微调过程中被限制
- 可能的原因：
  1. 只在检测格式数据上训练，导致灾难性遗忘
  2. 输出格式被限制为坐标生成
  3. 需要混合训练（检测任务 + 对话任务）来保留能力
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="finetuning/work_dirs/ovigod_sft_5ep",
        help="Path to checkpoint"
    )
    args = parser.parse_args()
    
    print(f"加载检查点: {args.checkpoint}")
    test_reasoning_capability(args.checkpoint)

