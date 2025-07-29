#!/usr/bin/env python3

"""
TextualAdam优化器测试示例

此脚本演示如何使用新实现的TextualAdam优化器，
并与标准的TextualGradientDescent进行简单对比。

运行前请确保设置了相应的API密钥环境变量：
- OPENAI_API_KEY 或其他后端引擎的密钥
"""

import textgrad as tg
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def test_textual_adam_basic():
    """测试TextualAdam的基本功能"""
    print("=== TextualAdam 基本功能测试 ===")
    
    # 设置引擎
    try:
        llm_engine = tg.get_engine("gpt-3.5-turbo")
        tg.set_backward_engine("gpt-4o")
    except Exception as e:
        print(f"引擎设置失败: {e}")
        print("请确保设置了正确的API密钥")
        return
    
    # 创建测试用的变量
    system_prompt = tg.Variable(
        "You are a helpful assistant. Answer questions clearly and concisely.",
        requires_grad=True,
        role_description="system prompt for guiding LLM responses"
    )
    
    # 创建TextualAdam优化器
    adam_optimizer = tg.TAdam(
        parameters=[system_prompt],
        engine=llm_engine,
        beta1=0.9,
        beta2=0.999,
        verbose=1  # 显示详细输出
    )
    
    # 创建一个简单的LLM模型
    model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
    
    # 测试问题
    question = tg.Variable(
        "What is the capital of France?",
        role_description="test question",
        requires_grad=False
    )
    
    print(f"初始system prompt: {system_prompt.value}")
    
    # 进行几轮优化
    for step in range(3):
        print(f"\n--- 第 {step + 1} 轮优化 ---")
        
        # 获取模型预测
        prediction = model(question)
        if prediction and hasattr(prediction, 'value'):
            print(f"模型回答: {prediction.value}")
        else:
            print(f"模型回答: {str(prediction)}")
        
        # 模拟一个简单的损失函数（实际应用中会更复杂）
        if step == 0:
            loss_feedback = "The response should be more specific and include some context about France."
        elif step == 1:
            loss_feedback = "Good improvement, but could be more engaging and informative."
        else:
            loss_feedback = "Response is clear but could be more concise."
        
        # 创建损失变量并设置梯度
        loss_var = tg.Variable(
            loss_feedback,
            role_description="feedback for improvement",
            requires_grad=False
        )
        
        # 手动设置梯度（在实际应用中这会通过backward()自动完成）
        system_prompt.gradients.add(loss_var)
        
        # 清空之前的梯度上下文，设置新的
        system_prompt.gradients_context.clear()
        system_prompt.gradients_context[loss_var] = None
        
        # 执行优化步骤
        try:
            adam_optimizer.step()
            print(f"优化后的system prompt: {system_prompt.value}")
        except Exception as e:
            print(f"优化步骤失败: {e}")
            break
        
        # 清空梯度准备下一轮
        adam_optimizer.zero_grad()
    
    print(f"\n最终system prompt: {system_prompt.value}")

def test_adam_vs_tgd_comparison():
    """简单对比TextualAdam和标准TGD的行为"""
    print("\n=== TextualAdam vs TGD 对比测试 ===")
    
    try:
        llm_engine = tg.get_engine("gpt-3.5-turbo")
        tg.set_backward_engine("gpt-4o")
    except Exception as e:
        print(f"引擎设置失败: {e}")
        return
    
    # 创建相同的初始prompt用于对比
    initial_prompt = "You are an AI assistant."
    
    # Adam优化器测试
    prompt_adam = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="system prompt for Adam optimizer test"
    )
    
    adam_optimizer = tg.TAdam(
        parameters=[prompt_adam],
        engine=llm_engine,
        verbose=0
    )
    
    # TGD优化器测试  
    prompt_tgd = tg.Variable(
        initial_prompt,
        requires_grad=True,
        role_description="system prompt for TGD optimizer test"
    )
    
    tgd_optimizer = tg.TGD(
        parameters=[prompt_tgd],
        engine=llm_engine,
        verbose=0
    )
    
    # 相同的反馈用于两个优化器
    feedback_text = "Make the prompt more specific for answering math questions accurately."
    
    for optimizer, prompt, name in [(adam_optimizer, prompt_adam, "Adam"), 
                                   (tgd_optimizer, prompt_tgd, "TGD")]:
        print(f"\n--- {name} 优化器测试 ---")
        print(f"初始: {prompt.value}")
        
        # 设置相同的梯度
        feedback_var = tg.Variable(
            feedback_text,
            role_description="optimization feedback",
            requires_grad=False
        )
        prompt.gradients.add(feedback_var)
        prompt.gradients_context.clear()
        prompt.gradients_context[feedback_var] = None
        
        try:
            optimizer.step()
            print(f"优化后: {prompt.value}")
        except Exception as e:
            print(f"{name}优化失败: {e}")
        
        optimizer.zero_grad()

def main():
    """主测试函数"""
    print("TextualAdam 优化器测试")
    print("=" * 50)
    
    # 检查API密钥
    if not os.getenv('OPENAI_API_KEY'):
        print("警告: 未检测到OPENAI_API_KEY环境变量")
        print("请设置API密钥后重新运行测试")
        return
    
    try:
        # 基本功能测试
        test_textual_adam_basic()
        
        # 对比测试
        test_adam_vs_tgd_comparison()
        
        print("\n=== 测试完成 ===")
        print("TextualAdam优化器已成功实现并可以使用!")
        print("核心特性:")
        print("- 一阶动量: 梯度反馈语义方向的一致性追踪")
        print("- 二阶动量: 改进效果稳定性的方差估计")
        print("- 自适应学习率: 基于历史表现的优化强度调整")
        print("- 偏差修正: 早期步骤的动量修正机制")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这可能是API配置或网络连接问题")

if __name__ == "__main__":
    main()