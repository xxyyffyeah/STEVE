import sys
import os

# 确保使用当前目录的textgrad
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

import textgrad as tg
from textgrad.tasks import load_task
from dotenv import load_dotenv

load_dotenv(override=True)

print("=== TextualAdam优化器演示 ===")
print("基于BBH object counting任务的系统提示优化\n")

# 检查API密钥
if not os.getenv('OPENAI_API_KEY'):
    print("警告: 未检测到OPENAI_API_KEY环境变量")
    print("请设置API密钥后重新运行")
    exit(1)

llm_engine = tg.get_engine("gpt-3.5-turbo-0125")
tg.set_backward_engine("gpt-4o")

train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", llm_engine)
question_str, answer_str = val_set[0]
question_str = "I have four cauliflowers, a garlic, a cabbage, a potato, a head of broccoli, three yams, a lettuce head, an onion, and a carrot. How many vegetables do I have?"
answer_str = "14"
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
answer = tg.Variable(str(answer_str), role_description="answer to the question", requires_grad=False)

print(f"测试问题: {question_str}")
print(f"正确答案: {answer_str}\n")

STARTING_SYSTEM_PROMPT = train_set.get_task_description()

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)

# 使用TextualAdam优化器替代TGD
optimizer = tg.TAdam(
    parameters=list(model.parameters()),
    engine=llm_engine,
    beta1=0.9,      # 一阶动量衰减
    beta2=0.999,    # 二阶动量衰减  
    verbose=1       # 显示详细优化过程
)

print("\n--- 初始预测 ---")
print(f"初始系统提示: {system_prompt.value}")
prediction = model(question)
print(f"模型预测: {prediction.value}")

loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))
print(f"损失反馈: {loss.value}\n")

print("--- TextualAdam三轮优化过程 ---")

# 进行三轮优化迭代
for iteration in range(3):
    print(f"\n=== 第 {iteration + 1} 轮优化 ===")
    
    # 前向推理并计算损失
    prediction = model(question)
    loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))
    print(f"当前预测: {prediction.value}")
    print(f"当前损失: {loss.value}")
    
    # 反向传播计算梯度
    optimizer.zero_grad()
    loss.backward()
    
    # 打印梯度信息
    gradient_text = system_prompt.get_gradient_text()
    print(f"系统提示的梯度数量: {len(system_prompt.gradients)}")
    if gradient_text:
        print(f"梯度内容: {gradient_text}")
    else:
        print("暂无梯度信息")
    
    # 执行优化步骤
    print("执行优化步骤...")
    optimizer.step()
    
    print(f"优化后系统提示: {system_prompt.value}")

print("\n--- 最终结果对比 ---")
final_prediction = model(question)
print(f"最终预测: {final_prediction.value}")
final_loss = eval_fn(inputs=dict(prediction=final_prediction, ground_truth_answer=answer))
print(f"最终损失: {final_loss.value}")

print("\n=== TextualAdam优化器特性 ===")
print("✅ 一阶动量: 追踪梯度反馈的语义方向一致性")
print("✅ 二阶动量: 评估改进效果的历史方差") 
print("✅ 自适应学习率: 基于历史表现调整优化强度")
print("✅ 偏差修正: 避免早期步骤的动量估计偏差")
print("\n🎉 TextualAdam优化器演示完成!")