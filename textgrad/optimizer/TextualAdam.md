# TextualAdam 优化器

## 概述

TextualAdam 是 TextGrad 框架中实现的创新文本优化器，将经典的 Adam 优化算法适配到自然语言处理的文本优化场景中。这是首个将数值优化的先进技术成功映射到语言模型文本优化的实现。

## 核心创新

### 🎯 概念映射

| 传统Adam概念 | TextualAdam映射 | 实现方式 |
|------------|----------------|----------|
| 一阶动量 | 语义方向一致性 | 追踪梯度反馈的语义相似性 |
| 二阶动量 | 改进效果方差 | 评估历史优化的稳定性 |
| 自适应学习率 | 智能优化强度 | 基于历史表现调整策略 |
| 偏差修正 | 早期步骤修正 | 避免初期动量估计偏差 |

### 🧠 工作原理

#### 1. 一阶动量：语义方向追踪
```python
# 计算与历史梯度的语义相似性
similarities = [compute_semantic_similarity(current_grad, hist_grad) 
               for hist_grad in recent_gradients]
# 更新动量：相似性高 → 方向一致 → 动量增强
momentum = beta1 * prev_momentum + (1 - beta1) * avg_similarity
```

#### 2. 二阶动量：稳定性估计
```python
# 评估改进效果的方差
variance = sum((score - mean_score)**2 for score in recent_scores) / len(scores)
# 更新二阶动量
second_moment = beta2 * prev_second + (1 - beta2) * (improvement_score**2)
```

#### 3. 自适应优化策略
- **高一致性 + 高稳定性** → 强优化："Apply strong optimization with high confidence"
- **中等表现** → 适度优化："Apply moderate optimization intensity"  
- **低一致性 + 低稳定性** → 保守优化："Apply conservative optimization"

## 使用方法

### 基本用法

```python
import textgrad as tg

# 创建需要优化的文本变量
system_prompt = tg.Variable(
    "You are an AI assistant.",
    requires_grad=True,
    role_description="system prompt for LLM guidance"
)

# 初始化 TextualAdam 优化器
optimizer = tg.TAdam(
    parameters=[system_prompt],
    engine="gpt-4o",
    beta1=0.9,          # 一阶动量衰减率
    beta2=0.999,        # 二阶动量衰减率
    epsilon=1e-8,       # 数值稳定性参数
    momentum_window=5,  # 动量窗口大小
    verbose=1           # 显示详细过程
)

# 执行优化步骤
optimizer.step()
```

### 完整优化流程

```python
import textgrad as tg

# 设置引擎
tg.set_backward_engine("gpt-4o")
llm_engine = tg.get_engine("gpt-3.5-turbo")

# 创建模型和优化器
system_prompt = tg.Variable("You are a helpful assistant.", 
                           requires_grad=True,
                           role_description="system prompt")
model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TAdam(parameters=[system_prompt], engine=llm_engine)

# 多轮优化
for step in range(3):
    # 模型预测
    question = tg.Variable("What is machine learning?", 
                          requires_grad=False, 
                          role_description="user question")
    prediction = model(question)
    
    # 评估和反向传播
    loss = some_evaluation_function(prediction)
    loss.backward()
    
    # Adam 优化步骤
    optimizer.step()
    optimizer.zero_grad()
```

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `beta1` | 0.9 | 一阶动量衰减率，控制语义方向记忆强度 |
| `beta2` | 0.999 | 二阶动量衰减率，控制稳定性记忆强度 |
| `epsilon` | 1e-8 | 数值稳定性参数，防止除零错误 |
| `momentum_window` | 5 | 动量历史窗口大小 |
| `constraints` | None | 优化约束条件列表 |
| `verbose` | 0 | 详细输出级别 (0=静默, 1=详细) |

## 与其他优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| **TGD** | 简单直接，基于当前梯度 | 简单优化任务，快速迭代 |
| **TGD with Momentum** | 考虑历史信息，避免震荡 | 需要稳定收敛的任务 |
| **TextualAdam** | 自适应学习率，智能优化 | 复杂优化任务，需要精细控制 |

## 核心优势

### 🚀 智能自适应
- 根据历史反馈自动调整优化强度
- 避免过度优化或优化不足的问题

### 🎯 语义感知
- 基于语义相似性而非数值梯度进行优化
- 更符合自然语言处理的特点

### 📈 稳定收敛
- 通过方差估计评估优化稳定性
- 在不稳定时自动采用保守策略

### 🔧 易于使用
- 与现有 TextGrad 框架完全兼容
- 提供丰富的参数配置选项

## 实现细节

### 语义相似性计算
当前使用基于词汇重叠的简化实现：
```python
def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0
```

*注：实际应用中可替换为更复杂的语义嵌入模型*

### 改进分数估算
结合多个因素评估优化效果：
- 语义变化程度
- 梯度反馈中的情感倾向
- 文本复杂度变化

## 扩展建议

### 🔮 未来改进方向

1. **增强语义计算**
   - 集成 BERT/RoBERTa 等预训练模型
   - 使用句子级语义嵌入

2. **多模态支持**
   - 扩展到图像+文本优化
   - 支持多模态梯度反馈

3. **自适应参数**
   - 动态调整 beta1, beta2 参数
   - 基于任务类型自动配置

4. **并行优化**
   - 支持多变量并行优化
   - 变量间依赖关系建模

## 引用和致谢

TextualAdam 基于以下工作：
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
- TextGrad 框架的优化器设计模式
- 语义相似性计算的相关研究

---

**开发者**: TextGrad 团队  
**版本**: 1.0.0  
**最后更新**: 2025-07-28

如有问题或建议，请提交 Issue 或 Pull Request。

