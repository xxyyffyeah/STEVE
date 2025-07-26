# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TextGrad is an autograd engine for textual gradients that implements backpropagation through text feedback provided by LLMs. It follows PyTorch-style API patterns for optimization of text-based variables like prompts, solutions, and code.

核心组件：
- **Variable**: 可优化的文本变量，类似PyTorch的tensor
- **BlackboxLLM**: 黑盒LLM模型包装器
- **TextLoss**: 自然语言描述的损失函数
- **TGD (Textual Gradient Descent)**: 文本梯度下降优化器
- **Engine**: 后端LLM引擎（OpenAI, Anthropic, Gemini等）

## Development Commands

### Installation
```bash
pip install -e .                    # 开发模式安装
pip install textgrad[vllm]         # 包含vllm支持
```

### Testing
```bash
python -m pytest tests/            # 运行所有测试
python -m pytest tests/test_basics.py  # 运行基础测试
python -m pytest tests/test_engines.py # 运行引擎测试
python -m pytest tests/test_api.py     # 运行API测试
```

### Running Examples
```bash
cd examples/notebooks/
jupyter notebook                    # 启动Jupyter查看教程
```

### Evaluation Scripts
```bash
python evaluation/prompt_optimization.py     # 提示优化评估
python evaluation/solution_optimization.py   # 解决方案优化评估
```

## Architecture

### Core Module Structure
- `textgrad/autograd/`: 自动微分核心功能
  - `string_based_ops.py`: 字符串操作的梯度计算
  - `llm_ops.py`: LLM操作包装
  - `function.py`: 计算图函数节点
- `textgrad/engine/`: LLM后端引擎
- `textgrad/optimizer/`: 文本优化器实现
- `textgrad/tasks/`: 预定义任务（BBH, GPQA, GSM8K等）

### Engine Configuration
设置后端引擎（必须在使用前配置）：
```python
import textgrad as tg
tg.set_backward_engine("gpt-4o", override=True)
```

支持的引擎：
- OpenAI: "gpt-4o", "gpt-3.5-turbo"
- Anthropic: "claude-3-sonnet"
- 实验性litellm引擎: "experimental:gpt-4o"

### Variable和计算图
所有可优化变量必须设置 `requires_grad=True`:
```python
variable = tg.Variable(content, requires_grad=True, role_description="描述")
```

### 日志系统
默认日志目录: `./logs/`
可通过环境变量 `TEXTGRAD_LOG_DIR` 修改

## Development Guidelines

### 引擎使用注意事项
- 必须设置API密钥环境变量（OPENAI_API_KEY等）
- 新的litellm引擎正在实验阶段，可能替代旧引擎
- 缓存功能可通过 `cache=True/False` 控制

### 测试要求
- 运行测试前确保设置了相应的API密钥
- 测试可能需要网络连接和API调用
- 某些测试可能需要较长时间完成

### 多模态支持
项目支持多模态优化，参见：
- `textgrad/autograd/multimodal_ops.py`
- `textgrad/tasks/multimodal/`
- examples中的多模态教程

### 代码贡献注意事项
- 遵循现有的代码风格和模块组织
- 新增引擎需同时支持同步和异步调用
- 确保向后兼容性，特别是Variable和Function接口
EOF < /dev/null