from abc import ABC, abstractmethod
from typing import List, Union, Optional
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts import construct_tgd_prompt, OPTIMIZER_SYSTEM_PROMPT, GRADIENT_TEMPLATE, GRADIENT_MULTIPART_TEMPLATE


def get_gradient_and_context_text(variable) -> Union[str, List[Union[str, bytes]]]:
    """For the variable, aggregates and returns 
    i. the gradients 
    ii. the context for which the gradients are computed.

    This is used by the optimizer.  
    :return: A string containing the aggregated gradients and their corresponding context.
    :rtype: str
    """

    gradient_content = []
    for g in variable.gradients:
        if variable.gradients_context[g] is None:
            gradient_content.append(g.value)
        else:
            # If context is a list, we handle it differently.
            context = variable.gradients_context[g]
            if isinstance(context["context"], str):
                # The context could be all string.
                criticism_and_context = GRADIENT_TEMPLATE.format(
                    feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context["context"], list):
                # The context may have a list of images / strings. In this case, we need to handle it differently.
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")
    
    # Check if all instances are string
    if all(isinstance(i, str) for i in gradient_content):
        return "\n".join(gradient_content)
    else:
        return gradient_content


class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        for parameter in parameters:
            if type(parameter.value) !=  str:
                raise NotImplementedError(f"We cannot yet update multimodal content and this data type: {type(parameter.value)}. We can only evaluate gradients using multimodal models. This may change soon (looking at you, GPT-5).")
        self.parameters = parameters
        
    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass


class TextualGradientDescent(Optimizer):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[EngineLM, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0):
        """TextualGradientDescent optimizer

        :param engine: the engine to use for updating variables
        :type engine: EngineLM
        :param parameters: the parameters to optimize
        :type parameters: List[Variable]
        :param verbose: whether to print iterations, defaults to 0
        :type verbose: int, optional
        :param constraints: a list of natural language constraints, defaults to []
        :type constraints: List[str], optional
        :param optimizer_system_prompt: system prompt to the optimizer, defaults to textgrad.prompts.OPTIMIZER_SYSTEM_PROMPT. Needs to accept new_variable_start_tag and new_variable_end_tag
        :type optimizer_system_prompt: str, optional
        :param in_context_examples: a list of in-context examples, defaults to []
        :type in_context_examples: List[str], optional
        :param gradient_memory: the number of past gradients to store, defaults to 0
        :type gradient_memory: int, optional
        """
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.do_constrained = (len(self.constraints) > 0)
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)

    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory
    
    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})
    
    def _update_prompt(self, variable: Variable) -> Union[str, List[Union[str, bytes]]]:
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory
        }
        
        prompt = construct_tgd_prompt(do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientDescent prompt for update", extra={"prompt": prompt})
        return prompt

    def step(self):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
            
            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)


class TextualGradientDescentwithMomentum(Optimizer):
    def __init__(self, 
                 engine: Union[str, EngineLM], 
                 parameters: List[Variable], 
                 momentum_window: int = 0, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 in_context_examples: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        
        if momentum_window == 0:
            return TextualGradientDescent(engine=engine, parameters=parameters, constraints=constraints)

        # Each item in the momentum storage will include past value and the criticism
        self.momentum_storage = [[] for _ in range(len(parameters))]
        self.momentum_window = momentum_window
        self.do_momentum = True
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)

        logger.info(f"TextualGradientDescent initialized with momentum window: {momentum_window}")

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def _update_prompt(self, variable: Variable, momentum_storage_idx: int):
        past_values = ""
        
        past_n_steps = self.momentum_storage[momentum_storage_idx]
        for i, step_info in enumerate(past_n_steps):
            past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "past_values": past_values,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples)
        }
        
        prompt = construct_tgd_prompt(do_momentum=(self.do_momentum and (past_values != "")), 
                                      do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientwithMomentum prompt for update", extra={"prompt": prompt})


    def _update_momentum_storage(self, variable: Variable, momentum_storage_idx: int):
        if len(self.momentum_storage[momentum_storage_idx]) >= self.momentum_window:
            self.momentum_storage[momentum_storage_idx].pop(0)
        
        self.momentum_storage[momentum_storage_idx].append({"value": variable.value, "gradients": get_gradient_and_context_text(variable)})
        
    def step(self):
        for idx, parameter in enumerate(self.parameters):
            self._update_momentum_storage(parameter, momentum_storage_idx=idx)
            prompt_update_parameter = self._update_prompt(parameter, momentum_storage_idx=idx)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescentwithMomentum optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescentwithMomentum updated text", extra={"parameter.value": parameter.value})


class TextualAdam(Optimizer):
    def __init__(self, 
                 parameters: List[Variable],
                 engine: Union[str, EngineLM], 
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 constraints: Optional[List[str]] = None,
                 new_variable_tags: Optional[List[str]] = None,
                 optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: Optional[List[str]] = None,
                 momentum_window: int = 5,
                 verbose: int = 0):
        """TextualAdam optimizer - 文本空间的Adam优化器
        
        将Adam优化器的核心概念映射到文本优化：
        - 一阶动量：梯度反馈的语义方向一致性
        - 二阶动量：改进效果的历史方差
        - 自适应学习率：基于历史表现调整优化强度
        
        :param parameters: 需要优化的变量列表
        :param engine: 用于更新变量的引擎
        :param beta1: 一阶动量衰减率，控制梯度方向记忆
        :param beta2: 二阶动量衰减率，控制方差记忆  
        :param epsilon: 数值稳定性参数
        :param momentum_window: 动量窗口大小
        """
        super().__init__(parameters)
        
        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]
            
        self.engine = validate_engine_or_get_default(engine)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum_window = momentum_window
        self.verbose = verbose
        
        # 为每个参数初始化Adam状态
        self.step_count = 0
        self.first_moment_storage = [[] for _ in range(len(parameters))]   # 一阶动量存储
        self.second_moment_storage = [[] for _ in range(len(parameters))]  # 二阶动量存储
        self.improvement_history = [[] for _ in range(len(parameters))]    # 改进效果历史
        
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0], 
            new_variable_end_tag=new_variable_tags[1]
        )
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        
        logger.info(f"TextualAdam initialized with beta1={beta1}, beta2={beta2}, window={momentum_window}")

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似性（简化版本）
        在实际应用中可以使用更复杂的语义嵌入模型
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _update_first_moment(self, param_idx: int, gradient_text: Union[str, List[Union[str, bytes]]]):
        """更新一阶动量 - 梯度语义方向的指数移动平均"""
        storage = self.first_moment_storage[param_idx]
        
        # 处理gradient_text可能是字符串或列表的情况
        if isinstance(gradient_text, list):
            gradient_str = "\n".join([str(item) for item in gradient_text if isinstance(item, str)])
        else:
            gradient_str = str(gradient_text)
        
        if len(storage) == 0:
            # 第一次，直接存储
            storage.append({
                'gradient_semantic': gradient_str,
                'momentum_value': 1.0
            })
        else:
            # 计算与历史梯度的语义相似性
            recent_gradients = [item['gradient_semantic'] for item in storage[-3:]]  
            similarities = [self._compute_semantic_similarity(gradient_str, grad) 
                          for grad in recent_gradients]
            avg_similarity = sum(similarities) / len(similarities)
            
            # 更新动量值：相似性高说明方向一致，动量增强
            new_momentum = self.beta1 * storage[-1]['momentum_value'] + (1 - self.beta1) * avg_similarity
            
            storage.append({
                'gradient_semantic': gradient_str,
                'momentum_value': new_momentum,
                'semantic_consistency': avg_similarity
            })
            
        # 保持窗口大小
        if len(storage) > self.momentum_window:
            storage.pop(0)

    def _update_second_moment(self, param_idx: int, improvement_score: float):
        """更新二阶动量 - 改进效果方差的指数移动平均"""
        storage = self.second_moment_storage[param_idx]
        
        if len(storage) == 0:
            storage.append({
                'improvement_score': improvement_score,
                'variance_estimate': improvement_score ** 2
            })
        else:
            # 计算改进效果的方差
            recent_scores = [item['improvement_score'] for item in storage]
            mean_score = sum(recent_scores) / len(recent_scores)
            variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(recent_scores)
            
            # 更新二阶动量
            new_variance = self.beta2 * storage[-1]['variance_estimate'] + (1 - self.beta2) * (improvement_score ** 2)
            
            storage.append({
                'improvement_score': improvement_score,
                'variance_estimate': new_variance,
                'stability_measure': 1.0 / (variance + self.epsilon)  # 稳定性度量
            })
            
        # 保持窗口大小
        if len(storage) > self.momentum_window:
            storage.pop(0)

    def _estimate_improvement_score(self, old_value: str, new_value: str, gradient_text: Union[str, List[Union[str, bytes]]]) -> float:
        """估算改进分数（简化版本）
        实际应用中可以使用更复杂的评估机制
        """
        # 确保输入是字符串
        old_str = str(old_value)
        new_str = str(new_value)
        grad_str = str(gradient_text) if isinstance(gradient_text, str) else "\n".join([str(item) for item in gradient_text if isinstance(item, str)])
        
        # 基于语义变化等指标
        semantic_change = 1 - self._compute_semantic_similarity(old_str, new_str)
        
        # 如果梯度提到"改进"、"更好"等词，给予正分
        positive_indicators = ['improve', 'better', 'enhance', 'good', 'excellent']
        negative_indicators = ['worse', 'bad', 'unclear', 'confusing', 'error']
        
        gradient_lower = grad_str.lower()
        pos_score = sum(1 for word in positive_indicators if word in gradient_lower)
        neg_score = sum(1 for word in negative_indicators if word in gradient_lower)
        
        sentiment_score = (pos_score - neg_score) / max(len(grad_str.split()), 1)
        
        return semantic_change * 0.5 + sentiment_score * 0.5

    def step(self):
        """执行Adam优化步骤"""
        self.step_count += 1
        
        for param_idx, parameter in enumerate(self.parameters):
            old_value = parameter.value
            gradient_text = get_gradient_and_context_text(parameter)
            
            # 更新一阶动量
            self._update_first_moment(param_idx, gradient_text)
            
            # 生成优化提示
            prompt_update_parameter = self._create_adam_prompt(parameter, param_idx)
            
            # 获取优化后的文本
            response = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            new_text = str(response) if response is not None else ""
            logger.info(f"TextualAdam optimizer response", extra={"optimizer.response": new_text})
            
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            except IndexError:
                logger.error(f"TextualAdam optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualAdam optimizer response could not be indexed. Response: {new_text}")
            
            # 估算改进分数并更新二阶动量
            improvement_score = self._estimate_improvement_score(str(old_value), new_value, gradient_text)
            self._update_second_moment(param_idx, improvement_score)
            
            # 更新参数值
            parameter.set_value(new_value)
            logger.info(f"TextualAdam updated text", extra={"parameter.value": parameter.value})


    def _create_adam_prompt(self, variable: Variable, param_idx: int) -> str:
        """创建Adam特定的优化提示"""
        # 获取动量信息
        first_moment_info = self._get_first_moment_text(param_idx)
        second_moment_info = self._get_second_moment_text(param_idx)
        
        # 计算偏差修正后的学习率调整
        bias_correction1 = 1 - (self.beta1 ** self.step_count)
        bias_correction2 = 1 - (self.beta2 ** self.step_count)
        
        # 获取自适应强度建议
        adaptive_intensity = self._get_adaptive_intensity(param_idx, bias_correction1, bias_correction2)
        
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "first_moment_info": first_moment_info,
            "second_moment_info": second_moment_info,
            "adaptive_intensity": adaptive_intensity,
            "step_count": self.step_count,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples)
        }
        
        # 构造Adam特定的提示
        prompt = self._construct_adam_prompt(**optimizer_information)
        
        logger.info(f"TextualAdam prompt for update", extra={"prompt": prompt})
        return prompt

    def _get_first_moment_text(self, param_idx: int) -> str:
        """获取一阶动量信息的文本描述"""
        storage = self.first_moment_storage[param_idx]
        if not storage:
            return "No momentum history available."
        
        recent_momentum = storage[-1]
        momentum_value = recent_momentum['momentum_value']
        
        if momentum_value > 0.7:
            consistency_desc = "high consistency in feedback direction"
        elif momentum_value > 0.4:
            consistency_desc = "moderate consistency in feedback direction"
        else:
            consistency_desc = "low consistency in feedback direction"
            
        return f"Gradient momentum shows {consistency_desc} (strength: {momentum_value:.2f})"

    def _get_second_moment_text(self, param_idx: int) -> str:
        """获取二阶动量信息的文本描述"""
        storage = self.second_moment_storage[param_idx]
        if not storage:
            return "No variance history available."
        
        recent_moment = storage[-1]
        stability = recent_moment.get('stability_measure', 1.0)
        
        if stability > 2.0:
            stability_desc = "high stability in improvements"
        elif stability > 1.0:
            stability_desc = "moderate stability in improvements"
        else:
            stability_desc = "low stability in improvements"
            
        return f"Improvement history shows {stability_desc} (stability: {stability:.2f})"

    def _get_adaptive_intensity(self, param_idx: int, bias_correction1: float, bias_correction2: float) -> str:
        """获取自适应优化强度建议"""
        if not self.first_moment_storage[param_idx] or not self.second_moment_storage[param_idx]:
            return "Apply moderate optimization intensity."
        
        momentum_strength = self.first_moment_storage[param_idx][-1]['momentum_value']
        stability = self.second_moment_storage[param_idx][-1].get('stability_measure', 1.0)
        
        # 应用偏差修正
        corrected_momentum = momentum_strength / bias_correction1
        corrected_stability = stability / bias_correction2
        
        # 自适应强度判断
        if corrected_momentum > 0.6 and corrected_stability > 1.5:
            return "Apply strong optimization with high confidence due to consistent and stable improvements."
        elif corrected_momentum > 0.4 or corrected_stability > 1.0:
            return "Apply moderate optimization intensity with careful attention to feedback."
        else:
            return "Apply conservative optimization due to inconsistent or unstable improvement history."

    def _construct_adam_prompt(self, **kwargs) -> str:
        """构造Adam特定的优化提示"""
        base_prompt = (
            f"Here is the role of the variable you will improve: <ROLE>{kwargs['variable_desc']}</ROLE>.\n\n"
            f"The variable is the text within the following span: <VARIABLE> {kwargs['variable_short']} </VARIABLE>\n\n"
            f"Here is the context and feedback we got for the variable:\n\n"
            f"<CONTEXT>{kwargs['variable_grad']}</CONTEXT>\n\n"
        )
        
        # 添加Adam特定信息
        adam_info = (
            f"## Adam Optimization Context (Step {kwargs['step_count']}):\n"
            f"**Momentum Analysis**: {kwargs['first_moment_info']}\n"
            f"**Stability Analysis**: {kwargs['second_moment_info']}\n"
            f"**Adaptive Strategy**: {kwargs['adaptive_intensity']}\n\n"
        )
        
        optimization_instruction = (
            f"Based on the Adam optimization analysis above, improve the variable ({kwargs['variable_desc']}) "
            f"using the feedback provided in <FEEDBACK> tags. The optimization strategy has been "
            f"automatically adapted based on the consistency of past feedback and stability of improvements.\n\n"
        )
        
        if self.do_constrained:
            optimization_instruction += f"You must follow the following constraints:\n\n<CONSTRAINTS>{kwargs['constraint_text']}</CONSTRAINTS>\n\n"
        
        if self.do_in_context_examples:
            optimization_instruction += f"You must base on the following examples when modifying the {kwargs['variable_desc']}:\n\n<EXAMPLES>{kwargs['in_context_examples']}</EXAMPLES>\n\n"
        
        suffix = (
            f"Send the improved variable in the following format:\n\n"
            f"{kwargs['new_variable_start_tag']}{{the improved variable}}{kwargs['new_variable_end_tag']}\n\n"
            f"Send ONLY the improved variable between the tags, and nothing else."
        )
        
        return base_prompt + adam_info + optimization_instruction + suffix
