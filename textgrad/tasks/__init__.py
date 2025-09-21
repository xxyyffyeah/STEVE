from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader
from .leetcode import LeetCodeHardEval
from .logiqa2 import LogiQA2, LogiQA2InstanceDataset
from .strategyqa import StrategyQA, StrategyQAInstanceDataset
from .date_understanding import DateUnderstanding, DateUnderstandingInstanceDataset

from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM

AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_word_sorting",
    "GSM8K_DSPy",
    "StrategyQA",
    "DateUnderstanding",
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond",
    "LeetCodeHardEval",
    "LogiQA2",
    "StrategyQA",
    "DateUnderstanding",
]

def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    """
    Args:
        task_name: the name of the task to evaluate
        evaluation_api: the engine to use for evaluation, if needed
    """
    if "object_counting" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        
        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "GSM8K_DSPy":
        from textgrad.tasks.gsm8k import GSM8K_DSPy
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        train_set = GSM8K_DSPy(split="train", *args, **kwargs)
        val_set = GSM8K_DSPy(split="val", *args, **kwargs)
        test_set = GSM8K_DSPy(split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "StrategyQA":
        from textgrad.tasks.strategyqa import StrategyQA, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        
        train_set = StrategyQA(split="train[:50]", *args, **kwargs)
        val_set = StrategyQA(split="train[50:150]", *args, **kwargs)  
        test_set = StrategyQA(split="train[150:250]", *args, **kwargs)
        
        fn_purpose = "String-based function that extracts Answer: True/False from model response and compares with ground truth boolean."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    elif task_name.startswith("MMLU_"):
        # Global prompt optimization for specific MMLU subset (e.g., MMLU_machine_learning)
        from textgrad.tasks.mmlu import MMLU
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        subset = task_name[5:]

        # Use native MMLU splits (train/dev, validation, test) without re-splitting
        train_set = MMLU(subset=subset, split="train", *args, **kwargs)
        val_set = MMLU(subset=subset, split="validation", *args, **kwargs)
        test_set = MMLU(subset=subset, split="test", *args, **kwargs)

        # Load a base split with sufficient size, then create sliced wrappers (since MMLU doesn't accept slice notation)
        base = MMLU(subset=subset, split="test", *args, **kwargs)
        n = len(base)
        # 60/20/20 split with minimal safeguards
        n_train = max(1, int(0.6 * n))
        n_val = max(1, int(0.2 * n)) if n - n_train >= 2 else max(0, n - n_train - 1)
        n_test = max(1, n - n_train - n_val)

        class _SlicedDataset(Dataset):
            def __init__(self, base_ds, start, end):
                self.base = base_ds
                self.start = int(start)
                self.end = int(end)
                # Reuse task description if available
                if hasattr(base_ds, "get_task_description"):
                    self._task_description = base_ds.get_task_description()
                elif hasattr(base_ds, "get_default_task_instruction"):
                    self._task_description = base_ds.get_default_task_instruction()
                else:
                    self._task_description = "You will answer multiple-choice questions. Think step by step."

            def __len__(self):
                return max(0, self.end - self.start)

            def __getitem__(self, idx):
                return self.base[self.start + int(idx)]

            def get_task_description(self):
                return self._task_description

        train_set = _SlicedDataset(base, 0, n_train)
        val_set = _SlicedDataset(base, n_train, n_train + n_val)
        test_set = _SlicedDataset(base, n_train + n_val, n)

        role_descriptions = [
            "Question for the task",
            "Ground truth answer (letter A-D)",
            "Prediction from the language model"
        ]
        evaluation_instruction = (
            "Below is a multiple-choice question, the ground truth answer letter, and a model prediction. "
            "Is the model's final answer correct (same letter as ground truth)? Say only 1 (yes) or 0 (no). "
            "Return your response within <ACCURACY> </ACCURACY> tags. e.g. <ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        )
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        return train_set, val_set, test_set, eval_fn
    elif task_name == "DateUnderstanding":
        from textgrad.tasks.date_understanding import DateUnderstanding
        from textgrad.autograd.string_based_ops import StringBasedFunction
        
        def string_based_letter_equality(prediction: Variable, ground_truth_answer: Variable):
            import re
            m = re.search(r"(?i)Answer\s*:\s*([A-F])", str(prediction.value))
            pred = m.group(1) if m else None
            return int(pred is not None and pred.upper() == str(ground_truth_answer.value).upper())

        train_set = DateUnderstanding(split="test[:50]", *args, **kwargs)
        val_set = DateUnderstanding(split="test[50:150]", *args, **kwargs)
        test_set = DateUnderstanding(split="test[150:250]", *args, **kwargs)
        fn_purpose = "String-based function that extracts Answer: [A-F] from model response and compares with ground truth letter."
        eval_fn = StringBasedFunction(string_based_letter_equality, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    else:
        raise ValueError(f"Task {task_name} not found.")


def load_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if "MMLU_" in task_name:
        subset = task_name[5:]
        test_set = MMLUInstanceDataset(evaluation_api=evaluation_api, subset=subset, split="test", *args, **kwargs)
        return test_set
    elif "GPQA" in task_name:
        from .gpqa import GPQAInstanceDataset
        test_set = GPQAInstanceDataset(evaluation_api=evaluation_api, subset=task_name.lower(), *args, **kwargs)
        return test_set
    elif task_name in ["LeetCodeHardEval"]:
        dataset = LeetCodeHardEval()
        return dataset
    elif task_name == "LogiQA2":
        test_set = LogiQA2InstanceDataset(evaluation_api=evaluation_api, split="test", *args, **kwargs)
        return test_set
    elif task_name == "StrategyQA":
        test_set = StrategyQAInstanceDataset(evaluation_api=evaluation_api, split="train", *args, **kwargs)
        return test_set
    elif task_name == "DateUnderstanding":
        from .date_understanding import DateUnderstandingInstanceDataset
        test_set = DateUnderstandingInstanceDataset(evaluation_api=evaluation_api, split="test", *args, **kwargs)
        return test_set
    else:
        raise ValueError(f"Instance task {task_name} not found.")
