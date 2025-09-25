import re
import platformdirs
from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime
from .base import Dataset


def parse_boolean_answer(answer_text):
    ANSWER_PATTERN_BOOLEAN = r"(?i)Answer\s*:\s*(True|False|Yes|No)"
    
    match = re.search(ANSWER_PATTERN_BOOLEAN, answer_text)
    if match:
        extracted_answer = match.group(1).lower()
        # Normalize to boolean
        if extracted_answer in ['true', 'yes']:
            return True
        elif extracted_answer in ['false', 'no']:
            return False
    return None

def string_based_equality_fn(prediction: Variable, ground_truth_answer: Variable):
    pred_bool = parse_boolean_answer(str(prediction.value))
    # Convert string ground truth back to boolean for comparison
    gt_str = str(ground_truth_answer.value)
    gt_bool = gt_str == "True" if isinstance(ground_truth_answer.value, str) else ground_truth_answer.value
    return int(pred_bool == gt_bool if pred_bool is not None else 0)



class StrategyQA(Dataset):
    def __init__(self, root: str = None, split: str = "train"):
        """
        StrategyQA dataset from HF.
        """
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        # Allow train split with slice notation
        if not (split == "train" or split.startswith("train[")):
            raise ValueError(f"Only 'train' split (with optional slice) available for StrategyQA, got {split}")
        self.data = load_dataset("tasksource/strategy-qa", cache_dir=root, split=split)


        self.split = split
        self._task_description =  "Answer the following yes/no question. Think step by step and provide reasoning before answering. The last line of your response should be of the following format: 'Answer: True' or 'Answer: False'."

            
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        facts = row["facts"]

        answer = "True" if row["answer"] else "False"
        question_prompt = question + "\n" + "Here are some facts that might be useful: " 
        for fact in facts:
            question_prompt += "\n" + fact
        
        return question_prompt, answer

    def __len__(self):
        return len(self.data)
    
    def get_task_description(self):
        return self._task_description 
    
    def get_default_task_instruction(self):
        return "Given a strategic reasoning question, the goal is to determine if the answer is True or False."


class StrategyQAInstanceDataset(StrategyQA):
    def __init__(self, evaluation_api, root: str = None, split: str = "train"):
        super().__init__(root, split)
        self.evaluation_api = evaluation_api

        
    def _get_instance_test_time_objective(self, question: str):
        evaluation_instruction = "Below is a strategic reasoning question and an answer. You are an expert in strategic reasoning and logical thinking. Your job is to investigate the answer. Critically go through the reasoning steps, consider factual accuracy, and see if the answer is correct or if there are any critical mistakes in the reasoning."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        return test_time_objective
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        answer = "True" if row["answer"] else "False"
        
        
        return question, answer, self._get_instance_test_time_objective(question_prompt)

    def get_default_task_instruction(self):
        return "Given a strategic reasoning question, the goal is to determine if the answer is True or False."

