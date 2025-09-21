import re
import platformdirs
from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime
from .base import Dataset


def extract_letter_from_answer_text(answer_text: str):
    pattern = r"(?i)Answer\s*:\s*([A-F])"
    m = re.search(pattern, answer_text)
    return m.group(1) if m else None

def build_mc_prompt(raw_input_text: str):
    """Prepend a fixed header that constrains answers to A–F.
    Assumes the raw input already contains the question and Options: (A)…(F)…
    """
    header = (
        "Answer the following multiple choice question. Think step by step, but on the FINAL LINE output exactly one uppercase letter from A–F in the format: 'Answer: X'. "
        "X must be one of A, B, C, D, E, or F. The final line must contain ONLY 'Answer: X' with no parentheses, punctuation, or extra text after the letter."
    )
    prompt = f"{header}\n\n{raw_input_text.strip()}"
    return prompt


class DateUnderstanding(Dataset):
    def __init__(self, root: str = None, split: str = "test"):
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        
        self.root = root
        
        self.data = load_dataset("lukaemon/bbh", "date_understanding", cache_dir=root, split=split)
        self.split = split
        self._task_description = (
            "Answer the following multiple choice question. Think step by step. The last line must be 'Answer: $LETTER'. $LETTER must be one of A, B, C, D, E, or F."
        )

    def __getitem__(self, index):
        row = self.data[index]
        raw_input = row.get("input")
        target = row.get("target")


        m = re.search(r"\(?([A-F])\)?", str(target), flags=re.I)
        answer_letter = m.group(1).upper()
        question_prompt = build_mc_prompt(raw_input)
        return question_prompt, answer_letter

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return self._task_description

    def get_default_task_instruction(self):
        return "Given a date reasoning question, the goal is to select the correct answer letter from A to F."


class DateUnderstandingInstanceDataset(DateUnderstanding):
    def __init__(self, evaluation_api, root: str = None, split: str = "train"):
        super().__init__(root, split)
        self.evaluation_api = evaluation_api

    def _get_instance_test_time_objective(self, question_prompt: str):
        evaluation_instruction = (
            "You are an expert in date reasoning. Given a multiple-choice question and a proposed answer, think step by step to verify correctness. "
            "Carefull read the question, and check whether the final letter matches the correct option. "
            "Your job is to investigate the answer. Critically go through the reasoning steps, consider factual accuracy, and see if the answer is correct or if there are any critical mistakes in the reasoning."
        )
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)

        def test_time_objective(instance: Variable):
            return eval_fn(question_prompt, instance)

        return test_time_objective

    def _get_instance_eval_fn(self, answer_letter: str):
        def instance_eval_fn(instance: Variable):
            pred = extract_letter_from_answer_text(str(instance.value))
            return 1 if (pred is not None and pred.upper() == str(answer_letter).upper()) else 0

        return instance_eval_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        raw_input = row.get("input")
        target = row.get("target")
        m = re.search(r"\(?([A-F])\)?", str(target), flags=re.I)
        answer_letter = m.group(1).upper()
        question_prompt = build_mc_prompt(raw_input)
        test_time_objective = self._get_instance_test_time_objective(question_prompt)
        instance_eval_fn = self._get_instance_eval_fn(answer_letter)
        # 返回格式与 solution_optimization.py 期望一致
        return question_prompt, answer_letter, test_time_objective, instance_eval_fn
