import re
import platformdirs
from textgrad.variable import Variable
from textgrad.loss import MultiFieldTokenParsedEvaluation, MultiChoiceTestTime
from .base import Dataset


def eval_string_based(response_text, correct_answer):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == correct_answer else 0.0
    return score


QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


class LogiQA2(Dataset):
    def __init__(self, root: str = None, split: str = "train", *args, **kwargs):
        """
        LogiQA 2.0 dataset from HF.
        """
        from datasets import load_dataset
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        assert split in ["train", "validation", "test"]
        self.data = load_dataset("csitfun/LogiQA2.0", cache_dir=root, split=split)
        self.split = split
        self._task_description = 'You will answer logical reasoning multiple-choice questions. Think step by step.'
            
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        choices = row["options"]
        answer = row["answer"]
        
        # Convert answer index to letter
        if isinstance(answer, int):
            answer_letter = chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
        else:
            # If answer is already a letter, use it directly
            answer_letter = answer.upper()
        
        # Format choices as A, B, C, D
        choices_dict = {
            "Question": question,
            "A": choices[0],
            "B": choices[1], 
            "C": choices[2],
            "D": choices[3]
        }
        
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        return question_prompt, answer_letter

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return "Given a logical reasoning multiple choice question, the goal is to select the correct final answer from the choices."


class LogiQA2InstanceDataset(LogiQA2):
    def __init__(self, evaluation_api, root: str = None, split: str = "train", max_samples=-1):
        super().__init__(root, split, max_samples)
        self.evaluation_api = evaluation_api

        
    def _get_instance_test_time_objective(self, question: str):
        evaluation_instruction = "Below is a logical reasoning question and an answer. You are an expert in logical reasoning. Your job is to investigate the answer. Critically go through the reasoning steps, consider logical principles, and see if the answer is correct or if there are any critical mistakes in the logical reasoning."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        return test_time_objective
        

    def _get_instance_eval_fn(self, question_prompt: str, answer: str):
        eval_string_based_fn = lambda response: eval_string_based(response.value, answer)
        return eval_string_based_fn
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        choices = row["options"]
        answer = row["answer"]
        
        # Convert answer index to letter
        if isinstance(answer, int):
            answer_letter = chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
        else:
            answer_letter = answer.upper()
        
        choices_dict = {
            "Question": question,
            "A": choices[0],
            "B": choices[1],
            "C": choices[2], 
            "D": choices[3]
        }
        
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        
        return question_prompt, answer_letter, self._get_instance_test_time_objective(question_prompt), self._get_instance_eval_fn(question_prompt, answer_letter)

    def get_default_task_instruction(self):
        return "Given a logical reasoning multiple choice question, the goal is to select the correct final answer from the choices."