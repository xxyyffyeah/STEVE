import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task

import numpy as np
import random

from collections import defaultdict
import logging
import io

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--test_engine", type=str, default="gpt-3.5-turbo-0125", help="The API to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--do_not_run_larger_model", action="store_true", help="Whether to run the larger model or not.")
    parser.add_argument("--num_threads", type=int, default=32, help="The number of threads to use for evaluation.")
    return parser.parse_args()

args = config()

def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        accuracy = int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        accuracy = int(eval_output_parsed)
    # Return detailed information including input, prediction, and accuracy
    return {
        'accuracy': accuracy,
        'input': x.value,
        'ground_truth': y.value,
        'prediction': response.value,
        'eval_output': eval_output_variable.value if hasattr(eval_output_variable, 'value') else str(eval_output_variable)
    }
    

def eval_dataset(test_set, eval_fn, model, max_samples: int=None, return_details: bool=False):
    if max_samples is None:
        max_samples = len(test_set)
    results_list = []
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            result = future.result()
            results_list.append(result)
            accuracy_list.append(result['accuracy'])
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    
    if return_details:
        return results_list
    else:
        return accuracy_list 

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


set_seed(args.seed)
llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
llm_api_test = tg.get_engine(engine_name=args.test_engine)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(STARTING_SYSTEM_PROMPT)


# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True, 
                            role_description="system prompt to the language model")
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

# if not args.do_not_run_larger_model:
#     reference = np.mean(eval_dataset(test_set, eval_fn, model_evaluation))

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
        requires_grad=True,
        role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")

model = tg.BlackboxLLM(llm_api_test, system_prompt)

results = {"test_acc": [], "prompt": [], "validation_acc": [], "rank": []}
# Get initial test results with details for comparison
initial_test_results = eval_dataset(test_set, eval_fn, model, return_details=True)
initial_test_acc = [r['accuracy'] for r in initial_test_results]
# results["test_acc"].append(initial_test_acc)
# results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

# Store initial results for comparison using input as key
initial_results_dict = {r['input']: r for r in initial_test_results}


for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                requires_grad=True,
                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")

        model = tg.BlackboxLLM(llm_api_test, system_prompt)
        optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
        losses = []
        responses = []
        for (x, y) in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            responses.append((x, y, response))
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        
        # Capture backward pass information
        backward_info = {
            "gradients_before": str(system_prompt.gradients).split("\n") if hasattr(system_prompt, 'gradients') else [],
            "loss_value": str(total_loss.value) if hasattr(total_loss, 'value') else None
        }
        
        # Create a custom logger handler to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        tg.logger.addHandler(handler)
        
        # Run backward
        total_loss.backward()
        
        # Capture backward logs
        backward_info["logs"] = log_capture.getvalue()
        backward_info["gradients_after"] = str(system_prompt.gradients).split("\n") if hasattr(system_prompt, 'gradients') else []
        
        # Reset log capture for step
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        tg.logger.handlers = [handler]
        
        # Run optimizer step
        optimizer.step()
        
        # Capture step logs
        step_info = {
            "logs": log_capture.getvalue(),
            "new_prompt": system_prompt.get_value()
        }
        
        # Remove the handler
        tg.logger.handlers = []
        # if args.run_validation:
        #     run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        # print("sys prompt: ", system_prompt)
        test_results = eval_dataset(test_set, eval_fn, model, return_details=True)
        test_acc = [r['accuracy'] for r in test_results]
        # results["test_acc"].append(test_acc)
        # results["prompt"].append(system_prompt.get_value())
        # Compare with initial results to find improved and degraded cases
        improved_cases = []  # Cases that were wrong initially but correct now
        degraded_cases = []  # Cases that were correct initially but wrong now
        
        for test_result in test_results:
            # Use input as key to match with initial results
            test_input = test_result['input']
            initial_result = initial_results_dict.get(test_input, {})
            initial_acc = initial_result.get('accuracy', 0)
            current_acc = test_result['accuracy']
            
            # Case improved: was wrong (0) now correct (1)
            if initial_acc == 0 and current_acc == 1:
                improved_json = {
                    "eval_output_before": initial_result.get('eval_output', ''),
                    "eval_output_after": test_result['eval_output'],
                    "input": test_result['input'],
                    "ground_truth": test_result['ground_truth'],
                    "prediction_before": initial_result.get('prediction', ''),
                    "prediction_after": test_result['prediction']
                }
                improved_cases.append(improved_json)
            
            # Case degraded: was correct (1) now wrong (0)
            elif initial_acc == 1 and current_acc == 0:
                degraded_json = {
                    "eval_output_before": initial_result.get('eval_output', ''),
                    "eval_output_after": test_result['eval_output'],
                    "input": test_result['input'],
                    "ground_truth": test_result['ground_truth'],
                    "prediction_before": initial_result.get('prediction', ''),
                    "prediction_after": test_result['prediction']
                }
                degraded_cases.append(degraded_json)
        results["rank"].append({
            "mean_accuracy": np.mean(test_acc),
            "initial_accuracy": np.mean(initial_test_acc),
            "eval_output": eval_output_variable.value,
            "system_prompt": str(system_prompt.get_value()).split("."),
            "training_sample": {
                "inputs": [x[0].value for x in responses],
                "ground_truths": [y[1].value for y in responses],
                "predictions": [y[2].value for y in responses]
            },
            "improved_cases": improved_cases,  # Cases that improved
            "degraded_cases": degraded_cases,  # Cases that degraded
            "num_improved": len(improved_cases),
            "num_degraded": len(degraded_cases),
            "backward_info": backward_info,
            "step_info": step_info
        })

        if steps == 50:
            break
results["rank"].sort(key=lambda x: x["mean_accuracy"], reverse=True)
# Also dump the final results
import json
with open(f"./figures/results_{args.task}_{args.test_engine}.json", "w") as f:
    json.dump(results, f)