import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task

import numpy as np
import random
import json

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
    parser.add_argument("--preserve_sample_size", type=int, default=20, help="Size of preserve sample for gating test.")
    parser.add_argument("--lambda_gating", type=float, default=1.5, help="Lambda parameter for gating rule.")
    parser.add_argument("--n_hard_examples", type=int, default=3, help="Number of hard examples to generate candidates each round.")
    return parser.parse_args()

args = config()

def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)
    

def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
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
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
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

# Initialize system prompt and model
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(llm_api_test, system_prompt)

# Convert train_set to list for easier manipulation  
train_set_list = []
for sample in train_set:
    train_set_list.append(sample)
print(f"Total training examples: {len(train_set_list)}")

def update_train_pools(current_model, eval_fn):
    """Dynamically update train_set_preserve and train_set_hard_pool based on current model performance using multithreading"""
    preserve_cases = []  # Cases where eval output = 1
    hard_cases = []     # Cases where eval output = 0
    
    print(f"\nRe-evaluating {len(train_set_list)} training cases with current prompt...")
    
    # Get current prompt
    current_prompt = current_model.system_prompt.get_value()
    
    # Use multithreaded evaluation
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for sample in train_set_list:
            future = executor.submit(evaluate_single_sample, sample, current_prompt, llm_api_test, eval_fn)
            futures.append((future, sample))
        
        for future, sample in tqdm(futures, desc="Evaluating training set"):
            result = future.result()
            if result == 1:
                preserve_cases.append(sample)
            else:
                hard_cases.append(sample)
    
    print(f"Current performance: {len(preserve_cases)} correct, {len(hard_cases)} incorrect")
    return preserve_cases, hard_cases

# Initial classification
train_set_preserve, train_set_hard_pool = update_train_pools(model, eval_fn)

train_loader_hard = tg.tasks.DataLoader(train_set_hard_pool, batch_size=args.batch_size, shuffle=True)
print(STARTING_SYSTEM_PROMPT)


# STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Follow these steps to ensure clarity and accuracy:\n\n1. **Identify and List**: Begin by identifying and listing each item mentioned in the query, categorizing them appropriately.\n\n2. **Step-by-Step Reasoning**: Clearly articulate each step of your reasoning process. Ensure that every component of the problem is explicitly mentioned and accounted for.\n\n3. **Verification**: After calculating the answer, re-check your arithmetic to confirm the result is accurate. Verify each step to ensure consistency with the query.\n\n4. **Contextual Awareness**: Consider all relevant details from the query and ensure they are accurately reflected in your response. Use the same terminology and numbers as provided in the query to maintain consistency.\n\n5. **Assumption Transparency**: If any assumptions are made, clearly state and justify them to ensure transparency.\n\n6. **Structured Format**: Present your reasoning in a clear, structured format. Conclude with the final answer in the specified format: 'Answer: $VALUE' where VALUE is a numerical value.\n\nBy following these guidelines, your response will be clear, accurate, and consistent."


optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": [], "validation_acc": [], "rank": []}
# Get initial test accuracy
initial_test_acc = eval_dataset(test_set, eval_fn, model)
results["test_acc"].append([int(x) for x in initial_test_acc])  # Convert numpy types to int
# results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())


def evaluate_single_sample(sample, prompt, model_engine, eval_fn):
    """Evaluate a single sample with given prompt"""
    x, y = sample
    # Create temporary model with given prompt
    temp_prompt = tg.Variable(prompt, requires_grad=False, role_description="temporary prompt for evaluation")
    temp_model = tg.BlackboxLLM(model_engine, temp_prompt)
    
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y_var = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = temp_model(x_var)
    
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x_var, y_var, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)

def evaluate_on_sample_set(sample_set, prompt, model_engine, eval_fn):
    """Evaluate model performance on a sample set with given prompt using multithreading"""
    if len(sample_set) == 0:
        return 0.0
    
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for sample in sample_set:
            future = executor.submit(evaluate_single_sample, sample, prompt, model_engine, eval_fn)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            acc_item = future.result()
            accuracy_list.append(acc_item)
    
    return sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0.0

for epoch in range(args.max_epochs):
    for steps in range(50):  # Limit to 50 steps
        print(f"\n=== Step {steps+1} ===\n")
        
        # Dynamic re-classification of training examples based on current prompt
        if steps > 0:  # Skip for first step since we already classified initially
            train_set_preserve, train_set_hard_pool = update_train_pools(model, eval_fn)
        
        # Check if we have any hard cases to work with
        if len(train_set_hard_pool) == 0:
            print("🎉 No more hard cases! All training examples are solved.")
            break
        
        # Step 1: Generate multiple candidate prompts using n hard examples
        current_prompt = system_prompt.get_value()
        candidates = []
        
        # Sample n hard cases from D_hard_pool  
        n_samples = min(args.n_hard_examples, len(train_set_hard_pool))
        if n_samples == 0:
            print("No hard cases available, skipping this step.")
            continue
        hard_samples = random.sample(train_set_hard_pool, n_samples)
        
        print(f"Generating {n_samples} candidates from {n_samples} hard cases...")
        
        for i, (d_hard_x, d_hard_y) in enumerate(hard_samples):
            print(f"\n--- Candidate {i+1} from hard case: {d_hard_x[:100]}... ---")
            
            # Reset to current prompt for each candidate generation
            system_prompt.set_value(current_prompt)
            
            # Run current prompt on hard case
            x_var = tg.Variable(d_hard_x, requires_grad=False, role_description="query to the language model")
            y_var = tg.Variable(d_hard_y, requires_grad=False, role_description="correct answer for the query")
            response = model(x_var)
            
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
            except:
                eval_output_variable = eval_fn([x_var, y_var, response])
            
            print(f"Current performance: {eval_output_variable.value}")
            
            if eval_output_variable.value == "1":
                print("Already solved, using current prompt as candidate")
                candidates.append({
                    'prompt': current_prompt,
                    'hard_case': (d_hard_x, d_hard_y),
                    'source': 'current_prompt'
                })
                continue
            
            # Generate gradient and create candidate
            optimizer.zero_grad()
            eval_output_variable.backward()
            optimizer.step()
            
            candidate_prompt = system_prompt.get_value()
            candidates.append({
                'prompt': candidate_prompt,
                'hard_case': (d_hard_x, d_hard_y),
                'source': 'optimized'
            })
            
            print(f"Generated candidate (first 150 chars): {candidate_prompt[:150]}...")
        
        # Step 2: Evaluate all candidates and select the best one
        print(f"\n--- Evaluating {len(candidates)} candidates ---")
        
        # Sample from D_preserve for testing
        if len(train_set_preserve) >= args.preserve_sample_size:
            preserve_sample = random.sample(train_set_preserve, args.preserve_sample_size)
        else:
            preserve_sample = train_set_preserve
        
        print(f"Testing on {len(preserve_sample)} preserve cases...")
        
        best_candidate = None
        best_score = float('-inf')
        best_metrics = {}
        candidate_results = []
        
        # Evaluate current prompt as baseline
        score_old = evaluate_on_sample_set(preserve_sample, current_prompt, llm_api_test, eval_fn)
        improvement_old = evaluate_on_sample_set(train_set_hard_pool, current_prompt, llm_api_test, eval_fn)
        
        for i, candidate in enumerate(candidates):
            candidate_prompt = candidate['prompt']
            
            # Evaluate on preserve set
            score_candidate = evaluate_on_sample_set(preserve_sample, candidate_prompt, llm_api_test, eval_fn)
            regression = score_old - score_candidate
            
            # Evaluate improvement on all hard cases
            improvement_candidate = evaluate_on_sample_set(train_set_hard_pool, candidate_prompt, llm_api_test, eval_fn)
            improvement = improvement_candidate - improvement_old
            
            # Calculate selection score: improvement - lambda * regression
            selection_score = improvement - args.lambda_gating * regression
            
            candidate_metrics = {
                'candidate_id': i + 1,
                'regression': regression,
                'improvement': improvement,
                'selection_score': selection_score,
                'preserve_score': score_candidate,
                'hard_score': improvement_candidate,
                'source': candidate['source']
            }
            candidate_results.append(candidate_metrics)
            
            print(f"Candidate {i+1}: Score={selection_score:.4f} (Imp={improvement:.2%}, Reg={regression:.2%})")
            
            if selection_score > best_score:
                best_score = selection_score
                best_candidate = candidate
                best_metrics = candidate_metrics
        
        # Step 3: Apply the best candidate
        print(f"\n--- Best candidate: #{best_metrics['candidate_id']} with score {best_score:.4f} ---")
        
        if best_candidate and best_score > 0:
            print("✓ Accepting best candidate")
            system_prompt.set_value(best_candidate['prompt'])
            accepted = True
            final_prompt = best_candidate['prompt']
        else:
            print("✗ No good candidate found, keeping current prompt")
            system_prompt.set_value(current_prompt)
            accepted = False
            final_prompt = current_prompt
        
        # Evaluate on full test set
        print("\nEvaluating on test set...")
        test_acc = eval_dataset(test_set, eval_fn, model)
        current_acc = np.mean(test_acc)
        
        results["test_acc"].append([int(x) for x in test_acc])  # Convert numpy types to int
        results["prompt"].append(system_prompt.get_value())
        results["rank"].append({
            "step": int(steps + 1),
            "mean_accuracy": float(current_acc),
            "n_preserve_cases": len(train_set_preserve),
            "n_hard_cases": len(train_set_hard_pool),
            "train_accuracy": float(len(train_set_preserve) / len(train_set_list)) if len(train_set_list) > 0 else 0.0,
            "n_candidates": len(candidates),
            "best_candidate_id": int(best_metrics['candidate_id']) if best_candidate else -1,
            "best_selection_score": float(best_score),
            "best_improvement": float(best_metrics.get('improvement', 0.0)),
            "best_regression": float(best_metrics.get('regression', 0.0)),
            "accepted": bool(accepted),
            "old_prompt": str(current_prompt[:500]),
            "final_prompt": str(final_prompt[:500]),
            "candidate_results": [{
                "id": int(cr['candidate_id']),
                "score": float(cr['selection_score']),
                "improvement": float(cr['improvement']),
                "regression": float(cr['regression']),
                "source": str(cr['source'])
            } for cr in candidate_results]
        })
        
        print(f"Test accuracy: {current_acc:.2%}")
        
        # if steps >= 10:  # Limit steps
        #     break

# Also dump the final results

with open(f"./figures/results_{args.task}_{args.test_engine}.json", "w") as f:
    json.dump(results, f)