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
    parser = argparse.ArgumentParser(description="Optimize a prompt with hard-pool + preserve gating and a static gate-set acceptance check.")
    parser.add_argument("--task", type=str, default="MMLU_college_physics", help="The task to evaluate the model on.")
    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="The API to use for evaluation (backward engine).")
    parser.add_argument("--test_engine", type=str, default="gpt-3.5-turbo-0125", help="The API to use for test-time forward.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of hard cases to accumulate gradients per candidate.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs to train.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--do_not_run_larger_model", action="store_true", help="Whether to run the larger model or not.")
    parser.add_argument("--num_threads", type=int, default=32, help="Number of threads for evaluation.")
    parser.add_argument("--preserve_sample_size", type=int, default=20, help="Size of preserve sample for candidate gating.")
    parser.add_argument("--lambda_gating", type=float, default=1.5, help="Lambda parameter for gating rule.")
    parser.add_argument("--n_hard_examples", type=int, default=3, help="Number of hard-example batches (candidates) per round.")
    # New: static gate-set acceptance
    parser.add_argument("--gate_sample_size", type=int, default=50, help="Size of static gate sample for acceptance.")
    parser.add_argument("--accept_epsilon", type=float, default=-0.10, help="Minimum gate-set improvement required to accept a candidate.")
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
    except Exception:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int = None):
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
    except Exception:
        eval_output_variable = eval_fn([x_var, y_var, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


set_seed(args.seed)
llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
llm_api_test = tg.get_engine(engine_name=args.test_engine)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

# Initialize system prompt and model
system_prompt = tg.Variable(
    STARTING_SYSTEM_PROMPT,
    requires_grad=True,
    role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task",
)
model = tg.BlackboxLLM(llm_api_test, system_prompt)

# Convert train_set to list for easier manipulation
train_set_list = []
for sample in train_set:
    train_set_list.append(sample)
print(f"Total training examples: {len(train_set_list)}")

# Static gate set for acceptance testing (fixed across steps to avoid drift)
gate_sample_size = min(args.gate_sample_size, len(train_set_list))
gate_set = random.sample(train_set_list, gate_sample_size) if gate_sample_size > 0 else train_set_list
print(f"Gate set size: {len(gate_set)}")


def update_train_pools(current_model, eval_fn):
    """Dynamically update train_set_preserve and train_set_hard_pool based on current model performance using multithreading"""
    preserve_cases = []  # Cases where eval output = 1
    hard_cases = []  # Cases where eval output = 0

    print(f"\nRe-evaluating {len(train_set_list)} training cases with current prompt...")

    # Get current prompt
    current_prompt = current_model.system_prompt.get_value()

    # Use multithreaded evaluation
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

print(STARTING_SYSTEM_PROMPT)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": [], "validation_acc": [], "rank": []}
# Get initial test accuracy
initial_test_acc = eval_dataset(test_set, eval_fn, model)
results["test_acc"].append([int(x) for x in initial_test_acc])  # Convert numpy types to int
results["prompt"].append(system_prompt.get_value())


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

        # Step 1: Generate multiple candidate prompts using batches of hard examples
        current_prompt = system_prompt.get_value()
        candidates = []

        n_candidates_to_generate = min(
            args.n_hard_examples, (len(train_set_hard_pool) + args.batch_size - 1) // args.batch_size
        )
        if n_candidates_to_generate == 0:
            print("No hard cases available, skipping this step.")
            continue

        print(
            f"Generating {n_candidates_to_generate} candidates, each from a batch of size up to {args.batch_size}..."
        )

        for i in range(n_candidates_to_generate):
            print(f"\n--- Generating Candidate {i+1}/{n_candidates_to_generate} from a random batch ---")

            system_prompt.set_value(current_prompt)

            # Sample a batch from hard pool
            batch_size = min(args.batch_size, len(train_set_hard_pool))
            batch = random.sample(train_set_hard_pool, batch_size)

            # Forward pass for each sample in batch, collect eval outputs
            eval_outputs = []
            correct_flags = []
            for bx, by in batch:
                x_var = tg.Variable(bx, requires_grad=False, role_description="query to the language model")
                y_var = tg.Variable(by, requires_grad=False, role_description="correct answer for the query")

                response = model(x_var)
                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
                    try:
                        ok = int(eval_output_variable.value)
                    except Exception:
                        ok = 1 if str(eval_output_variable.value).strip() in ["1", "<ACCURACY> 1 </ACCURACY>"] else 0
                except Exception:
                    eval_output_variable = eval_fn([x_var, y_var, response])
                    try:
                        ok = int(eval_fn.parse_output(eval_output_variable))
                    except Exception:
                        ok = 0

                eval_outputs.append(eval_output_variable)
                correct_flags.append(ok == 1)

            n_correct = sum(correct_flags)
            print(f"Batch correctness: {n_correct}/{len(batch)}")

            # If the whole batch is already solved, skip optimization for this candidate
            if all(correct_flags):
                print("This batch is already solved. Using current prompt as a candidate.")
                candidates.append({"prompt": current_prompt, "source": "current_prompt_batch_solved"})
                continue

            # Accumulate gradients from incorrect items in the batch and step once
            optimizer.zero_grad()
            for ev, ok in zip(eval_outputs, correct_flags):
                if not ok:
                    ev.backward()
            optimizer.step()

            candidate_prompt = system_prompt.get_value()
            candidates.append({"prompt": candidate_prompt, "source": f"optimized_from_batch_of_{len(batch)}"})

            print(f"Generated candidate (first 150 chars): {candidate_prompt[:150]}...")

        # Step 2: Evaluate all candidates and select the best one via preserve/hard gating
        print(f"\n--- Evaluating {len(candidates)} candidates ---")

        # Sample from D_preserve for testing
        if len(train_set_preserve) >= args.preserve_sample_size:
            preserve_sample = random.sample(train_set_preserve, args.preserve_sample_size)
        else:
            preserve_sample = train_set_preserve

        print(f"Testing on {len(preserve_sample)} preserve cases...")

        best_candidate = None
        best_score = float("-inf")
        best_metrics = {}
        candidate_results = []

        # Evaluate current prompt as baseline
        score_old = evaluate_on_sample_set(preserve_sample, current_prompt, llm_api_test, eval_fn)
        improvement_old = evaluate_on_sample_set(train_set_hard_pool, current_prompt, llm_api_test, eval_fn)

        for i, candidate in enumerate(candidates):
            candidate_prompt = candidate["prompt"]

            # Evaluate on preserve set
            score_candidate = evaluate_on_sample_set(preserve_sample, candidate_prompt, llm_api_test, eval_fn)
            regression = score_old - score_candidate

            # Evaluate improvement on all hard cases
            improvement_candidate = evaluate_on_sample_set(
                train_set_hard_pool, candidate_prompt, llm_api_test, eval_fn
            )
            improvement = improvement_candidate - improvement_old

            # Calculate selection score: improvement - lambda * regression
            selection_score = improvement - args.lambda_gating * regression

            candidate_metrics = {
                "candidate_id": i + 1,
                "regression": regression,
                "improvement": improvement,
                "selection_score": selection_score,
                "preserve_score": score_candidate,
                "hard_score": improvement_candidate,
                "source": candidate["source"],
            }
            candidate_results.append(candidate_metrics)

            print(
                f"Candidate {i+1}: Score={selection_score:.4f} (Imp={improvement:.2%}, Reg={regression:.2%})"
            )

            if selection_score > best_score:
                best_score = selection_score
                best_candidate = candidate
                best_metrics = candidate_metrics

        # Step 3: Apply the best candidate with static gate-set acceptance
        print(f"\n--- Best candidate: #{best_metrics['candidate_id']} with score {best_score:.4f} ---")

        # Evaluate on gate set to ensure global improvement before accepting
        gate_acc_old = evaluate_on_sample_set(gate_set, current_prompt, llm_api_test, eval_fn)
        gate_acc_new = (
            evaluate_on_sample_set(gate_set, best_candidate["prompt"], llm_api_test, eval_fn)
            if best_candidate
            else gate_acc_old
        )
        gate_improvement = gate_acc_new - gate_acc_old
        print(f"Gate acc old={gate_acc_old:.2%}, new={gate_acc_new:.2%}, Δ={gate_improvement:.2%}")

        if best_candidate and best_score > 0 and (gate_improvement >= args.accept_epsilon):
            print("✓ Accepting best candidate (passes gate-set check)")
            system_prompt.set_value(best_candidate["prompt"])
            accepted = True
            final_prompt = best_candidate["prompt"]
        else:
            print("✗ Rejecting candidate (fails gate-set check); keep current prompt")
            system_prompt.set_value(current_prompt)
            accepted = False
            final_prompt = current_prompt

        # Evaluate on full test set
        print("\nEvaluating on test set...")
        test_acc = eval_dataset(test_set, eval_fn, model)
        current_acc = np.mean(test_acc)

        results["test_acc"].append([int(x) for x in test_acc])  # Convert numpy types to int
        results["prompt"].append(system_prompt.get_value())
        results["rank"].append(
            {
                "step": int(steps + 1),
                "mean_accuracy": float(current_acc),
                "n_preserve_cases": len(train_set_preserve),
                "n_hard_cases": len(train_set_hard_pool),
                "train_accuracy": float(len(train_set_preserve) / len(train_set_list)) if len(train_set_list) > 0 else 0.0,
                "n_candidates": len(candidates),
                "best_candidate_id": int(best_metrics["candidate_id"]) if best_candidate else -1,
                "best_selection_score": float(best_score),
                "best_improvement": float(best_metrics.get("improvement", 0.0)),
                "best_regression": float(best_metrics.get("regression", 0.0)),
                "accepted": bool(accepted),
                "old_prompt": str(current_prompt[:500]),
                "final_prompt": str(final_prompt[:500]),
                "candidate_results": [
                    {
                        "id": int(cr["candidate_id"]),
                        "score": float(cr["selection_score"]),
                        "improvement": float(cr["improvement"]),
                        "regression": float(cr["regression"]),
                        "source": str(cr["source"]),
                    }
                    for cr in candidate_results
                ],
                "gate_acc_old": float(gate_acc_old),
                "gate_acc_new": float(gate_acc_new),
                "gate_improvement": float(gate_improvement),
            }
        )

        print(f"Test accuracy: {current_acc:.2%}")

        if steps >= 10:  # Limit steps
            break

# Also dump the final results

with open(f"./figures/results_gated_{args.task}_{args.test_engine}.json", "w") as f:
    json.dump(results, f)

