#!/usr/bin/env python3
"""
Analyze prompt differences between two JSON result files from randomness experiments.
This script performs line-by-line comparison of the prompt evolution in two different runs.
"""

import json
import difflib
from pathlib import Path
import argparse


def load_json_results(file_path):
    """Load JSON results file and return the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_prompt_differences(file1_path, file2_path):
    """Analyze prompt differences between two result files."""
    print(f"Analyzing prompt differences between:")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print("=" * 80)
    
    # Load the data
    data1 = load_json_results(file1_path)
    data2 = load_json_results(file2_path)
    
    prompts1 = data1.get('prompt', [])
    prompts2 = data2.get('prompt', [])
    
    print(f"File 1 has {len(prompts1)} prompt iterations")
    print(f"File 2 has {len(prompts2)} prompt iterations")
    print()
    
    # Compare each iteration
    max_iterations = max(len(prompts1), len(prompts2))
    differences_found = 0
    
    for i in range(max_iterations):
        print(f"ITERATION {i}:")
        print("-" * 40)
        
        if i >= len(prompts1):
            print(f"❌ File 1 missing iteration {i}")
            print(f"File 2 prompt: {prompts2[i][:100]}...")
            differences_found += 1
            continue
            
        if i >= len(prompts2):
            print(f"❌ File 2 missing iteration {i}")
            print(f"File 1 prompt: {prompts1[i][:100]}...")
            differences_found += 1
            continue
            
        prompt1 = prompts1[i]
        prompt2 = prompts2[i]
        
        if prompt1 == prompt2:
            print(f"✅ Prompts identical")
            print(f"Content: {prompt1[:100]}...")
        else:
            print(f"❌ Prompts differ!")
            differences_found += 1
            
            # Show detailed diff
            print(f"\nFile 1 length: {len(prompt1)} chars")
            print(f"File 2 length: {len(prompt2)} chars")
            
            # Split into lines for better comparison
            lines1 = prompt1.split('\n')
            lines2 = prompt2.split('\n')
            
            print(f"\nFile 1 lines: {len(lines1)}")
            print(f"File 2 lines: {len(lines2)}")
            
            # Generate unified diff
            diff = list(difflib.unified_diff(
                lines1, lines2,
                fromfile=f'File1_iter{i}',
                tofile=f'File2_iter{i}',
                lineterm=''
            ))
            
            if diff:
                print(f"\nDetailed differences:")
                for line in diff[:50]:  # Limit output to first 50 lines
                    print(line)
                if len(diff) > 50:
                    print(f"... (truncated, {len(diff)} total diff lines)")
            
            # Find character-level differences for short prompts
            if len(prompt1) < 500 and len(prompt2) < 500:
                print(f"\nCharacter-level analysis:")
                seq_matcher = difflib.SequenceMatcher(None, prompt1, prompt2)
                for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
                    if tag != 'equal':
                        print(f"  {tag}: File1[{i1}:{i2}]='{prompt1[i1:i2]}' vs File2[{j1}:{j2}]='{prompt2[j1:j2]}'")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY:")
    print(f"Total iterations compared: {max_iterations}")
    print(f"Differences found: {differences_found}")
    print(f"Identical prompts: {max_iterations - differences_found}")
    
    if differences_found == 0:
        print("🎉 ALL PROMPTS ARE IDENTICAL - No randomness in prompt evolution!")
    else:
        print(f"⚠️  RANDOMNESS DETECTED - {differences_found}/{max_iterations} iterations differ")
        
        # Calculate similarity metrics
        if len(prompts1) == len(prompts2):
            total_similarity = 0
            for i in range(len(prompts1)):
                seq_matcher = difflib.SequenceMatcher(None, prompts1[i], prompts2[i])
                similarity = seq_matcher.ratio()
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(prompts1)
            print(f"Average prompt similarity: {avg_similarity:.3f}")
    
    return differences_found, max_iterations


def compare_test_accuracy(file1_path, file2_path):
    """Compare test accuracy patterns between two files."""
    print("\n" + "=" * 80)
    print("TEST ACCURACY COMPARISON:")
    print("=" * 80)
    
    data1 = load_json_results(file1_path)
    data2 = load_json_results(file2_path)
    
    acc1 = data1.get('test_acc', [])
    acc2 = data2.get('test_acc', [])
    
    if len(acc1) != len(acc2):
        print(f"Different number of test epochs: {len(acc1)} vs {len(acc2)}")
        return
    
    for epoch in range(len(acc1)):
        if len(acc1[epoch]) == len(acc2[epoch]):
            matches = sum(1 for a, b in zip(acc1[epoch], acc2[epoch]) if a == b)
            total = len(acc1[epoch])
            agreement = matches / total
            print(f"Epoch {epoch}: {matches}/{total} samples agree ({agreement:.3f})")
        else:
            print(f"Epoch {epoch}: Different sample counts {len(acc1[epoch])} vs {len(acc2[epoch])}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt differences between randomness experiments")
    parser.add_argument("--include-accuracy", action="store_true", 
                       help="Also compare test accuracy patterns")
    
    args = parser.parse_args()
    current_dir = Path(__file__).parent
    
    folderPath = "9-compare_randomness_seed"
    # 文件路径
    file1_path = current_dir / folderPath / "results_BBH_object_counting_gpt-3.5-turbo-0125_9_top_p.json"
    file2_path = current_dir / folderPath / "results_BBH_object_counting_gpt-3.5-turbo-0125_10_top_p.json"
    
    # file1_path = Path(args.file1)
    # file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"Error: File {file1_path} not found")
        return 1
        
    if not file2_path.exists():
        print(f"Error: File {file2_path} not found")
        return 1
    
    try:
        differences, total = analyze_prompt_differences(file1_path, file2_path)
        
        if args.include_accuracy:
            compare_test_accuracy(file1_path, file2_path)
        
        return 0 if differences == 0 else 1
        
    except Exception as e:
        print(f"Error analyzing files: {e}")
        return 1


if __name__ == "__main__":
    exit(main())