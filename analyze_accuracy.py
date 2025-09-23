import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def load_data(json_file_path):
    """Load data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_accuracy_by_correctness(data):
    """Analyze accuracy data based on correctness of examples"""
    rank_data = data.get('rank', [])

    correct_accuracies = []
    incorrect_accuracies = []

    for item in rank_data:
        if len(item) >= 2:
            accuracy = item[0]  # First element is accuracy
            correctness = item[1]  # Second element is correctness ("1" or "0")

            if correctness == "1":
                correct_accuracies.append(accuracy)
            elif correctness == "0":
                incorrect_accuracies.append(accuracy)

    return correct_accuracies, incorrect_accuracies

def create_single_subplot(ax, correct_accuracies, incorrect_accuracies, title, subplot_idx, y_min=0, y_max=1.0):
    """Create a single subplot for accuracy comparison"""

    # Calculate statistics
    correct_mean = np.mean(correct_accuracies) if correct_accuracies else 0
    incorrect_mean = np.mean(incorrect_accuracies) if incorrect_accuracies else 0
    correct_std = np.std(correct_accuracies) if correct_accuracies else 0
    incorrect_std = np.std(incorrect_accuracies) if incorrect_accuracies else 0

    # Data for plotting
    categories = ['Correct', 'Incorrect']
    means = [correct_mean, incorrect_mean]
    stds = [correct_std, incorrect_std]

    # Color scheme (ICLR-appropriate colors)
    colors = ['#2E8B57', '#DC143C']  # Sea green for correct, crimson for incorrect

    # Create bar plot
    bars = ax.bar(categories, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Customize plot
    ax.set_ylabel('Average Accuracy' if subplot_idx == 0 else '', fontsize=12, fontweight='bold')
    ax.set_xlabel('Example Type', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Set y-axis limits with controllable zero point
    ax.set_ylim(y_min, y_max)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        # Adjust label position based on y-axis range
        label_offset = (y_max - y_min) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + std + label_offset,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add sample size information
    ax.text(0.02, 0.98, f'n_correct = {len(correct_accuracies)}\nn_incorrect = {len(incorrect_accuracies)}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return ax

def create_iclr_style_triple_plot(data_list, titles, y_limits=None, save_path=None):
    """Create ICLR-style triple bar chart comparing accuracies

    Args:
        data_list: List of (correct_accuracies, incorrect_accuracies) tuples
        titles: List of titles for each subplot
        y_limits: List of (y_min, y_max) tuples for each subplot, or single tuple for all
        save_path: Path to save the plot
    """

    # Set ICLR-style parameters
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'figure.dpi': 150
    })

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Handle y_limits parameter
    if y_limits is None:
        y_limits = [(0, 1.0)] * 3  # Default for all subplots
    elif isinstance(y_limits[0], (int, float)):
        # Single tuple provided, use for all subplots
        y_limits = [y_limits] * 3
    elif len(y_limits) == 1:
        # Single tuple in list, use for all subplots
        y_limits = y_limits * 3

    # Create each subplot
    for i, (data, title) in enumerate(zip(data_list, titles)):
        correct_accuracies, incorrect_accuracies = data
        y_min, y_max = y_limits[i] if i < len(y_limits) else (0, 1.0)
        create_single_subplot(axes[i], correct_accuracies, incorrect_accuracies, title, i, y_min, y_max)

    # Add overall title
    fig.suptitle('Accuracy Comparison Across Different Conditions',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, axes

def print_statistics(correct_accuracies, incorrect_accuracies):
    """Print detailed statistics"""
    print("=== ACCURACY ANALYSIS RESULTS ===")
    print(f"Correct Examples:")
    print(f"  Count: {len(correct_accuracies)}")
    print(f"  Mean: {np.mean(correct_accuracies):.4f}")
    print(f"  Std: {np.std(correct_accuracies):.4f}")
    print(f"  Min: {np.min(correct_accuracies):.4f}")
    print(f"  Max: {np.max(correct_accuracies):.4f}")

    print(f"\nIncorrect Examples:")
    print(f"  Count: {len(incorrect_accuracies)}")
    if incorrect_accuracies:
        print(f"  Mean: {np.mean(incorrect_accuracies):.4f}")
        print(f"  Std: {np.std(incorrect_accuracies):.4f}")
        print(f"  Min: {np.min(incorrect_accuracies):.4f}")
        print(f"  Max: {np.max(incorrect_accuracies):.4f}")
    else:
        print("  No incorrect examples found")

    print("=" * 35)

def create_synthetic_data_variations(correct_accuracies, incorrect_accuracies, baseline=0.7):
    """Create variations of the data for demonstration purposes

    Args:
        correct_accuracies: List of correct example accuracies
        incorrect_accuracies: List of incorrect example accuracies
        baseline: Baseline accuracy value for computing changes (default: 0.7)
    """

    # Original data (absolute accuracies)
    data1 = (correct_accuracies, incorrect_accuracies)

    # Variation 2: Changes relative to baseline (can be positive or negative)
    correct_var2 = [acc - baseline for acc in correct_accuracies]
    incorrect_var2 = [acc - baseline for acc in incorrect_accuracies]
    data2 = (correct_var2, incorrect_var2)

    # Variation 3: Percentage change relative to baseline
    correct_var3 = [(acc - baseline) / baseline * 100 if baseline != 0 else 0 for acc in correct_accuracies]
    incorrect_var3 = [(acc - baseline) / baseline * 100 if baseline != 0 else 0 for acc in incorrect_accuracies]
    data3 = (correct_var3, incorrect_var3)

    return [data1, data2, data3], baseline

def main():
    # File paths
    json_file = Path("figures/10-instance_analysis/instance_analysis_4.json")
    output_plot_single = Path("figures/10-instance_analysis/accuracy_comparison_single.png")
    output_plot_triple = Path("figures/10-instance_analysis/accuracy_comparison_triple.png")

    # Load and analyze data
    print("Loading data...")
    data = load_data(json_file)

    print("Analyzing accuracy by correctness...")
    correct_accuracies, incorrect_accuracies = analyze_accuracy_by_correctness(data)

    # Print statistics
    print_statistics(correct_accuracies, incorrect_accuracies)

    # Create synthetic variations for triple plot demonstration
    print("Creating data variations for triple plot...")
    np.random.seed(42)  # For reproducible results

    # Set baseline value - you can adjust this
    baseline_accuracy = 0.7
    data_variations, baseline = create_synthetic_data_variations(correct_accuracies, incorrect_accuracies, baseline_accuracy)

    # Titles for the three subplots
    titles = [
        'Absolute Accuracy\n(Original)',
        f'Change from Baseline\n(Baseline = {baseline:.1f})',
        f'Percentage Change\n(Baseline = {baseline:.1f})'
    ]

    # Define different y-axis limits for demonstration
    # Option 1: Same limits for all subplots
    # y_limits = (0, 1.0)

    # Option 2: Different limits for each subplot to accommodate positive/negative values
    y_limits = [
        (0, 1.0),      # First subplot: absolute accuracy (0-1 range)
        (-0.3, 0.3),   # Second subplot: change from baseline (can be negative)
        (-50, 50)      # Third subplot: percentage change (can be negative)
    ]

    # Create and save triple plot with custom y-axis limits
    print("Creating ICLR-style triple plot with custom y-axis limits...")
    fig, axes = create_iclr_style_triple_plot(data_variations, titles, y_limits, output_plot_triple)

    # Also create a version with uniform y-axis for comparison
    output_plot_uniform = Path("figures/10-instance_analysis/accuracy_comparison_uniform.png")
    print("Creating version with uniform y-axis...")
    fig_uniform, axes_uniform = create_iclr_style_triple_plot(data_variations, titles, (0, 1.0), output_plot_uniform)

    # Show plot
    plt.show()

    print("Analysis complete!")
    print(f"Custom y-axis plot saved to: {output_plot_triple}")
    print(f"Uniform y-axis plot saved to: {output_plot_uniform}")
    print(f"\nBaseline accuracy used: {baseline:.3f}")
    print("\nY-axis configurations used:")
    plot_types = ["Absolute accuracy", "Change from baseline", "Percentage change"]
    for i, (y_min, y_max) in enumerate(y_limits):
        print(f"  Subplot {i+1} ({plot_types[i]}): y-axis range [{y_min}, {y_max}]")

    # Print some baseline-relative statistics
    print(f"\nBaseline-relative statistics:")
    correct_mean = np.mean(correct_accuracies)
    incorrect_mean = np.mean(incorrect_accuracies)
    print(f"  Correct examples vs baseline: {correct_mean - baseline:+.3f} ({(correct_mean - baseline)/baseline*100:+.1f}%)")
    print(f"  Incorrect examples vs baseline: {incorrect_mean - baseline:+.3f} ({(incorrect_mean - baseline)/baseline*100:+.1f}%)")

if __name__ == "__main__":
    main()