#!/usr/bin/env python3
"""Create a three-panel bar chart comparing accuracy on correct vs incorrect examples."""

import json
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JSON_FILES = [
    Path("figures/10-instance_analysis/OBJECT_COUNTING.json"),
    Path("figures/10-instance_analysis/NAVIGATE.json"),
    Path("figures/10-instance_analysis/GSM8K.json"),
]


def load_data(json_file_path: Path):
    with json_file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_accuracy_by_correctness(data):
    """Return lists of accuracies split by correctness flag."""
    rank_data = data.get("rank", [])

    correct_accuracies = []
    incorrect_accuracies = []

    for item in rank_data:
        accuracy = None
        correctness = None

        if isinstance(item, list) and len(item) >= 2:
            accuracy, correctness = item[0], item[1]
        elif isinstance(item, dict):
            # Try common field names if dict structure is used
            accuracy = item.get("accuracy") or item.get("mean_accuracy")
            correctness = item.get("correct") or item.get("label")

        if accuracy is None or correctness is None:
            continue

        try:
            acc_value = float(accuracy)
        except (TypeError, ValueError):
            continue

        correct_flag = str(correctness).strip().lower()
        # Extract the first binary digit if present (handles strings like "<ACCURACY> 0 </ACCURACY>")
        match = re.search(r"[01]", correct_flag)
        if match:
            correct_flag = match.group()

        if correct_flag in {"1", "true"}:
            correct_accuracies.append(acc_value)
        elif correct_flag in {"0", "false"}:
            incorrect_accuracies.append(acc_value)

    return correct_accuracies, incorrect_accuracies


def create_single_subplot(ax, correct_accuracies, incorrect_accuracies, label, show_ylabel):
    correct_mean = np.mean(correct_accuracies) if correct_accuracies else 0.0
    incorrect_mean = np.mean(incorrect_accuracies) if incorrect_accuracies else 0.0
    correct_std = np.std(correct_accuracies) if correct_accuracies else 0.0
    incorrect_std = np.std(incorrect_accuracies) if incorrect_accuracies else 0.0

    categories = ["Correct", "Incorrect"]
    means = [correct_mean, incorrect_mean]
    stds = [correct_std, incorrect_std]
    # colors = ["#2E8B57", "#DC143C"]  # green vs crimson
    colors = ["#99B0C0", "#FABD6C"]
    bars = ax.bar(
        categories,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1,
    )

    if show_ylabel:
        ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel(label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.02,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )



def create_triple_plot(datasets, titles, save_path=None):
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ((correct_acc, incorrect_acc), title) in enumerate(zip(datasets, titles)):
        create_single_subplot(axes[idx], correct_acc, incorrect_acc, title, show_ylabel=(idx == 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy comparison for three datasets")
    parser.add_argument("json_files", nargs="*", default=[str(p) for p in DEFAULT_JSON_FILES], help="Paths to three JSON files")
    parser.add_argument("--titles", nargs="*", default=None, help="Titles for each subplot")
    parser.add_argument("--output", default="figures/10-instance_analysis/accuracy_triple.pdf", help="Output image path")
    args = parser.parse_args()

    if len(args.json_files) != 3:
        raise ValueError("Please provide exactly three JSON files.")

    datasets = []
    for file_path in args.json_files:
        data = load_data(Path(file_path))
        correct, incorrect = analyze_accuracy_by_correctness(data)
        datasets.append((correct, incorrect))

    if args.titles and len(args.titles) == 3:
        titles = args.titles
    else:
        titles = [Path(path).stem for path in args.json_files]

    create_triple_plot(datasets, titles, save_path=args.output)

    plt.show()

    # Print basic statistics to console
    for title, (correct, incorrect) in zip(titles, datasets):
        print(f"=== {title} ===")
        print(f"Correct: n={len(correct)} mean={np.mean(correct) if correct else float('nan'):.4f}")
        print(f"Incorrect: n={len(incorrect)} mean={np.mean(incorrect) if incorrect else float('nan'):.4f}")


if __name__ == "__main__":
    main()
