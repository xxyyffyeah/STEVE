#!/usr/bin/env python3
"""
TextGrad 优化结果可视化脚本

分析并比较TextualAdam与标准TextualGradientDescent优化器的性能表现。
专注于test_acc的变化趋势，提供详细的性能对比分析。

运行方式:
python figures/visualize_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import platform

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_results():
    """加载两个优化器的结果数据"""
    current_dir = Path(__file__).parent
    
    # 文件路径
    adam_file = current_dir / "results_BBH_object_counting_gpt-3.5-turbo-0125_textual_adam.json"
    standard_file = current_dir / "results_BBH_object_counting_gpt-3.5-turbo.json"
    
    results = {}
    
    # 加载TextualAdam结果
    if adam_file.exists():
        with open(adam_file, 'r', encoding='utf-8') as f:
            results['adam'] = json.load(f)
            print(f"加载TextualAdam结果: {len(results['adam']['test_acc'])}个训练步骤")
    else:
        print(f"警告: 找不到Adam结果文件 {adam_file}")
        results['adam'] = None
    
    # 加载标准优化器结果
    if standard_file.exists():
        with open(standard_file, 'r', encoding='utf-8') as f:
            results['standard'] = json.load(f)
            print(f"加载标准优化器结果: {len(results['standard']['test_acc'])}个训练步骤")
    else:
        print(f"警告: 找不到标准优化器结果文件 {standard_file}")
        results['standard'] = None
    
    return results

def calculate_accuracy_stats(test_acc_data):
    """计算每个步骤的准确率"""
    step_accuracies = []
    
    for step_data in test_acc_data:
        if isinstance(step_data, list):
            acc = np.mean(step_data)  # 0/1序列的均值就是准确率
        else:
            acc = step_data
        step_accuracies.append(acc)
    
    return np.array(step_accuracies)

def plot_accuracy_trends(results):
    """绘制准确率变化趋势对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'adam': '#FF6B6B', 'standard': '#4ECDC4'}
    labels = {'adam': 'TextualAdam', 'standard': 'TextualGradientDescent'}
    
    # 子图1: 准确率趋势线
    for optimizer_name, data in results.items():
        if data is None:
            continue
            
        accuracies = calculate_accuracy_stats(data['test_acc'])
        steps = range(len(accuracies))
        
        # 绘制准确率趋势线
        ax1.plot(steps, accuracies, 
                label=labels[optimizer_name], 
                color=colors[optimizer_name], 
                linewidth=2.5, 
                marker='o', 
                markersize=6)
    
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Trends During Optimization', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 子图2: 性能改进对比
    improvements = {}
    final_accs = {}
    
    for optimizer_name, data in results.items():
        if data is None:
            continue
            
        accuracies = calculate_accuracy_stats(data['test_acc'])
        initial_acc = accuracies[0]
        final_acc = accuracies[-1]
        max_acc = np.max(accuracies)
        
        improvements[optimizer_name] = {
            'initial': initial_acc,
            'final': final_acc,
            'max': max_acc,
            'improvement': final_acc - initial_acc,
            'max_improvement': max_acc - initial_acc
        }
        final_accs[optimizer_name] = final_acc
    
    # 绘制性能对比柱状图
    optimizers = list(improvements.keys())
    if optimizers:
        x_pos = np.arange(len(optimizers))
        
        initial_vals = [improvements[opt]['initial'] for opt in optimizers]
        final_vals = [improvements[opt]['final'] for opt in optimizers]
        max_vals = [improvements[opt]['max'] for opt in optimizers]
        
        width = 0.25
        ax2.bar(x_pos - width, initial_vals, width, 
               label='Initial Accuracy', color='lightgray', alpha=0.7)
        ax2.bar(x_pos, final_vals, width, 
               label='Final Accuracy', color=[colors[opt] for opt in optimizers])
        ax2.bar(x_pos + width, max_vals, width, 
               label='Max Accuracy', color=[colors[opt] for opt in optimizers], alpha=0.6)
        ax2.set_xlabel('Optimizer Type', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Performance Comparison Overview', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([labels[opt] for opt in optimizers])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 添加数值标签
        for i, (initial_val, final_val, max_val) in enumerate(zip(initial_vals, final_vals, max_vals)):
            # 初始值标签
            ax2.text(i - width, initial_val + 0.02, f'{initial_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            # 最终值标签
            ax2.text(i, final_val + 0.02, f'{final_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            # 最高值标签
            ax2.text(i + width, max_val + 0.02, f'{max_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig, improvements

def print_detailed_analysis(results):
    """打印详细的性能分析报告"""
    print("\n" + "="*60)
    print("📊 TextGrad 优化器性能分析报告")
    print("="*60)
    
    for optimizer_name, data in results.items():
        if data is None:
            continue
            
        optimizer_label = "TextualAdam" if optimizer_name == 'adam' else "TextualGradientDescent"
        print(f"\n🔍 {optimizer_label} 分析:")
        print("-" * 40)
        
        accuracies = calculate_accuracy_stats(data['test_acc'])
        
        print(f"总训练步骤: {len(accuracies)}")
        print(f"初始准确率: {accuracies[0]:.4f}")
        print(f"最终准确率: {accuracies[-1]:.4f}")
        print(f"最高准确率: {np.max(accuracies):.4f}")
        print(f"最低准确率: {np.min(accuracies):.4f}")
        print(f"平均准确率: {np.mean(accuracies):.4f}")
        print(f"准确率标准差: {np.std(accuracies):.4f}")
        print(f"性能改进: {accuracies[-1] - accuracies[0]:+.4f}")
        
        # 分析优化器特定信息
        if optimizer_name == 'adam' and 'optimizer_config' in data:
            config = data['optimizer_config']
            print(f"\nAdam参数配置:")
            print(f"  Beta1: {config.get('beta1', 'N/A')}")
            print(f"  Beta2: {config.get('beta2', 'N/A')}")
            print(f"  Epsilon: {config.get('epsilon', 'N/A')}")
            print(f"  动量窗口: {config.get('momentum_window', 'N/A')}")
        
        if 'performance_summary' in data:
            summary = data['performance_summary']
            print(f"\n性能摘要:")
            print(f"  改进幅度: {summary.get('improvement', 0):+.4f}")
            print(f"  优化步数: {summary.get('total_steps', len(accuracies)-1)}")

def compare_optimizers(results):
    """比较两个优化器的性能"""
    if results['adam'] is None or results['standard'] is None:
        print("\n⚠️  无法进行比较: 缺少某个优化器的数据")
        return
    
    print("\n" + "="*60)
    print("⚔️  优化器性能对比")
    print("="*60)
    
    adam_acc = calculate_accuracy_stats(results['adam']['test_acc'])
    standard_acc = calculate_accuracy_stats(results['standard']['test_acc'])
    
    adam_final = adam_acc[-1]
    standard_final = standard_acc[-1]
    
    print(f"\n最终性能对比:")
    print(f"TextualAdam:           {adam_final:.4f}")
    print(f"TextualGradientDescent: {standard_final:.4f}")
    print(f"性能差异:              {adam_final - standard_final:+.4f}")
    
    # 判断哪个更好
    if adam_final > standard_final:
        winner = "TextualAdam"
        margin = adam_final - standard_final
    elif standard_final > adam_final:
        winner = "TextualGradientDescent"
        margin = standard_final - adam_final
    else:
        winner = "平局"
        margin = 0
    
    print(f"\n🏆 优胜者: {winner}")
    if margin > 0:
        print(f"领先优势: {margin:.4f} ({margin*100:.2f}%)")
    
    # 收敛速度分析
    adam_improvement = adam_acc[-1] - adam_acc[0]
    standard_improvement = standard_acc[-1] - standard_acc[0]
    
    print(f"\n📈 改进幅度对比:")
    print(f"TextualAdam:           {adam_improvement:+.4f}")
    print(f"TextualGradientDescent: {standard_improvement:+.4f}")

def main():
    """主函数"""
    print("🚀 开始分析TextGrad优化结果...")
    
    # 加载数据
    results = load_results()
    
    if not any(results.values()):
        print("❌ 错误: 没有找到任何结果文件")
        return
    
    # 打印详细分析
    print_detailed_analysis(results)
    
    # 比较优化器性能
    compare_optimizers(results)
    
    # 生成可视化图表
    print(f"\n📊 生成可视化图表...")
    fig, _ = plot_accuracy_trends(results)
    
    # 保存图表
    output_path = Path(__file__).parent / "accuracy_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ 图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()
    
    print(f"\n🎉 分析完成！")

if __name__ == "__main__":
    main()