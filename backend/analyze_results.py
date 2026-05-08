"""
Results Analysis & Visualization
================================================================================
Post-evaluation analysis script to explore and visualize model performance.

This script loads the evaluation_results.json and provides various analyses
and visualizations to understand model performance better.

Usage: python analyze_results.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_file='evaluation_results.json'):
    """Load evaluation results."""
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run: python test_model.py")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded results from {results_file}")
    return results


def print_summary(results):
    """Print summary metrics."""
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Accuracy:      {results['accuracy']:.4f}")
    print(f"   Precision:     {results['precision_macro']:.4f}")
    print(f"   Recall:        {results['recall_macro']:.4f}")
    print(f"   F1-Score:      {results['f1_macro']:.4f}")
    print(f"   G-Mean:        {results['gmean']:.4f}")
    print(f"   AUC-ROC:       {results.get('auc_roc_macro', 'N/A')}")


def find_best_worst_classes(results, metric='f1_per_class'):
    """Find best and worst performing classes."""
    print("\n" + "="*70)
    print("🏆 CLASS PERFORMANCE")
    print("="*70)
    
    scores = results[metric]
    
    # Best classes
    best_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n🥇 Top 5 Classes (by {metric.replace('_per_class', '')}):")
    for i, (class_name, score) in enumerate(best_classes, 1):
        print(f"   {i}. {class_name:<20} {score:.4f}")
    
    # Worst classes
    worst_classes = sorted(scores.items(), key=lambda x: x[1])[:5]
    print(f"\n🚨 Bottom 5 Classes (by {metric.replace('_per_class', '')}):")
    for i, (class_name, score) in enumerate(worst_classes, 1):
        print(f"   {i}. {class_name:<20} {score:.4f}")


def analyze_per_class_metrics(results):
    """Analyze per-class metrics."""
    print("\n" + "="*70)
    print("📈 PER-CLASS DETAILED ANALYSIS")
    print("="*70)
    
    # Create DataFrame
    classes = list(results['accuracy_per_class'].keys())
    data = {
        'Class': classes,
        'Accuracy': [results['accuracy_per_class'][c] for c in classes],
        'Precision': [results['precision_per_class'][c] for c in classes],
        'Recall': [results['recall_per_class'][c] for c in classes],
        'F1-Score': [results['f1_per_class'][c] for c in classes],
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('F1-Score', ascending=False)
    
    # Statistics
    print(f"\n📊 Metric Statistics:")
    print(f"\nAccuracy:")
    print(f"   Mean:   {df['Accuracy'].mean():.4f}")
    print(f"   Min:    {df['Accuracy'].min():.4f}")
    print(f"   Max:    {df['Accuracy'].max():.4f}")
    print(f"   Std:    {df['Accuracy'].std():.4f}")
    
    print(f"\nF1-Score:")
    print(f"   Mean:   {df['F1-Score'].mean():.4f}")
    print(f"   Min:    {df['F1-Score'].min():.4f}")
    print(f"   Max:    {df['F1-Score'].max():.4f}")
    print(f"   Std:    {df['F1-Score'].std():.4f}")
    
    # Classes with low performance
    low_accuracy = df[df['Accuracy'] < 0.70]
    if len(low_accuracy) > 0:
        print(f"\n⚠️  Classes with <70% Accuracy:")
        for _, row in low_accuracy.iterrows():
            print(f"   • {row['Class']}: {row['Accuracy']:.2%}")
    else:
        print(f"\n✅ All classes above 70% accuracy!")
    
    return df


def analyze_precision_recall_tradeoff(results):
    """Analyze precision vs recall tradeoff."""
    print("\n" + "="*70)
    print("🔀 PRECISION vs RECALL ANALYSIS")
    print("="*70)
    
    classes = list(results['precision_per_class'].keys())
    
    large_gap = []
    for class_name in classes:
        precision = results['precision_per_class'][class_name]
        recall = results['recall_per_class'][class_name]
        gap = abs(precision - recall)
        if gap > 0.15:  # Gap > 15%
            large_gap.append((class_name, precision, recall, gap))
    
    if large_gap:
        print(f"\n⚠️  Classes with large Precision-Recall gap (>15%):")
        for class_name, prec, rec, gap in sorted(large_gap, key=lambda x: x[3], reverse=True):
            print(f"   • {class_name:<20} Precision: {prec:.4f}, Recall: {rec:.4f}, Gap: {gap:.4f}")
            if prec > rec:
                print(f"     → Model is conservative (high precision, missing some)")
            else:
                print(f"     → Model is aggressive (high recall, false positives)")
    else:
        print(f"\n✅ Balanced Precision-Recall for all classes")


def analyze_confusion_patterns(results):
    """Analyze confusion matrix patterns."""
    print("\n" + "="*70)
    print("🔀 CONFUSION ANALYSIS")
    print("="*70)
    
    cm = np.array(results['confusion_matrix'])
    classes = list(results['accuracy_per_class'].keys())
    
    # Find highest off-diagonal values
    print(f"\n🤔 Top Confusion Patterns:")
    
    off_diag_values = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                off_diag_values.append((cm[i][j], classes[i], classes[j]))
    
    off_diag_values.sort(reverse=True)
    
    for count, true_class, pred_class in off_diag_values[:10]:
        percentage = (count / cm[classes.index(true_class)].sum()) * 100
        print(f"   • {true_class:<20} misclassified as {pred_class:<20} ({count}x, {percentage:.1f}%)")


def create_performance_heatmap(results):
    """Create F1-score heatmap by class."""
    classes = list(results['accuracy_per_class'].keys())
    f1_scores = [results['f1_per_class'][c] for c in classes]
    
    # Sort by F1 score
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1])
    sorted_classes = [x[0] for x in sorted_pairs]
    sorted_f1 = [x[1] for x in sorted_pairs]
    
    # Create color map (red for low, green for high)
    colors = plt.cm.RdYlGn(np.array(sorted_f1))
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(sorted_classes, sorted_f1, color=colors)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
        plt.text(score - 0.05, i, f'{score:.3f}', va='center', ha='right', fontweight='bold', color='white')
    
    plt.xlabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Class (Sorted)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('per_class_f1_scores.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: per_class_f1_scores.png")
    plt.close()


def create_metrics_comparison(results):
    """Compare different metrics."""
    classes = list(results['accuracy_per_class'].keys())[:10]  # Top 10 for clarity
    
    accuracy = [results['accuracy_per_class'][c] for c in classes]
    precision = [results['precision_per_class'][c] for c in classes]
    recall = [results['recall_per_class'][c] for c in classes]
    f1_scores = [results['f1_per_class'][c] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.2
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    plt.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics Comparison (First 10 Classes)', fontsize=14, fontweight='bold')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: metrics_comparison.png")
    plt.close()


def create_metrics_scatter(results):
    """Create scatter plot of Precision vs Recall."""
    classes = list(results['precision_per_class'].keys())
    
    precision = [results['precision_per_class'][c] for c in classes]
    recall = [results['recall_per_class'][c] for c in classes]
    f1_scores = [results['f1_per_class'][c] for c in classes]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(precision, recall, s=200, c=f1_scores, cmap='RdYlGn', alpha=0.6, edgecolors='black')
    
    # Add class labels to some points
    for i, class_name in enumerate(classes[::3]):  # Every 3rd class to avoid clutter
        idx = classes.index(class_name)
        plt.annotate(class_name, (precision[idx], recall[idx]), fontsize=8, alpha=0.7)
    
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Precision vs Recall Analysis', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1 Score', fontsize=11)
    
    # Add diagonal line (ideal precision=recall)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('precision_recall_scatter.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: precision_recall_scatter.png")
    plt.close()


def main():
    print("🔍 MODEL EVALUATION RESULTS ANALYSIS\n")
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    # Print summary
    print_summary(results)
    
    # Analysis
    find_best_worst_classes(results)
    df = analyze_per_class_metrics(results)
    analyze_precision_recall_tradeoff(results)
    analyze_confusion_patterns(results)
    
    # Create visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    create_performance_heatmap(results)
    create_metrics_comparison(results)
    create_metrics_scatter(results)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print("\n📁 Generated Files:")
    print("   • per_class_f1_scores.png")
    print("   • metrics_comparison.png")
    print("   • precision_recall_scatter.png")
    
    print("\n💡 Next Steps:")
    print("   1. Review the visualizations above")
    print("   2. Focus on classes with low F1 scores")
    print("   3. Analyze confusion patterns (which classes get mixed up)")
    print("   4. Consider data augmentation or class rebalancing")


if __name__ == "__main__":
    main()
