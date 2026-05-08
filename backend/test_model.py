"""
Comprehensive Model Testing & Evaluation Script
================================================================================
Tests the trained cattle breed classification model and generates detailed metrics.

Metrics Include:
  - Accuracy
  - Precision (macro, weighted, per-class)
  - Recall (macro, weighted, per-class)
  - F1 Score (macro, weighted, per-class)
  - AUC-ROC (One-vs-Rest macro)
  - G-Mean Score
  - Matthews Correlation Coefficient
  - Confusion Matrix
  - Classification Report

Author: Script Generator
Date: 2024
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import timm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'model_path': './best_model.pth',
    'metadata_path': './metadata.json',
    'test_data_dir': './test_data',  # Adjust based on your test data location
    'img_size': 384,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(metadata_path):
    """Load model metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_model(model_path, metadata, device):
    """Load trained model."""
    model = timm.create_model(metadata['model_name'], num_classes=metadata['num_classes'])
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Try different possible keys
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's the state_dict directly
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Handle wrapper architecture (remove backbone/head prefix)
    # Check if keys have "backbone." or "head." prefix
    if any(k.startswith('backbone.') or k.startswith('head.') for k in state_dict.keys()):
        print("  ⓘ Detected wrapped architecture, restructuring weights...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                # Remove 'backbone.' prefix
                new_key = key[9:]  # len('backbone.') = 9
                new_state_dict[new_key] = value
            elif key.startswith('head.'):
                # Skip head keys (we have our own classifier)
                continue
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class CattleDataset(torch.utils.data.Dataset):
    """Custom dataset for cattle images."""
    
    def __init__(self, image_paths, labels, img_size=384):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            image = np.array(image) / 255.0
            image = torch.FloatTensor(image).permute(2, 0, 1)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, self.img_size, self.img_size))
        
        return image, label


def prepare_test_data(test_data_dir, class_names, img_size=384):
    """
    Prepare test data from directory structure: test_data_dir/class_name/image.jpg
    If test_data_dir is a single image directory, treat all images as one class.
    """
    image_paths = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    test_path = Path(test_data_dir)
    
    if not test_path.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return None, None, None
    
    # Check if we have subdirectories (one per class)
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    
    if subdirs and all(d.name in class_to_idx for d in subdirs):
        # Organized by class
        print(f"📂 Found {len(subdirs)} class directories")
        for class_dir in subdirs:
            class_idx = class_to_idx[class_dir.name]
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    image_paths.append(str(img_file))
                    labels.append(class_idx)
    else:
        # Treat as unorganized images (assign to first class for testing)
        print(f"📂 Found image files (treating as unlabeled data)")
        for img_file in test_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                image_paths.append(str(img_file))
                labels.append(0)  # Default to first class
    
    if not image_paths:
        print(f"❌ No images found in {test_data_dir}")
        return None, None, None
    
    print(f"✅ Loaded {len(image_paths)} test images")
    return image_paths, labels, class_to_idx


def compute_gmean(recall_scores):
    """Compute G-Mean (Geometric Mean) from per-class recalls."""
    gmean = np.power(np.prod(recall_scores), 1.0 / len(recall_scores))
    return gmean


def evaluate_model(model, dataloader, device, num_classes, class_names):
    """
    Evaluate model on test data and compute comprehensive metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔍 Running inference on test data...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CALCULATE METRICS
    # ─────────────────────────────────────────────────────────────────────────
    
    metrics = {}
    
    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    metrics['accuracy'] = accuracy
    
    # Precision (macro, weighted, per-class)
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    metrics['precision_per_class'] = {class_names[i]: float(score) for i, score in enumerate(precision_per_class)}
    
    # Recall (macro, weighted, per-class)
    metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    metrics['recall_per_class'] = {class_names[i]: float(score) for i, score in enumerate(recall_per_class)}
    
    # F1 Score (macro, weighted, per-class)
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    metrics['f1_per_class'] = {class_names[i]: float(score) for i, score in enumerate(f1_per_class)}
    
    # G-Mean Score
    gmean = compute_gmean(recall_per_class)
    metrics['gmean'] = gmean
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(all_labels, all_preds)
    metrics['mcc'] = mcc
    
    # AUC-ROC (One-vs-Rest macro)
    try:
        if num_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
            auc_score = roc_auc_score(all_labels_bin, all_probs, average='macro', multi_class='ovr')
        metrics['auc_roc_macro'] = auc_score
    except Exception as e:
        print(f"⚠️  Could not compute AUC-ROC: {e}")
        metrics['auc_roc_macro'] = None
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification Report (text format)
    cls_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    metrics['classification_report'] = cls_report
    
    # Get per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    metrics['accuracy_per_class'] = {class_names[i]: float(acc) for i, acc in enumerate(per_class_accuracy)}
    
    return metrics, all_preds, all_labels, all_probs


def print_metrics(metrics, class_names):
    """Print metrics in a formatted way."""
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"  • Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  • Precision (Macro):     {metrics['precision_macro']:.4f}")
    print(f"  • Precision (Weighted):  {metrics['precision_weighted']:.4f}")
    print(f"  • Recall (Macro):        {metrics['recall_macro']:.4f}")
    print(f"  • Recall (Weighted):     {metrics['recall_weighted']:.4f}")
    print(f"  • F1 Score (Macro):      {metrics['f1_macro']:.4f}")
    print(f"  • F1 Score (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  • G-Mean Score:          {metrics['gmean']:.4f}")
    print(f"  • MCC (Matthews Corr):   {metrics['mcc']:.4f}")
    if metrics['auc_roc_macro'] is not None:
        print(f"  • AUC-ROC (Macro):       {metrics['auc_roc_macro']:.4f}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(metrics['classification_report'])
    
    print(f"\n🎯 PER-CLASS METRICS:")
    print(f"\n{'Class Name':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 68)
    for class_name in class_names:
        acc = metrics['accuracy_per_class'].get(class_name, 0)
        prec = metrics['precision_per_class'].get(class_name, 0)
        rec = metrics['recall_per_class'].get(class_name, 0)
        f1 = metrics['f1_per_class'].get(class_name, 0)
        print(f"{class_name:<20} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    print("\n" + "="*80)


def save_metrics(metrics, output_path='evaluation_results.json'):
    """Save metrics to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        else:
            metrics_json[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"✅ Metrics saved to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(CONFIG['seed'])
    
    print("🚀 Starting Model Evaluation Pipeline")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Model: {CONFIG['model_path']}")
    print(f"   Metadata: {CONFIG['metadata_path']}")
    
    # Load metadata
    print("\n📂 Loading metadata...")
    metadata = load_metadata(CONFIG['metadata_path'])
    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    model_name = metadata['model_name']
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    
    # Load model
    print("\n🧠 Loading model...")
    model = load_model(CONFIG['model_path'], metadata, CONFIG['device'])
    print("   ✅ Model loaded successfully")
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    image_paths, labels, class_to_idx = prepare_test_data(
        CONFIG['test_data_dir'], 
        class_names, 
        CONFIG['img_size']
    )
    
    if image_paths is None:
        print("\n⚠️  Could not find test data at:", CONFIG['test_data_dir'])
        print("   Please ensure test data exists in one of these formats:")
        print("   1. Organized by class: test_dir/class_name/image.jpg")
        print("   2. All images in one folder: test_dir/image.jpg")
        return
    
    # Create dataset and dataloader
    test_dataset = CattleDataset(image_paths, labels, CONFIG['img_size'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    
    # Evaluate model
    print("\n🔬 Evaluating model...")
    metrics, preds, labels_true, probs = evaluate_model(
        model, test_loader, CONFIG['device'], num_classes, class_names
    )
    
    # Print results
    print_metrics(metrics, class_names)
    
    # Save results
    print("\n💾 Saving results...")
    save_metrics(metrics, 'evaluation_results.json')
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, 'confusion_matrix.png')
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
