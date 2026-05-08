# 🧪 Model Evaluation & Testing Guide

## Overview
This package contains comprehensive scripts for evaluating your cattle breed classification model. The main testing script computes all key performance metrics and generates detailed reports.

## Files Included

- **test_model.py** - Main evaluation script (comprehensive metrics computation)
- **prepare_test_data.py** - Helper script to organize test images by class
- **requirements_testing.txt** - Additional dependencies

## Metrics Computed

### 📊 Overall Performance
- **Accuracy** - Overall percentage of correct predictions
- **Precision (Macro/Weighted)** - How many predicted positives are actually positive
- **Recall (Macro/Weighted)** - How many actual positives are correctly identified
- **F1 Score (Macro/Weighted)** - Harmonic mean of Precision and Recall
- **G-Mean Score** - Geometric mean of per-class recalls (good for imbalanced data)
- **MCC** - Matthews Correlation Coefficient (correlation metric)
- **AUC-ROC** - Area Under the ROC curve (One-vs-Rest macro average)

### 📈 Per-Class Metrics
- Per-class Accuracy
- Per-class Precision
- Per-class Recall
- Per-class F1 Score

### 📋 Additional Outputs
- **Confusion Matrix** - Shows misclassification patterns
- **Classification Report** - Detailed per-class statistics
- **Visualizations** - Confusion matrix heatmap

## Setup

### 1. Install Additional Dependencies
```bash
cd backend
pip install -r requirements_testing.txt
```

### 2. Organize Test Data

Your test data should be organized in one of two ways:

**Option A: Organized by Class (Recommended)**
```
test_data_dir/
├── Gir/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Sahiwal/
│   ├── img1.jpg
│   └── ...
└── ...
```

**Option B: Single Folder (All Images)**
```
test_data_dir/
├── image1.jpg
├── image2.jpg
└── ...
```

### 3. Update Config (if needed)

Edit `test_model.py` and update the CONFIG dictionary:

```python
CONFIG = {
    'model_path': './best_model.pth',           # Path to your model
    'metadata_path': './metadata.json',         # Path to metadata
    'test_data_dir': '../../../results/outputs', # Path to test images
    'img_size': 384,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}
```

## Usage

### Basic Usage

```bash
cd backend
python test_model.py
```

### Expected Output

The script will:
1. Load the model and metadata ✅
2. Load test images 📂
3. Run inference on all images 🔍
4. Compute comprehensive metrics 📊
5. Generate confusion matrix visualization 📈
6. Save results to `evaluation_results.json` 💾

### Sample Output
```
================================================================================
📊 MODEL EVALUATION RESULTS
================================================================================

🎯 OVERALL METRICS:
  • Accuracy:              0.8750
  • Precision (Macro):     0.8621
  • Precision (Weighted):  0.8750
  • Recall (Macro):        0.8545
  • Recall (Weighted):     0.8750
  • F1 Score (Macro):      0.8579
  • F1 Score (Weighted):   0.8750
  • G-Mean Score:          0.8523
  • MCC (Matthews Corr):   0.8412
  • AUC-ROC (Macro):       0.9876

📋 CLASSIFICATION REPORT:
              precision    recall  f1-score   support
         Gir       0.90      0.85      0.87        20
    Sahiwal       0.87      0.92      0.89        25
 Red_Sindhi       0.85      0.80      0.82        15
       ...
 weighted avg       0.88      0.88      0.88       280

🎯 PER-CLASS METRICS:
Class Name           Accuracy     Precision    Recall       F1 Score
--------------------------------------------------------------------
Gir                  0.9500       0.9000       0.8500       0.8700
Sahiwal              0.9200       0.8700       0.9200       0.8900
Red_Sindhi           0.9100       0.8500       0.8000       0.8200
...
```

## Output Files

After running the script, you'll get:

1. **evaluation_results.json** - All metrics in JSON format (for programmatic use)
2. **confusion_matrix.png** - Visual representation of classification performance

## Understanding the Metrics

### Accuracy
- Best for balanced datasets
- Percentage of correct predictions out of total
- Formula: (TP + TN) / (TP + TN + FP + FN)

### Precision
- "Of what we predicted as positive, how much was actually correct?"
- Important when false positives are costly
- Formula: TP / (TP + FP)

### Recall
- "Of what was actually positive, how much did we catch?"
- Important when false negatives are costly
- Formula: TP / (TP + FN)

### F1 Score
- Harmonic mean of Precision and Recall
- Good for imbalanced datasets
- Formula: 2 × (Precision × Recall) / (Precision + Recall)

### G-Mean Score
- Geometric mean of per-class recalls
- Especially useful for imbalanced multi-class datasets
- Balances performance across all classes

### AUC-ROC
- Area Under the Receiver Operating Characteristic Curve
- Measures ability to distinguish between classes
- Range: 0 to 1 (1 = perfect classifier)

### Confusion Matrix
- Shows which classes are being confused with each other
- Diagonal elements = correct predictions
- Off-diagonal elements = misclassifications

## Troubleshooting

### "No images found"
- Check that your test data directory path is correct
- Ensure images have extensions: .jpg, .jpeg, .png, .gif, .bmp
- Make sure you have read permissions

### Model loading fails
- Verify `best_model.pth` exists
- Verify `metadata.json` exists
- Check that `model_path` and `metadata_path` in CONFIG are correct

### Out of memory
- Reduce `batch_size` in CONFIG (try 16 or 8)
- Use CPU instead of GPU (set device to 'cpu')

### No CUDA/GPU detected
- Set `device` to 'cpu' in CONFIG
- Processing will be slower but still works

## Advanced: Custom Metrics

To add custom metrics, edit the `evaluate_model()` function and add your calculation:

```python
# Example: Add top-5 accuracy
top5_accuracy = topk_accuracy(all_preds, all_labels, k=5)
metrics['top5_accuracy'] = top5_accuracy
```

## Questions?

If your test data is in a specific format, let me know and I can help adjust the scripts!

---
**Created**: 2024
**Model**: EfficientNetV2-M (36 Cattle Breeds)
