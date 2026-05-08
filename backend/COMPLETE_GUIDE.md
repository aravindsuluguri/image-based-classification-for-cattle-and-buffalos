# 📊 Complete Model Evaluation & Testing Guide

## 🎯 Quick Start (TL;DR)

```bash
cd cattle\ updated/backend
bash run_evaluation.sh --test-dir /path/to/test/images
```

That's it! The script will:
1. ✅ Load your model and metadata
2. ✅ Prepare test data
3. ✅ Run comprehensive evaluation
4. ✅ Generate detailed metrics
5. ✅ Create visualizations

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| **test_model.py** | Main evaluation script with all metrics |
| **prepare_test_data.py** | Organize images by class for evaluation |
| **run_evaluation.sh** | One-command quick start script |
| **requirements_testing.txt** | All Python dependencies |
| **TESTING_README.md** | Detailed documentation |

---

## 🚀 Step-by-Step Usage

### Step 1: Install Dependencies
```bash
cd cattle\ updated/backend
pip install -r requirements_testing.txt
```

### Step 2: Prepare Your Test Data

**Option A: Already organized by class**
```
test_data/
├── Gir/
├── Sahiwal/
├── Red_Sindhi/
└── ...
```
Skip to Step 3.

**Option B: Unorganized images**
```bash
python prepare_test_data.py \
  --input-dir /path/to/images \
  --output-dir ./test_data \
  --metadata-path ./metadata.json
```

The script will:
- Look for class names in filenames (e.g., "Gir_001.jpg")
- Organize images into class folders
- Show organization summary

### Step 3: Run Evaluation

**Method 1: Using the quick start script (Recommended)**
```bash
bash run_evaluation.sh --test-dir ./test_data
```

**Method 2: Direct Python execution**
```bash
python test_model.py
```
*(First edit test_model.py to set correct test_data_dir)*

**Method 3: Python with custom paths**
```python
from test_model import *

CONFIG['test_data_dir'] = './your_test_data'
CONFIG['model_path'] = './best_model.pth'
CONFIG['batch_size'] = 32  # Adjust based on GPU memory

main()
```

### Step 4: Review Results

After evaluation, you'll have:

**evaluation_results.json** - All metrics
```json
{
  "accuracy": 0.8750,
  "precision_macro": 0.8621,
  "recall_macro": 0.8545,
  "f1_macro": 0.8579,
  "gmean": 0.8523,
  "mcc": 0.8412,
  "auc_roc_macro": 0.9876,
  "classification_report": "..."
}
```

**confusion_matrix.png** - Visual heatmap

---

## 📊 Metrics Explained

### Core Metrics

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 0-1 | Overall correctness of predictions |
| **Precision** | 0-1 | Of predicted positives, how many are correct |
| **Recall** | 0-1 | Of actual positives, how many we caught |
| **F1 Score** | 0-1 | Harmonic mean of Precision & Recall |
| **G-Mean** | 0-1 | Geometric mean (good for imbalanced data) |
| **AUC-ROC** | 0-1 | Discrimination ability (higher = better) |

### Example Interpretation

```
Model Performance Summary:
────────────────────────────────────────
🎯 Accuracy:     87.5%  ← 87.5% of predictions correct
📍 Precision:    86.2%  ← When we predict a breed, 86% chance it's correct
🎪 Recall:       85.5%  ← We correctly identify 85.5% of breeds
⭐ F1 Score:     85.8%  ← Good trade-off between precision & recall
📈 G-Mean:       85.2%  ← Balanced performance across all classes
🔮 AUC-ROC:      98.8%  ← Excellent discrimination between breeds
```

---

## 🔍 Common Use Cases

### Case 1: Quick Test Before Full Evaluation
```bash
bash run_evaluation.sh --test-dir /path/to/images --quick
```
Creates subset with 5 images per class for fast testing.

### Case 2: Evaluate on CPU (No GPU)
```bash
bash run_evaluation.sh --test-dir /path/to/images --no-gpu
```

### Case 3: Evaluate specific class performance
Edit test_model.py and look for `metrics['accuracy_per_class']`:
```python
# Shows performance for each breed
for breed, accuracy in metrics['accuracy_per_class'].items():
    print(f"{breed}: {accuracy:.4f}")
```

### Case 4: Find problem classes (low performance)
```bash
python -c "
import json
with open('evaluation_results.json') as f:
    metrics = json.load(f)

# Classes with < 80% accuracy
problem_classes = {
    k: v for k, v in metrics['accuracy_per_class'].items() 
    if v < 0.80
}

if problem_classes:
    print('🚨 Classes needing improvement:')
    for cls, acc in sorted(problem_classes.items(), key=lambda x: x[1]):
        print(f'  {cls}: {acc:.2%}')
"
```

---

## 🛠️ Configuration Options

Edit `test_model.py` CONFIG section:

```python
CONFIG = {
    'model_path': './best_model.pth',       # Model weights file
    'metadata_path': './metadata.json',     # Class names & config
    'test_data_dir': './test_data',         # Test images location
    'img_size': 384,                        # Input image size
    'batch_size': 32,                       # Increase for faster eval
    'device': 'cuda',                       # 'cuda' or 'cpu'
    'seed': 42,                             # Reproducibility
}
```

### Tuning Tips

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `batch_size` (try 16/8) |
| Slow Evaluation | Increase `batch_size` (if VRAM allows) |
| Different Results | Keep `seed` constant for reproducibility |
| CPU vs GPU | Change `device` to 'cpu' or 'cuda' |

---

## 📈 Interpreting Confusion Matrix

The confusion matrix shows:
- **Diagonal**: Correct predictions ✅
- **Off-diagonal**: Misclassifications ❌

```
Example:
            Predicted
Actual  Gir Sahiwal Red_Sindhi
Gir      18    2       0
Sahiwal   1   22       2
Red_S     0    1      14
```

**Analysis:**
- Gir: 90% correct (18/20)
- Sahiwal: 88% correct (22/25)
- Sahiwal confused with Red_Sindhi 2 times

---

## 🐛 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision
```

### ❌ "No images found"
- Check test directory path is correct
- Verify image extensions: .jpg, .jpeg, .png
- Ensure directory has read permissions

### ❌ "CUDA out of memory"
```bash
# Option 1: Reduce batch size
# Edit CONFIG: 'batch_size': 8

# Option 2: Use CPU
# Edit CONFIG: 'device': 'cpu'
```

### ❌ "Model loading failed"
```bash
# Check files exist
ls -la best_model.pth
ls -la metadata.json

# Verify model format
python -c "import torch; torch.load('best_model.pth')"
```

---

## 📝 Example Workflow

### Step 1: Organize test data
```bash
python prepare_test_data.py \
  --input-dir ~/Downloads/cattle_test_images \
  --output-dir ./test_data
```

### Step 2: Run quick evaluation
```bash
bash run_evaluation.sh --test-dir ./test_data --quick
```

### Step 3: Review results
```bash
cat evaluation_results.json | python -m json.tool
# and open confusion_matrix.png
```

### Step 4: If results look good, run full evaluation
```bash
bash run_evaluation.sh --test-dir ./test_data
```

### Step 5: Analyze problem areas
```bash
python -c "
import json
with open('evaluation_results.json') as f:
    m = json.load(f)
print(m['classification_report'])
"
```

---

## 💡 Advanced: Custom Analysis

Add this to analyze specific metrics:

```python
import json
import numpy as np

with open('evaluation_results.json') as f:
    metrics = json.load(f)

# 1. Find best/worst classes
f1_scores = metrics['f1_per_class']
best = max(f1_scores.items(), key=lambda x: x[1])
worst = min(f1_scores.items(), key=lambda x: x[1])

print(f"Best performer:  {best[0]} (F1: {best[1]:.4f})")
print(f"Worst performer: {worst[0]} (F1: {worst[1]:.4f})")

# 2. Check if model is balanced across classes
print(f"\nF1 Score Range: {min(f1_scores.values()):.4f} - {max(f1_scores.values()):.4f}")
print(f"F1 Score Std Dev: {np.std(list(f1_scores.values())):.4f}")

# 3. Precision-Recall trade-off
for class_name in list(f1_scores.keys())[:5]:
    prec = metrics['precision_per_class'][class_name]
    rec = metrics['recall_per_class'][class_name]
    print(f"{class_name}: Precision={prec:.4f}, Recall={rec:.4f}")
```

---

## 📞 Support

**Questions about:**
- Test data format? → See TESTING_README.md
- Metrics calculation? → See code comments in test_model.py
- Script errors? → Run with `--help` flag

---

## ✅ Checklist

Before running evaluation:
- [ ] Test data is organized or path is known
- [ ] best_model.pth exists
- [ ] metadata.json exists
- [ ] Dependencies installed (`pip install -r requirements_testing.txt`)
- [ ] Python 3.8+ available
- [ ] Sufficient disk space for outputs (~1MB per 100 images)

After evaluation:
- [ ] evaluation_results.json generated
- [ ] confusion_matrix.png generated
- [ ] All metrics are reasonable (not NaN)
- [ ] Results match expectations

---

**Version**: 1.0  
**Model**: EfficientNetV2-M  
**Classes**: 36 Cattle Breeds  
**Last Updated**: 2024
