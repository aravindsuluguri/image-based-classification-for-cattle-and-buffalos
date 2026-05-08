"""
Quick Verification Script
================================================================================
Verifies that all dependencies, model files, and test data are available
before running the full evaluation.

Usage: python verify_setup.py
"""

import os
import sys
import json
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🐍 Python Version Check")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        print(f"   ✅ Python {version} (OK)")
        return True
    else:
        print(f"   ❌ Python {version} (Need 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Dependencies Check")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'TIMM',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (MISSING)")
            missing.append(name)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install -r requirements_testing.txt")
        return False
    return True


def check_model_files():
    """Check if model and metadata files exist."""
    print("\n🧠 Model Files Check")
    
    files = {
        'best_model.pth': 'Model weights',
        'metadata.json': 'Model metadata',
    }
    
    all_exist = True
    for filename, description in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {filename} (NOT FOUND)")
            all_exist = False
    
    if not all_exist:
        print(f"\n   Make sure these files exist in: {os.getcwd()}")
    
    return all_exist


def check_metadata():
    """Validate metadata.json structure."""
    print("\n🔍 Metadata Validation")
    
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        required_keys = ['model_name', 'num_classes', 'class_names']
        for key in required_keys:
            if key in metadata:
                if key == 'class_names':
                    print(f"   ✅ {key} ({len(metadata[key])} classes)")
                else:
                    print(f"   ✅ {key}: {metadata[key]}")
            else:
                print(f"   ❌ {key} (MISSING)")
                return False
        
        return True
    except Exception as e:
        print(f"   ❌ Error reading metadata.json: {e}")
        return False


def check_test_data():
    """Check if test data directory exists."""
    print("\n📂 Test Data Check")
    
    default_paths = [
        '../../../results/outputs',
        './test_data',
        './test_images',
    ]
    
    found = False
    for path in default_paths:
        if os.path.isdir(path):
            # Count images
            img_count = 0
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        img_count += 1
            
            if img_count > 0:
                print(f"   ✅ Found: {path}")
                print(f"      └─ {img_count} images")
                found = True
                break
    
    if not found:
        print(f"   ⚠️  No test data found")
        print(f"      Checked: {', '.join(default_paths)}")
        print(f"      You can:")
        print(f"      1. Organize test images with: python prepare_test_data.py")
        print(f"      2. Create test_data/ directory and add images")
        print(f"      3. Specify path in test_model.py CONFIG")
    
    return found  # Not critical if missing


def check_gpu():
    """Check if GPU is available."""
    print("\n🎮 GPU Check")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU Available: {device_name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print(f"   ℹ️  GPU Not Available (will use CPU)")
            return False
    except Exception as e:
        print(f"   ⚠️  GPU Check Failed: {e}")
        return False


def main():
    print("="*70)
    print("🔍 MODEL EVALUATION SETUP VERIFICATION")
    print("="*70 + "\n")
    print(f"Current Directory: {os.getcwd()}\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Metadata", check_metadata),
        ("GPU/CUDA", check_gpu),
        ("Test Data", check_test_data),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append(result)
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    critical = results[:4]  # Python, Dependencies, Model Files, Metadata
    optional = results[4:]  # GPU, Test Data
    
    if all(critical):
        print("\n✅ All critical requirements met!")
        print("\n🚀 You can now run:")
        print("   python test_model.py")
        print("   OR")
        print("   bash run_evaluation.sh --test-dir ./test_data")
    else:
        print("\n❌ Some critical requirements missing!")
        print("\n   Please fix the issues above before running evaluation.")
        
        if not results[1]:  # Dependencies
            print("\n   Install dependencies:")
            print("   pip install -r requirements_testing.txt")
        
        if not results[2]:  # Model Files
            print("\n   Ensure best_model.pth and metadata.json exist in:")
            print(f"   {os.getcwd()}")
    
    if not results[4]:  # GPU
        print("\n   💡 Tip: Model will run on CPU (slower but works)")
    
    if not results[5]:  # Test Data
        print("\n   💡 Tip: Prepare test data using:")
        print("      python prepare_test_data.py --input-dir /path/to/images \\")
        print("                --output-dir ./test_data \\")
        print("                --metadata-path ./metadata.json")
    
    print("\n" + "="*70)
    
    return all(critical)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
