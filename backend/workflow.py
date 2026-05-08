#!/usr/bin/env python3
"""
Complete Evaluation Workflow
================================================================================
Master script that handles all steps: verification, data preparation,
evaluation, and analysis.

Usage:
    python workflow.py                          # Interactive mode
    python workflow.py --auto                   # Automatic mode (default settings)
    python workflow.py --test-dir ./test_data   # With custom test dir

================================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a header."""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def run_command(cmd, description, show_output=True):
    """Run a command and return success status."""
    print_info(f"Running: {description}")
    try:
        if show_output:
            result = subprocess.run(cmd, shell=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            print_success(f"{description} completed")
            return True
        else:
            print_error(f"{description} failed")
            return False
    except Exception as e:
        print_error(f"Error running {description}: {e}")
        return False


def step_1_verify_setup():
    """Step 1: Verify setup."""
    print_header("STEP 1: VERIFY SETUP")
    
    print("Running setup verification...\n")
    if run_command("python verify_setup.py", "Setup verification"):
        return True
    else:
        print_error("Setup verification failed. Please fix the issues above.")
        return False


def step_2_prepare_test_data(test_dir):
    """Step 2: Prepare test data."""
    print_header("STEP 2: PREPARE TEST DATA")
    
    if test_dir and os.path.isdir(test_dir):
        print_info(f"Test data directory provided: {test_dir}")
        
        # Check if organized
        subdirs = [d for d in Path(test_dir).iterdir() if d.is_dir()]
        if subdirs:
            print_success(f"Found {len(subdirs)} subdirectories (appears organized)")
            return True
        else:
            # Try to organize
            print_warning("Images not organized by class. Organizing now...")
            cmd = f"python prepare_test_data.py --input-dir {test_dir} --output-dir ./test_data"
            if run_command(cmd, "Data preparation"):
                return True
    else:
        print_warning("No test directory specified or found")
        print_info("Create test_data directory with images organized by breed:")
        print("   test_data/")
        print("   ├── Gir/")
        print("   │   ├── img1.jpg")
        print("   │   └── ...")
        print("   ├── Sahiwal/")
        print("   └── ...")
        
        # Check default locations
        default_paths = ['./test_data', '../../../results/outputs']
        for path in default_paths:
            if os.path.isdir(path):
                print_success(f"Found test data at: {path}")
                return True
        
        print_warning("No test data found. You can still run evaluation with sample data.")
        return True  # Don't fail, user might add data manually


def step_3_run_evaluation():
    """Step 3: Run evaluation."""
    print_header("STEP 3: RUN MODEL EVALUATION")
    
    print("Evaluating model on test data...\n")
    if run_command("python test_model.py", "Model evaluation"):
        return True
    else:
        print_error("Model evaluation failed")
        return False


def step_4_analyze_results():
    """Step 4: Analyze results."""
    print_header("STEP 4: ANALYZE RESULTS")
    
    if not os.path.exists('evaluation_results.json'):
        print_warning("Results file not found. Skipping analysis.")
        return False
    
    print("Analyzing evaluation results...\n")
    if run_command("python analyze_results.py", "Results analysis"):
        return True
    else:
        print_warning("Results analysis encountered issues but evaluation data is available")
        return True


def print_summary(success):
    """Print workflow summary."""
    print_header("WORKFLOW SUMMARY")
    
    if success:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ ALL STEPS COMPLETED SUCCESSFULLY!{Colors.ENDC}\n")
        print("📊 Output Files Generated:")
        print("   • evaluation_results.json  - All metrics in JSON format")
        print("   • confusion_matrix.png     - Confusion matrix visualization")
        print("   • per_class_f1_scores.png  - F1 scores by class")
        print("   • metrics_comparison.png   - Metrics comparison chart")
        print("   • precision_recall_scatter.png - Precision vs Recall plot")
        
        print("\n📖 Next Steps:")
        print("   1. Review evaluation_results.json for all metrics")
        print("   2. Examine confusion_matrix.png to see misclassification patterns")
        print("   3. Check per_class_f1_scores.png to identify weak classes")
        print("   4. Use metrics_comparison.png to see metric trade-offs")
        
        print("\n📚 Documentation:")
        print("   • QUICK_START.txt     - Quick reference")
        print("   • TESTING_README.md   - Detailed manual")
        print("   • COMPLETE_GUIDE.md   - Complete reference")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ WORKFLOW ENCOUNTERED ISSUES{Colors.ENDC}\n")
        print("Please review the error messages above and fix any issues.")


def interactive_mode():
    """Interactive workflow mode."""
    print_header("🚀 INTERACTIVE EVALUATION WORKFLOW")
    
    print("This workflow will:")
    print("  1. Verify all dependencies and model files")
    print("  2. Prepare test data")
    print("  3. Run comprehensive model evaluation")
    print("  4. Analyze and visualize results")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return False
    
    steps = [
        ("Setup Verification", step_1_verify_setup, None),
        ("Test Data Preparation", step_2_prepare_test_data, None),
        ("Model Evaluation", step_3_run_evaluation, None),
        ("Results Analysis", step_4_analyze_results, None),
    ]
    
    results = []
    for i, (step_name, step_func, *args) in enumerate(steps, 1):
        print(f"\n{Colors.BOLD}Progress: Step {i}/{len(steps)}{Colors.ENDC}")
        
        try:
            arg = args[0] if args else None
            success = step_func(arg) if arg is not None else step_func()
            results.append(success)
            
            if not success and step_name != "Results Analysis":
                response = input("\nContinue despite failure? (y/n): ").strip().lower()
                if response != 'y':
                    print("Workflow cancelled.")
                    return False
        except KeyboardInterrupt:
            print("\nWorkflow interrupted by user.")
            return False
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                return False
    
    return all(results[:3])  # Check if main steps succeeded


def auto_mode(test_dir=None):
    """Automatic workflow mode."""
    print_header("🚀 AUTOMATIC EVALUATION WORKFLOW")
    
    steps = [
        ("Setup Verification", step_1_verify_setup, None),
        ("Test Data Preparation", step_2_prepare_test_data, test_dir),
        ("Model Evaluation", step_3_run_evaluation, None),
        ("Results Analysis", step_4_analyze_results, None),
    ]
    
    results = []
    for i, (step_name, step_func, *args) in enumerate(steps, 1):
        print(f"\n{Colors.BOLD}Progress: Step {i}/{len(steps)}{Colors.ENDC}")
        
        try:
            arg = args[0] if args else None
            success = step_func(arg) if arg is not None else step_func()
            results.append(success)
            
            if not success and step_name != "Results Analysis":
                print_error(f"Step failed: {step_name}")
                if step_name == "Setup Verification":
                    print("Cannot continue without successful setup verification.")
                    return False
        except Exception as e:
            print_error(f"Error in {step_name}: {e}")
            if step_name == "Setup Verification":
                return False
    
    return all(results[:3])


def main():
    parser = argparse.ArgumentParser(
        description="Complete Model Evaluation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow.py                          # Interactive mode
  python workflow.py --auto                   # Automatic mode
  python workflow.py --auto --test-dir ./data # Auto with custom test dir
        """
    )
    
    parser.add_argument('--auto', action='store_true',
                        help='Run in automatic mode (non-interactive)')
    parser.add_argument('--test-dir', type=str,
                        help='Path to test data directory')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MODEL EVALUATION WORKFLOW" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print(f"{Colors.ENDC}")
    
    try:
        if args.auto:
            success = auto_mode(args.test_dir)
        else:
            success = interactive_mode()
        
        print_summary(success)
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Workflow interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
