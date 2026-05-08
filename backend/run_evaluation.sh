#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# Quick Start Script for Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════
# Usage: bash run_evaluation.sh [options]
#
# Options:
#   --test-dir DIR      Path to test images directory
#   --no-gpu            Use CPU instead of GPU
#   --quick             Quick test (5 images per class)
#   --help              Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_DIR="../../../results/outputs"
USE_GPU=true
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "Usage: bash run_evaluation.sh [options]"
            echo ""
            echo "Options:"
            echo "  --test-dir DIR      Path to test images directory (default: ../../../results/outputs)"
            echo "  --no-gpu            Use CPU instead of GPU"
            echo "  --quick             Quick test (creates small subset for testing)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 Model Evaluation Quick Start${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "\n${YELLOW}📦 Checking dependencies...${NC}"
if [ ! -f "requirements_testing.txt" ]; then
    echo -e "${RED}❌ requirements_testing.txt not found${NC}"
    exit 1
fi

# Check if models exist
if [ ! -f "best_model.pth" ]; then
    echo -e "${RED}❌ best_model.pth not found in current directory${NC}"
    exit 1
fi

if [ ! -f "metadata.json" ]; then
    echo -e "${RED}❌ metadata.json not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Model files found${NC}"

# Install dependencies
echo -e "\n${YELLOW}📦 Installing dependencies...${NC}"
pip install -q -r requirements_testing.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check test directory
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${YELLOW}⚠️  Test directory not found: $TEST_DIR${NC}"
    echo -e "${YELLOW}   Please provide test images with --test-dir option${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Test data found: $TEST_DIR${NC}"

# Quick mode: create subset
if [ "$QUICK_MODE" = true ]; then
    echo -e "\n${YELLOW}🔍 Creating quick test subset...${NC}"
    python3 prepare_test_data.py \
        --input-dir "$TEST_DIR" \
        --output-dir ./test_data_quick \
        --metadata-path ./metadata.json \
        --subset \
        --subset-size 5
    TEST_DIR="./test_data_quick"
fi

# Run evaluation
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🧪 Running Model Evaluation...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}\n"

# Modify test_model.py temporarily with correct test directory if needed
python3 << 'PYTHON_RUNNER'
import sys
import os

# Read the test_model.py file
with open('test_model.py', 'r') as f:
    content = f.read()

# Update test data directory in config
test_dir = sys.argv[1] if len(sys.argv) > 1 else "../../../results/outputs"
content = content.replace(
    "'test_data_dir': '../../../results/outputs',",
    f"'test_data_dir': '{test_dir}',"
)

# Execute the modified script
exec(compile(content, 'test_model.py', 'exec'))
PYTHON_RUNNER "$TEST_DIR"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Evaluation Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${GREEN}📊 Output Files:${NC}"
echo -e "   • ${YELLOW}evaluation_results.json${NC}  - Detailed metrics in JSON format"
echo -e "   • ${YELLOW}confusion_matrix.png${NC}     - Confusion matrix visualization"

echo -e "\n${GREEN}Next Steps:${NC}"
echo -e "   1. Review the metrics in evaluation_results.json"
echo -e "   2. Check confusion_matrix.png for misclassifications"
echo -e "   3. Identify classes with low performance"

echo ""
