#!/bin/bash

# Inference path: JSONL file path or model inference dir path
### - Directly give the JSONL path if eval acc of a specific dataset
### - Give only model dir without JSONL path if eval vision-dependent acc
JSONL_PATH="./infer_outputs/PAPOGalaxy-PAPO-G-Qwen2.5-VL-7B-best_val/hiyouga_geometry3k.jsonl"

# Default values
N_ROLLOUT=8
OUTPUT_DIR="./eval_results"
ACC_FUNCTION_TYPE="boxed_math"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: bash run_eval.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file JSONL_PATH         Path to JSONL file with model responses (required)"
    echo "  -a, --acc-func FUNC_TYPE      Accuracy function type (default: boxed_math)"
    echo "  -n, --n-rollout N             Number of rollouts to evaluate (default: 8)"
    echo "  -o, --output-dir DIR          Output directory for results (default: ./eval_results)"
    echo "  -s, --script SCRIPT_PATH      Path to evaluation script (default: run_eval.py)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run_eval.sh -f responses.jsonl"
    echo "  bash run_eval.sh -f responses.jsonl -n 16 -a boxed_math"
    echo "  bash run_eval.sh --file /path/to/responses.jsonl --output-dir ./results"
}

# Parse command line arguments
PYTHON_SCRIPT="./papo_eval/run_eval.py"
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            JSONL_PATH="$2"
            shift 2
            ;;
        -a|--acc-func)
            ACC_FUNCTION_TYPE="$2"
            shift 2
            ;;
        -n|--n-rollout)
            N_ROLLOUT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$JSONL_PATH" ]]; then
    print_error "JSONL file path is required!"
    show_usage
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python evaluation script not found: $PYTHON_SCRIPT"
    print_info "Make sure run_eval.py is in the current directory or specify the correct path with -s"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract filename without extension for output naming
FILENAME=$(basename "$JSONL_PATH" .jsonl)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/${FILENAME}_eval_${TIMESTAMP}.json"

print_info "Starting evaluation..."
print_info "JSONL file: $JSONL_PATH"
print_info "Accuracy function: $ACC_FUNCTION_TYPE"
print_info "Number of rollouts: $N_ROLLOUT"
print_info "Output file: $OUTPUT_FILE"

# Run the evaluation
python "$PYTHON_SCRIPT" \
    --jsonl_path "$JSONL_PATH" \
    --acc_function_type "$ACC_FUNCTION_TYPE" \
    --n_rollout "$N_ROLLOUT" \
    --output_path "$OUTPUT_FILE"

# Check if evaluation was successful
if [[ $? -eq 0 ]]; then
    print_success "Evaluation completed successfully!"
    print_info "Results saved to: $OUTPUT_FILE"
    
    # Extract and display key metrics from the output file
    if [[ -f "$OUTPUT_FILE" ]]; then
        print_info "Evaluation Summary:"
        python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    print(f\"Test Samples: {int(results['total_examples'] / results['n_rollout'])}\")
    print(f\"  Total Examples: {data.get('total_examples', 'N/A')}\")
    print(f\"  Number of Rollout: {data.get('n_rollout', 'N/A')}\")
    print(f\"  Mean Accuracy @ {data.get('n_rollout', 'N/A')}: {data.get('avg_mean_accuracy', 0) * 100:.10f} %\")
except Exception as e:
    print(f\"  Could not parse results: {e}\")
"
    fi
else
    print_error "Evaluation failed!"
    exit 1
fi
