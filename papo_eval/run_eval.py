import os
import json
import argparse
from statistics import mean
from typing import List, Dict, Any
from eval_utils import ACC_FUNCTION_MAP


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_vision_dependent_jsonl(file_path: str) -> List[Dict[str, Any]]:
    vision_dependent_jsonl = [
        "BUAADreamer_clevr_count_70k.jsonl",
        "MMMU_MMMU_Pro.jsonl",
        "AI4Math_MathVerse.jsonl",
    ]
    vision_dependent_data = []
    for jsonl_name in vision_dependent_jsonl:
        jsonl_file_path = os.path.join(file_path, jsonl_name)
        vision_dependent_data += load_jsonl(jsonl_file_path)

    return vision_dependent_data

def evaluate_jsonl_responses(
    jsonl_path: str, 
    acc_function_type: str = "boxed_math",
    n_rollout: int = 8
) -> Dict[str, float]:
    """
    Evaluate JSONL responses and compute avg@n_rollout max and mean accuracies.
    
    Args:
        jsonl_path: Path to JSONL file with saved responses
        acc_function_type: Type of accuracy function to use
        n_rollout: Number of rollouts to evaluate (default: 8)
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Load the data
    if jsonl_path.endswith('.jsonl'):
        data = load_jsonl(jsonl_path)
    else:
        data = load_vision_dependent_jsonl(jsonl_path)
    
    # Eval function
    acc_func = ACC_FUNCTION_MAP.get(acc_function_type)
    
    if acc_func is None:
        raise ValueError(f"Unknown accuracy function type: {acc_function_type}")
    
    avg_accs = []
    all_results = {}
    
    print(f"Evaluating {len(data)} examples...")
    
    for i, example in enumerate(data):
        # Extract the prediction and ground truth
        test_id = example.get('id', '')
        prediction = example.get('predict', '')
        ground_truth = example.get('label', '')
        prompt = example.get('prompt', '')
        qa_key = f"Test ID: {test_id}\nQ:\n{prompt}\nA:\n{ground_truth}"

        if qa_key not in all_results:
            all_results[qa_key] = []
        
        acc = acc_func(prediction, ground_truth)
        all_results[qa_key].append(acc)
    
    for qa_key in all_results:
        assert len(all_results[qa_key]) % n_rollout == 0

        if len(all_results[qa_key]) > n_rollout:
            for idx in range(0, len(all_results[qa_key]), n_rollout):
                sub_accs = all_results[qa_key][idx:idx+n_rollout]
                avg_accs.append(mean(all_results[qa_key]))

        else:
            avg_accs.append(mean(all_results[qa_key]))
    
    # Calculate overall acc
    overall_avg_acc = mean(avg_accs) if avg_accs else 0.0
    
    results = {
        "n_rollout": n_rollout,
        "total_examples": len(data),
        "avg_mean_accuracy": overall_avg_acc,
        "individual_results": all_results
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL responses with avg@n max and mean accuracy."
    )
    
    parser.add_argument(
        "--jsonl_path", required=True,
        help="Path to JSONL file with model responses."
    )
    parser.add_argument(
        "--acc_function_type", default="boxed_math",
        help="Type of accuracy computation function."
    )
    parser.add_argument(
        "--n_rollout", type=int, default=8,
        help="Number of rollouts to evaluate (default: 8)."
    )
    parser.add_argument(
        "--output_path", default=None,
        help="Path to save detailed evaluation results (optional)."
    )
    
    args = parser.parse_args()
    
    # Evaluate the responses
    results = evaluate_jsonl_responses(
        args.jsonl_path,
        args.acc_function_type,
        args.n_rollout
    )
    
    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Test Samples: {int(results['total_examples'] / results['n_rollout'])}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Number of Rollout: {results['n_rollout']}")
    print(f"Mean Accuracy @ {results['n_rollout']}: {results['avg_mean_accuracy'] * 100:.10f} %")
    
    # Save detailed results if requested
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nDetailed results saved to: {args.output_path}")

if __name__ == "__main__":
    main()