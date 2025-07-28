
import re
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def compute_accuracy_boxed_math(predict: str, ground_truth: str) -> float:
    predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

ACC_FUNCTION_MAP = {
    "boxed_math": compute_accuracy_boxed_math,
}