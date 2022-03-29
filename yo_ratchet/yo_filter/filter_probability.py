from typing import Dict


def passes_probability_threshold(
    class_id: str,
    probability: float,
    prob_thresholds: Dict[int, float],
    looseness: float = 1.0,
) -> bool:
    """
    Checks bounding box prediction confidence level to see whether it exceeds any
    "min_prob" value corresponding to the class_id in the thresholds json file.

    """
    threshold = prob_thresholds[int(class_id)] / looseness
    if threshold and probability < threshold:
        return False
    else:
        return True
