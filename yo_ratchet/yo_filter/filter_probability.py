from typing import Dict


def passes_probability_threshold(
    class_id: str,
    probability: float,
    lower_prob_thresholds: Dict[int, float],
    upper_prob_thresholds: Dict[int, float],
) -> bool:
    """
    Checks bounding box prediction confidence level to see whether it exceeds any
    "min_prob" value corresponding to the class_id in the thresholds json file.

    """
    lower_threshold = lower_prob_thresholds.get(int(class_id), 0.1)
    upper_threshold = upper_prob_thresholds.get(int(class_id), 1.0)
    if lower_threshold <= probability <= upper_threshold:
        return True
    else:
        return False
