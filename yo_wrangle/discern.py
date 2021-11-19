from pathlib import Path
from typing import List

from yo_wrangle.common import get_all_txt_recursive


def get_list_misclassified_image_names(
    inferred_annotations_dir: Path,
    ground_truths_dir: Path,
    confidence_threshold: float,
) -> List[str]:
    """
    This function examines the bounding box ground truths to inferred bounding boxes
    and returns a list of image names that have CLASSIFICATION inference errors.

    To be used for discerning the most valuable training data which is available for
    addition to a training data set.

    The list is subject to a constraint on the minimum level of prediction confidence to
    ensure that the most mistaken data is harvested.

    NOTE::
        Does not work on a bounding box intersection over union (IOU) threshold. Instead,
        the bounding box data is converted to a per image classification. Before applying
        the confidence_threshold constraint, the preliminary list will contain any images
        that have at least one False Positive or False Negative.

    """
    for truth_path in get_all_txt_recursive(root_dir=ground_truths_dir):
        with open(truth_path, "r") as truth_file:
            truth_lines = truth_file.readlines()


# def get_yolo_mis_classifications(
#     detected_annotations_path: Path,
#     truth_annotations: Path,
# ) -> List[Path]:
#     """
#     Idea is to collect the most valuable data for adding to training datasets.
#     Finds images that are predicted wrongly by a model, both false positives and
#     false negatives.
#
#     Could also be used to do quality control on existing training data.
#
#     Returns a list of paths to the misclassified images.
#     """
#     for detected_annotation_path in get_all_txt_recursive(detected_annotations_path):
#         pass  # (detected_annotations):