import json
from pathlib import Path
from typing import List, Optional, Tuple

from yo_ratchet.yo_valuate.reference_csv import get_thresholds
from yo_ratchet.yo_filter.filter_central import defect_is_central
from yo_ratchet.yo_filter.filter_classes import insufficient_expectation
from yo_ratchet.yo_filter.filter_probability import passes_probability_threshold
from yo_ratchet.yo_filter.filter_size import passes_size_filters


def get_classes_info(classes_json_path: Path):
    with open(str(classes_json_path), "r") as json_obj:
        classes_info = json.load(json_obj)
    return classes_info


def apply_filters(
    lines: List[str],
    classes_json_path: Path,
    object_threshold_width: float = 0.02,
    filter_horizon: Optional[float] = None,
    wedge_constants: Optional[Tuple[float, float]] = None,
    wedge_gradients: Optional[Tuple[float, float]] = None,
    classes_to_remove: Optional[List[int]] = None,
    marginal_classes: Optional[List[str]] = None,  # e.g. ["2", "5"]
    min_count_marginal: Optional[int] = None,
    looseness: float = 1.0,
) -> List[List]:
    """
    Takes a list of predictions strings (class, yolo coordinates (scaled 0-1) plus probability)
    e.g.::
        [
            [class_id, centroid_x, centroid_y, width, height, probability],
            [class_id, centroid_x, centroid_y, width, height, probability],
            ...
            [class_id, centroid_x, centroid_y, width, height, probability],
        ]

    and filters the defect for localisation in the central wedge, and according to
    defect width and probability.

    Optionally filters for defects only in the lower part of the image below `image_horizon`
    where image horizon is a y-axis value scaled 0-1 like with 0 being at the top of the
    image and 1 at the bottom (same as for yolo coordinates).

    """
    new_lines = []
    classes_info = get_classes_info(classes_json_path=classes_json_path)
    prob_thresholds = get_thresholds(classes_info=classes_info)
    for line in lines:
        line = line.strip().split(" ")
        class_id = line[0]
        line = line[1:]
        line = [float(el) for el in line]
        yolo_box = line[0:5]
        centroid_y = yolo_box[1]
        prob = line[4] if len(line) >= 5 else None
        width = yolo_box[2]
        if classes_to_remove and int(class_id) in classes_to_remove:
            continue
        elif prob and not passes_probability_threshold(
            class_id=class_id,
            probability=prob,
            prob_thresholds=prob_thresholds,
            looseness=looseness,
        ):
            continue
        elif not passes_size_filters(
            class_id=class_id, yolo_box=yolo_box, classes_info=classes_info
        ):
            continue
        elif filter_horizon and centroid_y < filter_horizon:
            continue
        elif (
            wedge_gradients
            and wedge_constants
            and not defect_is_central(
                yolo_coordinates=yolo_box,
                wedge_gradients=wedge_gradients,
                wedge_constants=wedge_constants,
            )
        ):
            continue  # Object does not fall within the wedge shaped envelop
        elif width < object_threshold_width:
            continue
        else:
            pass  # At this point, all filters have been passed so append data to result

        line.insert(0, int(class_id))
        new_lines.append(line)

    if min_count_marginal and insufficient_expectation(
        new_lines=new_lines,
        marginal_classes=marginal_classes,
        min_count_marginal=min_count_marginal,
    ):
        new_lines = []  # Nulls all detections for this image: may be false alarms.
    else:
        pass  # Do not apply any filter in relation to aggregated expectation

    return new_lines
