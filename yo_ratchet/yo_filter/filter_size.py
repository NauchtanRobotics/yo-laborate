from typing import List, Dict

MIN_WIDTH_TO_BOX = 0.015


def defect_exceeds_width_threshold(
    class_id: str, yolo_box: List[float], classes_info: Dict
) -> bool:
    """
    Checks bounding box width to see whether it exceeds any "min_width_to_box"
    value corresponding to the class_id in the thresholds json file.

    If no "min_width_to_box" is specified then the default MIN_WIDTH_TO_BOX is
    applied.

    """
    defect_width = yolo_box[2]
    min_width_to_box = classes_info[class_id].get("min_width_to_box", MIN_WIDTH_TO_BOX)

    if defect_width < min_width_to_box:
        return False  # Defect is too narrow
    else:
        return True


def defect_exceeds_area_threshold(
    class_id: str, yolo_box: List[float], classes_info: Dict
) -> bool:
    """
    Checks bounding box area to see whether it exceeds any "min_area" value
    corresponding to the class_id in the thresholds json file.

    """
    defect_area = yolo_box[2] * yolo_box[3]
    min_width_to_box = classes_info[class_id].get("min_area")

    if min_width_to_box and defect_area < min_width_to_box:
        return False  # Defect area is too small
    else:
        return True


def passes_size_filters(
    class_id: str, yolo_box: List[float], classes_info: Dict
) -> bool:
    """
    An umbrella function to combine all bounding box dimensional related
    filters in this module.

    """
    if not defect_exceeds_width_threshold(
        class_id=class_id, yolo_box=yolo_box, classes_info=classes_info
    ):
        return False
    elif not defect_exceeds_area_threshold(
        class_id=class_id, yolo_box=yolo_box, classes_info=classes_info
    ):
        return False
    else:
        return True
