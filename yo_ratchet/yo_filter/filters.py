import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from yo_ratchet.yo_filter.unsupervised import (
    OutlierParams,
    get_delta_for_patch,
)
from yo_ratchet.yo_filter.filter_central import defect_is_central
from yo_ratchet.yo_filter.filter_classes import insufficient_expectation
from yo_ratchet.yo_filter.filter_probability import passes_probability_threshold
from yo_ratchet.yo_filter.filter_size import passes_size_filters

MIN_PROB_KEY = "min_prob"


def get_classes_info(classes_json_path: Path):
    with open(str(classes_json_path), "r") as json_obj:
        classes_info = json.load(json_obj)
    return classes_info


def apply_filters(
    lines: List[str],
    classes_info: Dict,
    lower_probability_coefficient: float = 0.7,
    upper_probability_coefficient: Optional[float] = None,
    object_threshold_width: float = 0.02,
    filter_horizon: Optional[float] = None,
    wedge_constants: Optional[Tuple[float, float]] = None,
    wedge_gradients: Optional[Tuple[float, float]] = None,
    classes_to_remove: Optional[List[int]] = None,
    marginal_classes: Optional[List[str]] = None,  # e.g. ["2", "5"]
    outlier_params: Optional[OutlierParams] = None,
    image_path: Optional[Path] = None,  # Only required if filtering outliers patches
    min_count_marginal: Optional[int] = None,
    remove_probability: bool = False,
    cpy_dst_outliers: Optional[Path] = None
) -> List[List]:
    """
    A function to filter object detections for use in production (not training data harvesting).

    Takes a list of yolo predictions lines (class, yolo coordinates (scaled 0-1) plus probability) for
    a single image, and filters out according to selected options including::
        * objects above a certain horizon (y_centroid < filter_horizon)
        * outside the central wedge
        * confidence below class specific min_prob
        * certain classes can be removed.
        * require a minimum count of a class for selected classes if the raw detections only
          included these marginal classes.
        * object width is narrower than the nominated object_threshold_width
        * object detected is an outlier from others based on unsupervised learning/imagenet resnet50 layer8.

    Not suitable for training data harvesting as it removes detections that are lower in confidence without
    providing opportunity for review.

    :param lines: E.g.::
        [
            [class_id, centroid_x, centroid_y, width, height, probability],
            [class_id, centroid_x, centroid_y, width, height, probability],
            ...
            [class_id, centroid_x, centroid_y, width, height, probability],
        ]
    :param classes_info: is the dict read in from classes.json. Provide class specific "min_prob" threshold.
    :param cpy_dst_outliers:
    """
    lower_prob_thresholds = get_lower_probability_thresholds(
        classes_info=classes_info,
        lower_probability_coefficient=lower_probability_coefficient,
    )
    upper_prob_thresholds = get_upper_probability_thresholds(
        classes_info=classes_info,
        upper_probability_coefficient=upper_probability_coefficient,
    )
    new_lines = []
    for line in lines:
        line = line.strip().split(" ")
        class_id = line[0]
        line = line[1:]
        line = [float(el) for el in line]
        yolo_box = line[0:4]
        centroid_y = yolo_box[1]
        prob = line[4] if len(line) >= 5 else None
        width = yolo_box[2]
        if classes_to_remove and int(class_id) in classes_to_remove:
            continue
        elif prob and not passes_probability_threshold(
            class_id=class_id,
            probability=prob,
            lower_prob_thresholds=lower_prob_thresholds,
            upper_prob_thresholds=upper_prob_thresholds,
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
        elif (
            outlier_params
            and int(class_id) in outlier_params.normalising_params
            and image_path
            and get_delta_for_patch(
                class_id=int(class_id),
                yolo_box=yolo_box,
                image_path=image_path,
                outlier_params=outlier_params,
            )
            > outlier_params.outlier_config.control_limit_coefficient
        ):  # expensive operation, so do this test last.
            print("Outlier of class " + class_id + " found in " + image_path.name)
            if cpy_dst_outliers is not None:
                shutil.copy(
                    src=str(image_path),
                    dst=str(cpy_dst_outliers / image_path.name),
                )
            else:
                pass
            continue
        else:
            pass  # At this point, all filters have been passed so append data to result

        if remove_probability:
            yolo_box.insert(0, int(class_id))
            new_lines.append(yolo_box)
        else:
            line.insert(0, int(class_id))
            new_lines.append(line)

    if (
        marginal_classes
        and min_count_marginal
        and len(new_lines) > 0
        and insufficient_expectation(
            new_lines=new_lines,
            marginal_classes=marginal_classes,
            min_count_marginal=min_count_marginal,
        )
    ):
        new_lines = []  # Nulls all detections for this image: may be false alarms.
    else:
        pass  # Do not apply any filter in relation to aggregated expectation

    return new_lines


def get_lower_probability_thresholds(
    classes_info,
    lower_probability_coefficient: float = 0.7,
):
    return {
        int(key): float(info_dict.get(MIN_PROB_KEY, 0.1)) * lower_probability_coefficient
        for key, info_dict in classes_info.items()
    }


def get_upper_probability_thresholds(
    classes_info,
    upper_probability_coefficient: Optional[float] = None,
):
    if upper_probability_coefficient is None:
        return {int(key): 1.0 for key, info_dict in classes_info.items()}
    else:
        return {
            int(key): float(info_dict.get(MIN_PROB_KEY, 0.1))
            * upper_probability_coefficient
            for key, info_dict in classes_info.items()
        }
