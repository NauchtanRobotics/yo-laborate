import numpy
import pandas
from pathlib import Path
from sklearn import metrics as skm
from tabulate import tabulate
from typing import Dict, List, Any, Optional

from yo_wrangle.common import get_all_jpg_recursive


def get_memberships(value: Any, group_to_members_mapping: Dict[str, List[Any]]) -> str:
    memberships = []
    for key, vals in group_to_members_mapping.items():
        if isinstance(vals[0], str):
            vals = [val.capitalize() for val in vals if val != ""]
            if value.capitalize() in vals:
                memberships.append(key)
        elif value in vals:
            memberships.append(key)
        else:
            pass  # No membership in any group identified this round
    return ", ".join(memberships)


def get_group_to_int_mapping(mapping: Dict[str, Any]):
    return {val: i for i, val in enumerate(mapping.keys())}


def _get_group_memberships_from_dataframe(
    df: pandas.DataFrame,
    csv_group_mappings: Dict[str, List[str]],
    image_key: str = "Photo_Name",
    classifications_key: str = "Class_Memberships",
) -> Dict[str, List[str]]:
    """
    Given a dataframe, this function examines the class memberships column
    (having a column label corresponding to the classification_key param)
    and determines group memberships as defined in the group_filters param.

    Example group_filters =
        {
            "Group_1": {"Juice", "Soda"},
            "Group_2": {"Beer", "Wine"}
        }

    Returns a dict wherein the keys corresponding to a unique image id
    (as provided by df[image_key]) and the values for each image is in the form
    of a list containing zero or more group memberships. E.g.
        {
            "Photo_1": [],
            "Photo_2": ["Group_1"],
            "Photo_3": ["Group_2"],
            "Photo_4: ["Group_1", "Group_2"],
         }

    """
    df = df[[image_key, classifications_key]]
    res = {val[image_key]: val[classifications_key].split() for _, val in df.iterrows()}
    res = {
        key: list({get_memberships(el, csv_group_mappings) for el in vals})
        for key, vals in res.items()
    }
    res = {key: [el for el in vals if el != ""] for key, vals in res.items()}
    return res


def get_thresholds(classes_info):
    return {
        int(key): info_dict.get("min_prob", 0.1)
        for key, info_dict in classes_info.items()
    }


def get_classification_performance(
    images_root: Path,
    truths_csv: Path,
    root_inferred_bounding_boxes: Path,
    csv_group_filters: Dict[str, List[str]],
    yolo_group_filters: Dict[str, List[int]],
    classes_info: Dict[str, Dict],
    image_key: str = "Photo_Name",
    classifications_key: str = "Final_Remedy",
    print_table: bool = True,
) -> pandas.DataFrame:
    """
    Sequence of operations:
    * df = get_actual_vs_inferred_df()
    * calculate metrics based on actual vs inferred using sklearn
    """
    df = get_actual_vs_inferred_df(
        images_root=images_root,
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        csv_group_filters=csv_group_filters,
        yolo_group_filters=yolo_group_filters,
        classes_info=classes_info,
        image_key=image_key,
        classifications_key=classifications_key,
    )
    groupings = get_group_to_int_mapping(mapping=csv_group_filters)
    results = {}
    for group_name, idx in groupings.items():
        y_truths = df["actual_classifications"]
        y_inferences = df["inferred_classifications"]
        count = y_truths.size
        assert count == y_inferences.size
        inferences = numpy.array([y[idx] for y in y_inferences])
        truths = numpy.array([y[idx] for y in y_truths])

        labels = None
        precision = skm.precision_score(
            y_true=truths,
            y_pred=inferences,
            labels=labels,
            pos_label=1,
            average="binary",
            sample_weight=None,
            zero_division="warn",
        )
        recall = skm.recall_score(
            y_true=truths,
            y_pred=inferences,
            labels=labels,
            pos_label=1,
            average="binary",
            sample_weight=None,
            zero_division="warn",
        )
        f1 = skm.f1_score(
            y_true=truths,
            y_pred=inferences,
            labels=labels,
            pos_label=1,
            average="binary",
            sample_weight=None,
            zero_division="warn",
        )
        results[group_name] = {
            "P": "{:.2f}".format(precision),
            "R": "{:.2f}".format(recall),
            "F1": "{:.2f}".format(f1),
        }
    df = pandas.DataFrame(results)
    if print_table:
        print(
            tabulate(
                df.transpose(),
                headers="keys",
                showindex="always",
                tablefmt="pretty",
            )
        )
    return df


def get_actual_vs_inferred_df(
    images_root: Path,
    truths_csv: Path,
    root_inferred_bounding_boxes: Path,
    csv_group_filters: Dict[str, List[str]],
    yolo_group_filters: Dict[str, List[int]],
    classes_info: Dict[str, Dict],
    image_key: str = "Photo_Name",
    classifications_key: str = "Final_Remedy",
) -> pandas.DataFrame:
    """
    A function for converting object detection data to classification.

    Achieves this by finding a unique list (set) of inferred class memberships
    which an image can claim based on YOLO annotation file, then compares
    this to the class memberships proclaimed in a truths_csv.

    Sequence of operations:
    * get_true_group_memberships
    * get_inferred_group_memberships
    * combine these results into a dataframe which is returned by this function.
    """
    results_dict = {}
    if (
        truths_csv.exists()
        and truths_csv.is_file()
        and truths_csv.suffix.lower() == ".csv"
    ):
        actual_group_memberships = get_group_memberships_truths(
            truths_csv=truths_csv,
            csv_group_filters=csv_group_filters,
            image_key=image_key,
            classifications_key=classifications_key,
        )
    else:
        raise RuntimeError("Path provided for root_ground_truths does not exist.")
    thresholds = get_thresholds(classes_info=classes_info)
    inferred_group_memberships = get_group_membership_inferences(
        images_root=images_root,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        csv_group_filters=csv_group_filters,
        yolo_group_filters=yolo_group_filters,
        thresholds=thresholds,
    )
    for image_path in sorted(get_all_jpg_recursive(img_root=images_root)):
        image_name = image_path.name
        if image_name not in actual_group_memberships:
            continue  # image not covered by truths
        actual_groups = actual_group_memberships[image_name]
        inferred_groups = inferred_group_memberships[image_name]
        results_dict[image_name] = {
            "actual_classifications": numpy.array(actual_groups),
            "inferred_classifications": inferred_groups,
        }
    df = pandas.DataFrame(results_dict)
    df = df.transpose()
    return df


def get_group_memberships_truths(
    truths_csv: Path,
    csv_group_filters: Dict[str, List[str]],
    image_key: str = "Photo_Name",
    classifications_key: str = "Final_Remedy",
) -> Dict[str, List[bool]]:
    """
    Gets a dict of truths where key is the image name and
    the values are a list of truth booleans wherein each element of the list
    corresponds to a group from the groups defined by the keys of
    csv_group_filters. E.g.

        {
            "Photo_1.jpg": [True, False],
            "Photo_2.jpg": [False, False],
         }
    """
    group_to_int_mapping = get_group_to_int_mapping(mapping=csv_group_filters)
    classifications = pandas.read_csv(
        filepath_or_buffer=str(truths_csv),
    )
    classifications = _get_group_memberships_from_dataframe(
        df=classifications,
        csv_group_mappings=csv_group_filters,
        image_key=image_key,
        classifications_key=classifications_key,
    )
    num_groups = len(csv_group_filters.keys())
    results_dict = {}
    for image_name, group_memberships in classifications.items():
        actual_groups = [False for i in range(num_groups)]
        for group_name in group_memberships:
            group_int = group_to_int_mapping.get(group_name, None)
            if group_int is not None:
                actual_groups[group_int] = True
        results_dict[image_name] = actual_groups
    return results_dict


def get_group_membership_inferences(
    images_root: Path,
    root_inferred_bounding_boxes: Path,
    csv_group_filters: Dict[str, List[str]],
    yolo_group_filters: Dict[str, List[int]],
    thresholds: Dict[int, float],
) -> Dict[str, List[bool]]:
    """
    Gets a dict of inferences where key is the image name and
    the values are a list of inference booleans wherein each element of the list
    corresponds to a group from the groups defined by the keys of
    csv_group_filters. E.g.

        {
            "Photo_1.jpg": [True, False],
            "Photo_2.jpg": [False, False],
         }
    """
    num_groups = len(csv_group_filters.keys())
    group_to_int_mapping = get_group_to_int_mapping(mapping=csv_group_filters)
    results_dict = {}
    count_positive = 0
    for image_path in get_all_jpg_recursive(img_root=images_root):
        inferred_annotations_path = (
            root_inferred_bounding_boxes / f"{image_path.stem}.txt"
        )
        inferred_groups = [False for i in range(num_groups)]
        if inferred_annotations_path.exists():
            with open(str(inferred_annotations_path), "r") as annotations_file:
                inferred_annotations_lines = annotations_file.readlines()

            for inference_line in inferred_annotations_lines:
                line_split = inference_line.split(" ")
                class_id = int(line_split[0])
                conf = float(line_split[5])
                if conf < thresholds[class_id]:
                    continue
                groups = get_memberships(class_id, yolo_group_filters)

                group_name = groups  #:
                group_int = group_to_int_mapping.get(group_name, None)
                if group_int is not None:
                    inferred_groups[group_int] = True
                else:
                    if class_id == 0:
                        count_positive += 1

        else:
            pass  # print(f"File not found: {str(inferred_annotations_path)}")

        results_dict[image_path.name] = inferred_groups
    return results_dict
