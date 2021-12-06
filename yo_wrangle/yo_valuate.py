import numpy
import pandas
from tabulate import tabulate
from typing import Optional, List, Dict
from sklearn import metrics as skm
from pathlib import Path

from yo_wrangle.common import get_all_jpg_recursive, get_id_to_label_map


def get_truth_vs_inferred_dict_by_photo(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    num_classes: int,
) -> pandas.DataFrame:
    """
    A function for converting object detection data to classification.

    Achieves this by finding a unique list (set) of class memberships an
    image can claim based on YOLO annotation file.

    Simply provide the root directories for the annotations corresponding
    to ground truths and inferences (detections).

    Returns a dataframe that contains a list of actual classifications, and
    a list of inferred classification for each image index.

    """
    results_dict = {}
    for image_path in get_all_jpg_recursive(img_root=images_root):
        ground_truth_path = root_ground_truths / f"{image_path.stem}.txt"
        actual_classifications = [False for i in range(num_classes)]
        if ground_truth_path.exists():
            with open(str(ground_truth_path), "r") as truth_file:
                ground_truth_lines = truth_file.readlines()
            for ground_truth_line in ground_truth_lines:
                class_id = ground_truth_line.split(" ")[0]
                actual_classifications[int(class_id)] = True
        else:
            pass  # ground_truth_classification already initialized to False

        inferred_annotations_path = (
            root_inferred_bounding_boxes / f"{image_path.stem}.txt"
        )
        inferred_classifications = [False for i in range(num_classes)]
        if inferred_annotations_path.exists():
            with open(str(inferred_annotations_path), "r") as annotations_file:
                inferred_annotations_lines = annotations_file.readlines()

            for inference_line in inferred_annotations_lines:
                class_id = inference_line.split(" ")[0]
                inferred_classifications[int(class_id)] = True
        else:
            pass  # inference already initialized to False

        results_dict[image_path] = {
            "actual_classifications": numpy.array(actual_classifications),
            "inferred_classifications": inferred_classifications,
        }
    df = pandas.DataFrame(results_dict)
    df = df.transpose()
    return df


def _get_classification_metrics_for_group(
    df: pandas.DataFrame,
    idxs: List[int],
    to_console: bool = False,
):
    """
    Given a dataframe that contains a list of actual classifications, and
    a list of inferred classification for each image, returns
    precision, recall, f1-score and accuracy CLASSIFICATION metrics.

    """
    if isinstance(idxs, int):
        idxs = [idxs]
    else:
        pass
    y_truths = df["actual_classifications"]
    y_inferences = df["inferred_classifications"]
    count = y_truths.size
    assert count == y_inferences.size

    group_truths = numpy.array([False for i in range(count)])
    group_inferences = numpy.array([False for i in range(count)])
    for i, idx in enumerate(idxs):
        group_truths = numpy.logical_or(
            group_truths, numpy.array([y[idx] for y in y_truths])
        )
        group_inferences = numpy.logical_or(
            group_inferences, numpy.array([y[idx] for y in y_inferences])
        )

    labels = None
    precision = skm.precision_score(
        y_true=group_truths,
        y_pred=group_inferences,
        labels=labels,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    )
    recall = skm.recall_score(
        y_true=group_truths,
        y_pred=group_inferences,
        labels=labels,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    )
    f1 = skm.f1_score(
        y_true=group_truths,
        y_pred=group_inferences,
        labels=labels,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    )
    accuracy = skm.accuracy_score(
        y_true=group_truths,
        y_pred=group_inferences,
        sample_weight=None,
    )
    if to_console:
        print("Precision: {:.1f}".format(precision * 100))
        print("Recall:    {:.1f}".format(recall * 100))
        print("F1-score:  {:.1f}".format(f1 * 100))
        print("Accuracy:  {:.1f}".format(accuracy * 100))
        print("\n")

    return precision, recall, f1, accuracy


def _get_binary_classification_metrics_for_idx(
    df: pandas.DataFrame,
    idx: int,
    to_console: bool = False,
):
    """
    A simple interface for binary metrics that passes through to the more
    generalised function to assess model metrics.

    """
    if isinstance(idx, int):
        idx = [idx]  # Convert to list
    elif isinstance(idx, list):
        pass  # This is okay too
    else:
        raise Exception("idx should be an int")
    return _get_classification_metrics_for_group(df=df, idxs=idx, to_console=to_console)


def analyse_model_binary_metrics(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    class_names_path: Path,
    print_first_n: Optional[int] = None,
    dst_csv: Optional[Path] = None,
):
    """
    Prints (and optionally saves) results for CLASSIFICATION performance from
    object detection ground truths and predictions.

    This approach is appropriate when you don't care for object detection
    and just want classification performance per image, not per bounding box.

    """
    classes_map = get_id_to_label_map(class_name_list_path=class_names_path)
    num_classes = len(classes_map)

    df = get_truth_vs_inferred_dict_by_photo(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        num_classes=num_classes,
    )

    if dst_csv:
        df.to_csv(dst_csv, index=False)

    results = {}
    print_first_n = num_classes if print_first_n is None else print_first_n
    for class_id in range(print_first_n):
        class_name = classes_map.get(class_id, "Unknown")
        precision, recall, f1, _ = _get_classification_metrics_for_group(
            df=df, idxs=[class_id]
        )
        results[class_name] = {
            "P": "{:.2f}".format(precision),
            "R": "{:.2f}".format(recall),
            "F1": "{:.2f}".format(f1),
        }

    print("\n")
    print(
        tabulate(
            pandas.DataFrame(results).transpose(),
            headers="keys",
            showindex="always",
            tablefmt="pretty",
        )
    )


def analyse_model_binary_metrics_for_groups(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    class_names_path: Path,
    groupings: Dict[
        str, List[int]
    ],  # E.g. {"Risk Defects": [3, 4], "Cracking": [0, 1, 2, 11, 16]}
    dst_csv: Optional[Path] = None,
):
    """
    Prints (and optionally saves) results for CLASSIFICATION performance
    (per image, not per bounding box) for groups of classes, according
    ti the groupings parameter::
        {
          <group_1_label>: [<one or more integer ids of classes that conform to group_1>],
          <group_2_label>: [<one or more integer ids of classes that conform to group_2>],
        }

    For an image if there is any bounding box ground truth from any of the class ids
    corresponding to a group, and there is any bounding box prediction for a class id
    for the said group, then this counts as a true positive.

    """
    classes_map = get_id_to_label_map(class_name_list_path=class_names_path)
    num_classes = len(classes_map)

    df = get_truth_vs_inferred_dict_by_photo(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        num_classes=num_classes,
    )
    if dst_csv:
        df.to_csv(dst_csv, index=False)

    results = {}
    for group_name, group_members in groupings.items():
        precision, recall, f1, _ = _get_classification_metrics_for_group(
            df=df, idxs=group_members
        )
        results[group_name] = {
            "P": "{:.2f}".format(precision),
            "R": "{:.2f}".format(recall),
            "F1": "{:.2f}".format(f1),
        }

    print("\n")
    print(
        tabulate(
            pandas.DataFrame(results).transpose(),
            headers="keys",
            showindex="always",
            tablefmt="pretty",
        )
    )


def binary_and_group_classification_performance(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    class_names_path: Path,
    print_first_n: Optional[int] = None,
    groupings: Dict[str, List[int]] = None,
):
    analyse_model_binary_metrics(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        class_names_path=class_names_path,
        print_first_n=print_first_n,
        dst_csv=None,
    )

    analyse_model_binary_metrics_for_groups(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        class_names_path=class_names_path,
        groupings=groupings,
        dst_csv=None,
    )
