import json

import numpy
import pandas
from tabulate import tabulate
from typing import Optional, List, Dict, Tuple
from sklearn import metrics as skm
from pathlib import Path

from yo_ratchet.yo_wrangle.common import (
    get_all_jpg_recursive,
    get_id_to_label_map,
    get_config_items,
    PERFORMANCE_FOLDER,
    save_output_to_text_file,
    get_label_to_id_map,
)

RECALL = "R"
PRECISION = "P"
F1 = "F1"
CONF_MIN_STR = "conf_min"

CONF_TEST_LEVELS = [
    0.35,
    0.35,
    0.35,
    0.45,
    0.45,
    0.50,
    0.1,
    0.45,
    0.45,
    0.25,
    0.55,
    0.35,
    0.25,
    0.15,
    0.16,
    0.32,
    0.35,
    0.15,
    0.1,
    0.4,
]

F1_PERFORMANCE_JSON = "f1_performance.json"


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
                if len(actual_classifications) > int(class_id):
                    actual_classifications[int(class_id)] = True
                else:
                    print("\nTrying to exceed length of actual_classifications list. class_id=" + class_id)
                    print("in file: " + str(ground_truth_path))
        else:
            pass  # ground_truth_classification already initialized to False

        inferred_annotations_path = (
            root_inferred_bounding_boxes / f"{image_path.stem}.txt"
        )
        inferred_classifications = [False for i in range(num_classes)]
        confidence = [0.0 for i in range(num_classes)]
        if inferred_annotations_path.exists():
            with open(str(inferred_annotations_path), "r") as annotations_file:
                inferred_annotations_lines = annotations_file.readlines()

            for inference_line in inferred_annotations_lines:
                line_split = inference_line.split(" ")
                class_id = line_split[0]
                inferred_classifications[int(class_id)] = True
                conf = line_split[5]
                confidence[int(class_id)] = float(conf)
        else:
            pass  # inference already initialized to False

        results_dict[image_path] = {
            "actual_classifications": numpy.array(actual_classifications),
            "inferred_classifications": inferred_classifications,
            "confidence": confidence,
        }
    df = pandas.DataFrame(results_dict)
    df = df.transpose()
    return df


def _get_classification_metrics_for_group(
    df: pandas.DataFrame,
    idxs: List[int],
    optimised_thresholds: Dict[int, float],
    to_console: bool = False,
) -> Tuple[float, float, float, float]:
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
    y_confidences = df["confidence"]

    count = y_truths.size
    assert count == y_inferences.size

    group_truths = numpy.array([False for _ in range(count)])
    group_inferences = numpy.array([False for _ in range(count)])
    for i, idx in enumerate(idxs):
        confidences = numpy.array([y[idx] for y in y_confidences])
        threshold = optimised_thresholds.get(int(idx), 0.1)
        cond = confidences >= threshold
        inferences = numpy.array([y[idx] for y in y_inferences])
        truths = numpy.array([y[idx] for y in y_truths])
        re_inferences = inferences & cond
        group_truths = numpy.logical_or(group_truths, truths)
        group_inferences = numpy.logical_or(group_inferences, re_inferences)

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


def get_groups_classification_metrics(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    classes_map: Dict[int, str],
    groupings: Dict[str, List[int]],
    optimised_thresholds: Dict[int, float],
    dst_csv: Optional[Path] = None,
    print_table: bool = True,
) -> pandas.DataFrame:
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
        p, r, f1, _ = _get_classification_metrics_for_group(
            df=df, idxs=group_members, optimised_thresholds=optimised_thresholds
        )
        results[group_name] = {
            "P": "{:.2f}".format(p),
            "R": "{:.2f}".format(r),
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


def optimise_binary_and_get_group_classification_performance(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    classes_map: Dict[int, str],
    print_first_n: Optional[int] = None,
    groupings: Dict[str, List[int]] = None,
    base_dir: Path = None,
    dataset_label: str = None,
):
    df = optimise_analyse_model_binary_metrics(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        classes_map=classes_map,
        print_first_n=print_first_n,
        dst_csv=None,
    )
    table_str = tabulate(
        df.transpose(),
        headers="keys",
        showindex="always",
        tablefmt="pretty",
    )
    table_str += "\n"
    thresholds = {
        class_name: float(metrics_dict["@conf"])
        for class_name, metrics_dict in df.to_dict().items()
    }
    label_to_id_dict = get_label_to_id_map(base_dir=base_dir)
    conf_dict = {
        label_to_id_dict.get(class_name, 34): conf_threshold for class_name, conf_threshold in thresholds.items()
    }
    if groupings:
        df = get_groups_classification_metrics(
            images_root=images_root,
            root_ground_truths=root_ground_truths,
            root_inferred_bounding_boxes=root_inferred_bounding_boxes,
            classes_map=classes_map,
            groupings=groupings,
            optimised_thresholds=conf_dict,
            dst_csv=None,
        )
        table_str += "\n"
        table_str += tabulate(
            df.transpose(),
            headers="keys",
            showindex="always",
            tablefmt="pretty",
        )
    if base_dir:
        if dataset_label is None:
            dataset_label = root_ground_truths.parents[1].name
        file_name = f"{dataset_label}_performance_for_optimum_conf.txt"
        save_output_to_text_file(
            content=table_str, base_dir=base_dir, file_name=file_name, commit=False
        )
    else:
        print(table_str)
    return table_str


def classification_metrics_for_cross_validation_set(
    base_dir: Path,
    dataset_prefix: str,  # E.g. 14.4  - do not include patch
    print_table: bool = False,
    groupings: Dict[str, List[int]] = None,
):
    """
    Loops through cross_validation training and inferences data in the configured
    yolo_root folder, and find the conf threshold which leads to the optimised
    classification F1_score.

    Does this individual for each cross validation part, and then calculates the
    average results for the set of cross validation parts.

    Saves the individual and averaged classification results for the cross validation
    set of training and detection runs as pretty tables in text files.

    Also serialises the average F1 and minimum optimum conf values in an accumulating
    json file for use in plotting performance ratcheting over time (e.g in GUI
    interface).

    """
    from yo_ratchet.workflow import K_FOLDS, CONF_PCNT  # To prevent circular references
    if n_folds is None:
        n_folds = K_FOLDS

    _, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    f1_scores = []
    confidences = []
    for i in range(n_folds):
        dataset_label = f"{dataset_prefix}.{str(i + 1)}"
        (
            inferences_path,
            detect_images_root,
            ground_truth_path,
        ) = get_paths_for_cross_validation_part(
            base_dir=base_dir, dataset_label=dataset_label, conf_pcnt=CONF_PCNT
        )
        if print_table:
            print(f"\nDataset: {dataset_label}")
        df = optimise_analyse_model_binary_metrics(
            images_root=detect_images_root,
            root_ground_truths=ground_truth_path,
            root_inferred_bounding_boxes=inferences_path,
            classes_map=classes_map,
            dst_csv=None,
            print_table=print_table,
        )

        f1_scores.append(df.loc[[F1]])
        confidences.append(df.loc[["@conf"]])
        optimise_binary_and_get_group_classification_performance(
            images_root=detect_images_root,
            root_ground_truths=ground_truth_path,
            root_inferred_bounding_boxes=inferences_path,
            classes_map=classes_map,
            groupings=groupings,
            base_dir=base_dir,
        )

    df = pandas.concat(f1_scores, axis=0, ignore_index=True).astype(float)
    df_conf = pandas.concat(confidences, axis=0, ignore_index=True).astype(float)

    new_df = pandas.DataFrame()
    new_df[F1] = f1_mean = df.mean(axis=0)
    update_performance_json(
        base_dir, version=dataset_prefix, label=F1, performance=f1_mean
    )
    new_df["min"] = df.min(axis=0)
    new_df["max"] = df.max(axis=0)
    new_df["conf_min"] = conf_min = df_conf.min(axis=0)
    update_performance_json(
        base_dir, version=dataset_prefix, label=CONF_MIN_STR, performance=conf_min
    )
    conf_thresholds = {
        i: val for i, val in enumerate(df_conf.mean(axis=0).to_dict().values())
    }
    new_df["conf_max"] = df_conf.max(axis=0)
    new_df = new_df.applymap(lambda x: round(x, 3))
    tbl_str = tabulate(
        new_df,
        headers="keys",
        showindex="always",
        tablefmt="pretty",
    )
    if groupings:
        df = get_average_group_metrics_for_cv_set(
            dataset_prefix=dataset_prefix,
            yolo_root=yolo_root,
            k_folds=K_FOLDS,
            conf_pcnt=CONF_PCNT,
            classes_map=classes_map,
            groupings=groupings,
            optimised_thresholds=conf_thresholds,
        )
        tbl_str += "\n"
        tbl_str += tabulate(
            df,
            headers="keys",
            showindex="always",
            tablefmt="pretty",
            floatfmt=".2f",
        )
    output_file = f"{dataset_prefix}_classification_f1_summary.txt"
    save_output_to_text_file(
        content=tbl_str, base_dir=base_dir, file_name=output_file, commit=True
    )


def get_average_group_metrics_for_cv_set(
    dataset_prefix: str,
    yolo_root: str,
    k_folds: int,
    conf_pcnt: int,
    classes_map: Dict[int, str],
    groupings: Dict[str, List[int]],
    optimised_thresholds: Dict[int, float],
) -> pandas.DataFrame:
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(k_folds):
        dataset_label = f"{dataset_prefix}.{str(i + 1)}"
        (
            inferences_path,
            images_path,
            ground_truths_path,
        ) = get_paths_for_cross_validation_part(
            yolo_root=yolo_root, dataset_label=dataset_label, conf_pcnt=conf_pcnt
        )
        df = get_groups_classification_metrics(
            images_root=images_path,
            root_ground_truths=ground_truths_path,
            root_inferred_bounding_boxes=inferences_path,
            classes_map=classes_map,
            groupings=groupings,
            optimised_thresholds=optimised_thresholds,
            dst_csv=None,
        )
        f1_scores.append(df.loc[[F1]])
        precision_scores.append(df.loc[[PRECISION]])
        recall_scores.append(df.loc[[RECALL]])
    precision_series = (
        pandas.concat(precision_scores, axis=0, ignore_index=True)
        .astype(float)
        .mean(axis=0)
    )
    precision_series.name = PRECISION
    recall_series = (
        pandas.concat(recall_scores, axis=0, ignore_index=True)
        .astype(float)
        .mean(axis=0)
    )
    recall_series.name = RECALL
    f1_scores_series = (
        pandas.concat(f1_scores, axis=0, ignore_index=True).astype(float).mean(axis=0)
    )
    f1_scores_series.name = F1
    results_df = pandas.concat(
        [precision_series, recall_series, f1_scores_series], axis=1
    ).round(decimals=2)
    return results_df


def get_paths_for_cross_validation_part(
    yolo_root: str, dataset_label: str, conf_pcnt: int
) -> Tuple[Path, Path, Path]:
    """
    Using conventions, provides Pathlib paths to the root of the inference annotations files,
    the root of the image paths and the root of the ground truth annotation files.
    """
    inferences_path = Path(
        f"{yolo_root}/runs/detect/{dataset_label}_val__{dataset_label}_conf{conf_pcnt}pcnt/labels"
    ).resolve()
    detect_images_root = Path(f"{yolo_root}/datasets/{dataset_label}/val").resolve()
    ground_truth_path = Path(
        f"{yolo_root}/datasets/{dataset_label}/val/labels"
    ).resolve()
    return inferences_path, detect_images_root, ground_truth_path


def update_performance_json(
    base_dir: Path, version: str, label: str, performance: pandas.Series
):
    (base_dir / PERFORMANCE_FOLDER).mkdir(exist_ok=True)
    output_path = base_dir / PERFORMANCE_FOLDER / F1_PERFORMANCE_JSON
    if output_path.exists():
        with open(str(output_path), "r") as file_obj:
            serialising_dict = json.load(file_obj)
    else:
        serialising_dict = {}
    data_to_add_under_version = performance.to_dict()
    existing_data_under_version = serialising_dict.get(version, None)
    if existing_data_under_version:
        version_dict = {**existing_data_under_version, label: data_to_add_under_version}
    else:
        version_dict = {label: data_to_add_under_version}
    serialising_dict.update({version: version_dict})
    with open(str(output_path), "w") as file_obj:
        json.dump(serialising_dict, fp=file_obj, indent=4)


def optimise_analyse_model_binary_metrics(
    images_root: Path,
    root_ground_truths: Path,
    root_inferred_bounding_boxes: Path,
    classes_map: Dict[int, str],
    print_first_n: Optional[int] = None,
    dst_csv: Optional[Path] = None,
    confidence_threshold: Optional[float] = None,
    print_table: bool = True,
) -> pandas.DataFrame:
    """
    Prints (and optionally saves) results for CLASSIFICATION performance from
    object detection ground truths and predictions.

    This approach is appropriate when you don't care for object detection
    and just want classification performance per image, not per bounding box.

    """
    num_classes = len(classes_map)
    df = get_truth_vs_inferred_dict_by_photo(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferred_bounding_boxes,
        num_classes=num_classes,
    )
    if dst_csv:
        df.to_csv(dst_csv, index=False)

    print_first_n = num_classes if print_first_n is None else print_first_n
    results = _optimise_analyse_model_binary_metrics(
        df=df,
        classes_map=classes_map,
        print_first_n=print_first_n,
        confidence_threshold=confidence_threshold,
    )
    results = pandas.DataFrame(results)
    if print_table:
        print(
            tabulate(
                results.transpose(),
                headers="keys",
                showindex="always",
                tablefmt="pretty",
            )
        )
    return results


def _optimise_analyse_model_binary_metrics(
    df: pandas.DataFrame,
    classes_map: Dict[int, str],
    print_first_n: int,
    confidence_threshold: float = None,
) -> Dict[str, Dict[str, str]]:
    """
    Finds the individual 'conf' level that produces the highest F1 score for
    each class_id and returns the F1, recall and precision scores ('F1', 'R',
    'P') corresponding to the 'optimum' conf level for the respective class.

    For example, an object detection model for Cats and Dogs might return:

        {
            "Cats": {"@conf": "0.18", "F1": "1.00", "P": "1.00", "R": "1.00"},
            "Dogs": {"@conf": "0.25", "F1": "0.50", "P": "0.50", "R": "0.50"},
        }

    """
    y_truths = df["actual_classifications"]
    y_inferences = df["inferred_classifications"]
    y_confidences = df["confidence"]
    count = y_truths.size
    assert count == y_inferences.size

    if confidence_threshold is None:
        confidence_levels = CONF_TEST_LEVELS
    else:
        confidence_levels = [confidence_threshold]

    results = {}

    for class_id in range(print_first_n):
        precision = 0
        recall = 0
        f1 = 0
        optimum_conf = 0
        truths = numpy.array([y[class_id] for y in y_truths])
        inferences = numpy.array([y[class_id] for y in y_inferences])
        confidences = numpy.array([y[class_id] for y in y_confidences])
        for conf in confidence_levels:
            cond = confidences >= conf
            re_inferences = inferences & cond
            labels = None
            p = skm.precision_score(
                y_true=truths,
                y_pred=re_inferences,
                labels=labels,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
            )
            r = skm.recall_score(
                y_true=truths,
                y_pred=re_inferences,
                labels=labels,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
            )
            f = skm.f1_score(
                y_true=truths,
                y_pred=re_inferences,
                labels=labels,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
            )
            if f > f1:
                recall = r
                precision = p
                f1 = f
                optimum_conf = conf

        class_name = classes_map.get(class_id, "Unknown")
        results[class_name] = {
            "P": "{:.2f}".format(precision),
            "R": "{:.2f}".format(recall),
            "F1": "{:.2f}".format(f1),
            "@conf": "{:.2f}".format(optimum_conf),
        }
    return results
