import json
from pathlib import Path
from typing import Dict, List

import numpy
import pandas
from tabulate import tabulate

from yo_valuate.as_classification import _get_classification_metrics_for_group
from yo_valuate.reference_csv import get_thresholds
from yo_wrangle.common import get_all_jpg_recursive


def get_truth_vs_inferred_for_batch(
    images_root: Path,
    root_ground_truths: Path,
    batch_inferences_file: Path,
    num_classes: int = 27,
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
    inferences_df = pandas.read_csv(batch_inferences_file, sep=" ", header=None, usecols=[0, 1, 9])

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

        inferred_classifications = [False for i in range(num_classes)]
        confidence = [0.0 for i in range(num_classes)]
        inference_rows = inferences_df.loc[inferences_df[0] == image_path.name]
        if len(inference_rows) > 0:
            for index, inference_row in inference_rows.iterrows():
                class_id = inference_row[1]
                inferred_classifications[int(class_id)] = True
                conf = inference_row[9]
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


def measure_group_classification_performance_for_batch(
    images_root: Path,
    root_ground_truths: Path,
    batch_inferences_file: Path,
    classes_json: Path,
    groupings: Dict[str, List[int]] = None,
):
    """
    batch_inferences_file a single text file containing all inferences after
    various filtering measures have been applied.

    """
    with open(str(classes_json), "r") as f_in:
        classes_info = json.load(f_in)
    thresholds = get_thresholds(classes_info=classes_info)
    df_raw = get_truth_vs_inferred_for_batch(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        batch_inferences_file=batch_inferences_file
    )
    results = {}
    for group_name, group_members in groupings.items():
        p, r, f1, _ = _get_classification_metrics_for_group(
            df=df_raw, idxs=group_members, optimised_thresholds=thresholds
        )
        results[group_name] = {
            "P": "{:.2f}".format(p),
            "R": "{:.2f}".format(r),
            "F1": "{:.2f}".format(f1),
        }
    df = pandas.DataFrame(results)
    return df


def test_measure_group_classification_performance_for_batch():
    df = measure_group_classification_performance_for_batch(
        images_root=Path("/home/david/RACAS/640_x_640/Charters_Towers_subsamples_mix"),
        root_ground_truths=Path("/home/david/RACAS/640_x_640/Charters_Towers_subsamples_mix/YOLO_darknet"),
        batch_inferences_file=Path(
            "/home/david/defect_detection/defect_detection/evaluate/Charters_Towers_subsamples_mix.ai"),
        classes_json=Path("/home/david/RACAS/sealed_roads_dataset/classes.json"),
        groupings={
            "Risk Defects": [3, 4],
            "Potholes Big/Small": [3, 18],
            "Cracking": [0, 1, 2, 11, 16],
            "Stripping": [12, 17, 18, 19, 20, ]
        }
    )
    tbl_str = tabulate(
                df.transpose(),
                headers="keys",
                showindex="always",
                tablefmt="pretty",
    )
    print(tbl_str)


def test_get_truth_vs_inferred_for_batch():
    results_df = get_truth_vs_inferred_for_batch(
        images_root=Path("/home/david/RACAS/sealed_roads_dataset/CT_EB_D40_Cracking_hard_pos"), # 640_x_640/Scenic_Rim_2021_sealed"),
        root_ground_truths=Path("/home/david/RACAS/sealed_roads_dataset/CT_EB_D40_Cracking_hard_pos/YOLO_darknet"),
        batch_inferences_file=Path("/home/david/defect_detection/defect_detection/evaluate/ChartersTowers_combined_20conf.ai")
    )
    print(results_df)
