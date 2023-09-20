import json
import shutil

import fiftyone as fo
from pathlib import Path

import numpy

from yo_ratchet.workflow import (
    run_prepare_dataset_and_train,
    set_globals,
    run_find_errors,
)
import wrangling_example as dataset_workbook
from yo_ratchet.yo_valuate.as_classification import (
    optimise_binary_and_get_group_classification_performance,
    classification_metrics_for_cross_validation_set,
)
from yo_ratchet.yo_valuate.reference_csv import (
    get_classification_performance,
    get_actual_vs_inferred_df,
    get_severity_dict,
)
from yo_ratchet.yo_wrangle.common import get_id_to_label_map, get_config_items
from yo_ratchet.yo_wrangle.mine import prepare_training_data_subset_from_reviewed_yolo_file


def test_prepare_dataset_and_train():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_prepare_dataset_and_train()


def test_find_errors():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filters=["WS"],
        limit=64,
    )


def test_export():
    dataset = fo.load_dataset("v9b")
    rel_path = Path(__file__).parent
    dataset.export(
        export_dir="./.export",
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
        rel_dir=str(rel_path),
    )


def test_import():
    # from fiftyone.utils.data.exporters import FiftyOneDatasetExporter # FiftyOneDatasetImporter as Importer
    name = "dsXXX"
    import fiftyone as fo

    if name in fo.list_datasets():
        fo.delete_dataset(name=name)
    else:
        pass
    dataset = fo.Dataset.from_dir(
        dataset_dir="./.export",
        dataset_type=fo.types.FiftyOneDataset,
        name=name,
    )
    # fo.launch_app(dataset)
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filters=["WS"],
        limit=64,
        dataset_label=name,
    )


def test_errors():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="group",
        label_filters=["Str", "Stp1", "Stp2", "D40", "PP"],
        limit=64,
        dataset_label="srd20.1",
    )


def test_performance_metrics_for_charters_towers():
    truths_csv = Path(
        "/home/david/RACAS/sealed_roads_dataset/.shapefile_data/CTRC_all_sealed.csv"
    )
    run_name = "Charters_Towers__srd19.1_conf10pcnt"
    inferences_path = Path(
        f"/home/david/addn_repos/yolov5/runs/detect/{run_name}/labels"
    )
    images_root = Path("/home/david/RACAS/640_x_640/RACAS_CTRC_2021_sealed")
    csv_group_filters = {
        "Cracking": ["Cracking"],
        "Pothole": ["POTHOLE", "pothole", "Potholes"],
        "Edge Break": ["Edge"],
        "Stripping": ["Stripping", "Strip", "Failure"],
    }
    yolo_group_filters = {
        "Cracking": [0, 1, 2, 16],
        "Pothole": [3],
        "Edge Break": [4],
        "Stripping": [17, 12],  # Scf = 12; Stp = 17.
    }
    with open("classes.json", "r") as f_in:
        classes_info = json.load(f_in)
    print()
    get_classification_performance(
        images_root=images_root,
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=inferences_path,
        csv_group_filters=csv_group_filters,
        yolo_group_filters=yolo_group_filters,
        classes_info=classes_info,
        image_key="Photo_Name",
        classifications_key="Final_Remedy",
        print_table=True,
    )


def test_arrange_images_per_classification_errors():
    every_n_th = 10
    dst_folder_fp = Path("/home/david/RACAS/CT_Errors_19_1/fp_sev_8")
    dst_folder_fn = Path("/home/david/RACAS/CT_Errors_19_1/fn_sev_8")
    images_root = Path(
        "/home/david/RACAS/640_x_640/RACAS_CTRC_2021_sealed"
    )  # "/home/david/addn_repos/yolov5/runs/detect/Charters_Towers_errors__srd18.3_conf5pcnt")  #
    truths_csv = Path(
        "/home/david/RACAS/sealed_roads_dataset/.shapefile_data/CTRC_all_sealed.csv"
    )
    run_name = "Charters_Towers__srd19.1_conf10pcnt"
    inferences_path = Path(
        f"/home/david/addn_repos/yolov5/runs/detect/{run_name}/labels"
    )
    boxed_images = Path(f"/home/david/addn_repos/yolov5/runs/detect/{run_name}")

    csv_group_filters = {
        "Cracking": ["Cracking"],
        "Pothole": ["POTHOLE", "pothole", "Potholes"],
        "Edge Break": ["Edge"],
        "Stripping": ["Stripping", "Strip", "Failure"],
    }
    yolo_group_filters = {
        "Cracking": [0, 1, 2, 16, 6],  # #6=Rutting which often has cracks
        "Pothole": [3],
        "Edge Break": [4],
        "Stripping": [17, 12],  # Scf = 12; Stp = 17.
    }
    with open("classes.json", "r") as f_in:
        classes_info = json.load(f_in)
    print()
    keys = list(csv_group_filters.keys())
    for key in keys:
        (dst_folder_fp / key).mkdir(parents=True)
        for sev in ["8", "9", "10"]:
            (dst_folder_fn / sev / key).mkdir(parents=True)
    df = get_actual_vs_inferred_df(
        images_root=images_root,
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=inferences_path,
        csv_group_filters=csv_group_filters,
        yolo_group_filters=yolo_group_filters,
        classes_info=classes_info,
        image_key="Photo_Name",
        classifications_key="Final_Remedy",
    )
    true_positive = true_negative = 0
    severity_dict = get_severity_dict(
        truths_csv=truths_csv,
        field_for_severity="severity",
        field_for_key="Photo_Name"
    )
    count = 0
    for index, row in df.iterrows():
        count += 1
        if count % every_n_th != 0:
            continue
        actual = row["actual_classifications"]
        inferred = row["inferred_classifications"]
        comparison = numpy.logical_xor(actual, inferred)
        image_name = str(index)
        false_negative = numpy.logical_and(comparison, actual).nonzero()[0]
        if len(inferred.nonzero()[0]) == 0 and len(comparison.nonzero()[0]) == 0:
            true_negative += 1
        elif len(inferred.nonzero()[0]) > 0 and len(comparison.nonzero()[0]) == 0:
            true_positive += 1
        if len(false_negative) > 0:
            for el in false_negative:
                class_name = keys[el]
                severity = severity_dict[image_name]
                shutil.copy(
                    src=str(boxed_images / image_name),
                    dst=str(dst_folder_fn / str(severity) / class_name / image_name),
                )
        false_positive = numpy.logical_and(comparison, inferred).nonzero()[0]
        if (
            len(false_positive) > 0
        ):  # There is a disagreement either False Positive or False Negative
            for el in false_positive:
                class_name = keys[el]
                shutil.copy(
                    src=str(boxed_images / image_name),
                    dst=str(dst_folder_fp / class_name / image_name),
                )
    print()
    print(f"True Positive = ", true_positive)
    print(f"True Negative = ", true_negative)
    print("Count = ", count)


def test_group_performance():
    _, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=Path(__file__).parent
    )
    yolo_root = Path(yolo_root)
    detect_images_root = yolo_root / "datasets" / "srd20.1.1" / "val" / "images"
    ground_truth_path = yolo_root / "datasets" / "srd20.1.1" / "val" / "labels"
    inferences_path = (
        yolo_root / "runs/detect/srd20.1.1_val__srd20.1.1_conf5pcnt/labels"
    )
    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())

    optimise_binary_and_get_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inferences_path,
        classes_map=classes_map,
        groupings=dataset_workbook.GROUPINGS,
        base_dir=None,
    )


def test_optimise_conf():
    cross_validation_prefix = "srd20.1"
    classification_metrics_for_cross_validation_set(
        dataset_prefix=cross_validation_prefix,
        base_dir=Path(__file__).parent,
        groupings=dataset_workbook.GROUPINGS,
    )
