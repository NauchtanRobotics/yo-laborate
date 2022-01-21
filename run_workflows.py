import json

import fiftyone as fo
from pathlib import Path

from yo_ratchet.workflow import (
    run_prepare_dataset_and_train,
    set_globals,
    run_find_errors,
)
import wrangling_example as dataset_workbook
from yo_valuate.reference_csv import get_classification_performance


def test_prepare_dataset_and_train():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_prepare_dataset_and_train()


def test_find_errors():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filter="WS",
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
        label_filter="WS",
        limit=64,
        dataset_label=name,
    )


def test_errors():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filter="WS",
        limit=64,
    )


def test_performance_metrics_for_charters_towers():
    truths_csv = Path("/home/david/RACAS/sealed_roads_dataset/CTRC_all_sealed.csv")
    inferences_path = Path(
            "/home/david/addn_repos/yolov5/runs/detect/Charters_Towers__srd16.3_conf7pcnt/labels"
        )
    csv_group_filters = {
        "Cracking": ["Cracking"],
        "Risk Defects": ["POTHOLE", "pothole", "Edge", "Potholes"],
        "Stripping": ["Stripping", "Strip", "Failure"]
    }
    yolo_group_filters = {
        "Cracking": [0, 1, 2, 16],
        "Risk Defects": [3, 4],
        "Stripping": [17, 12],  # Scf = 12; Stp = 17.
    }
    with open("classes.json", "r") as f_in:
        classes_info = json.load(f_in)
    print()
    get_classification_performance(
        images_root=Path("/home/david/RACAS/640_x_640/RACAS_CTRC_2021_sealed"),
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=inferences_path,
        csv_group_filters=csv_group_filters,
        yolo_group_filters=yolo_group_filters,
        classes_info=classes_info,
        image_key="Photo_Name",
        classifications_key="Final_Remedy",
        print_table=True,
    )
