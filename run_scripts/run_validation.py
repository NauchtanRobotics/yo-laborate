from pathlib import Path

from yo_ratchet.validation import (
    run_ensemble_detections_without_filtering_then_validate,
    run_detections_without_filtering_then_validate
)


def test_run_ensemble_detections_without_filtering_then_validate_srd36():
    base_dir = Path("/home/david/production/sealed_roads_dataset")
    model_version = "srd36.1"
    data_folder_name = "Murrumbidgee_2023"
    detect_images_root = (Path("/home/david/production") / data_folder_name)
    ground_truth_path = detect_images_root / "YOLO_darknet"
    groupings = GROUPINGS = {
        "Risk Defects": [3, 4, 17],
        "Potholes Big/Small": [3, 18],
        "Cracking": [0, 1, 2, 11, 14, 16],
        "Stripping": [12, 17, 18, 19, 20, 33]
    }
    run_ensemble_detections_without_filtering_then_validate(
        base_dir=base_dir,
        model_version=model_version,
        data_folder_name=data_folder_name,
        detect_images_root=detect_images_root,
        ground_truth_path=ground_truth_path,
        groupings=groupings
    )


def test_run_detections_without_filtering_then_validate_srd39():
    base_dir = Path("/home/david/production/sealed_roads_dataset")
    model_version = "srd39.3"
    data_folder_name = "Murrumbidgee_2023"
    detect_images_root = (Path("/home/david/production") / data_folder_name)
    ground_truth_path = detect_images_root / "YOLO_darknet"
    groupings = GROUPINGS = {
        "Risk Defects": [3, 4, 17],
        "Potholes Big/Small": [3, 18],
        "Cracking": [0, 1, 2, 11, 14, 16],
        "Stripping": [12, 17, 18, 19, 20, 33]
    }
    run_detections_without_filtering_then_validate(
        base_dir=base_dir,
        model_version=model_version,
        data_folder_name=data_folder_name,
        detect_images_root=detect_images_root,
        ground_truth_path=ground_truth_path,
        groupings=groupings
    )
