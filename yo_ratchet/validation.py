from pathlib import Path
from typing import Optional, Dict

from yo_ratchet.modelling import run_detections
from yo_ratchet.yo_valuate.as_classification import optimise_binary_and_get_group_classification_performance
from yo_ratchet.yo_wrangle.common import get_config_items, get_id_to_label_map


def run_detections_without_filtering_then_validate(
    base_dir: Path,
    model_version: str,
    data_folder_name: str,
    detect_images_root: Path,
    ground_truth_path: Path,  # e.g. detect_images_root / "YOLO_darknet"
    model_path: Optional[Path] = None,  # full path to weights/best.pt
    groupings: Optional[Dict] = None
):
    """
    Prints classification performance for each class on a per-image basis
    (not per bounding box).

    """

    assert detect_images_root.exists()
    assert ground_truth_path.exists()

    _, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    yolo_root = Path(yolo_root)
    if model_path is None:
        model_path = (yolo_root / "runs/train" / model_version / "weights/best.pt")
    assert model_path.exists()

    detect_folder_name = run_detections(
        images_path=detect_images_root,
        dataset_version=data_folder_name,
        model_path=model_path,
        model_version=model_version,
        base_dir=base_dir,
        conf_thres=0.05,
        device=1,
        img_size=1000
    )
    inferences_path = (
            yolo_root / "runs/detect" / detect_folder_name / "labels"
    )

    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    optimise_binary_and_get_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inferences_path,
        classes_map=classes_map,
        groupings=groupings,
        base_dir=base_dir,
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

