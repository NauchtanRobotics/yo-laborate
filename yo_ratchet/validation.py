from pathlib import Path
from typing import Optional, Dict, List

from yo_ratchet.modelling import run_detections, run_detections_using_cv_ensemble_given_paths, DETECT_IMAGE_SIZE
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


def run_ensemble_detections_without_filtering_then_validate(
    base_dir: Path,
    model_version: str,
    data_folder_name: str,
    detect_images_root: Path,
    ground_truth_path: Path,  # e.g. detect_images_root / "YOLO_darknet"
    explicit_model_paths: Optional[List[Path]] = None,
    groupings: Optional[Dict] = None,
    img_size: int = DETECT_IMAGE_SIZE  # resizes to this before running detections
):
    """
    Prints classification performance for each class on a per-image basis
    (not per bounding box).

    """
    assert detect_images_root.exists()
    assert ground_truth_path.exists()

    python_path, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    yolo_root = Path(yolo_root)
    conf = 0.05
    inferences_root_path = run_detections_using_cv_ensemble_given_paths(
        images_path=detect_images_root,
        model_version=model_version,
        detection_dataset_name=data_folder_name,
        k_folds=3,
        python_path=Path(python_path),
        yolo_root=Path(yolo_root),
        conf_thres=conf,
        device=0,
        img_size=img_size,
        explicit_model_paths=explicit_model_paths,
    )
    inference_labels_root = inferences_root_path / "labels"
    assert inference_labels_root.exists()

    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    optimise_binary_and_get_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inference_labels_root,
        classes_map=classes_map,
        groupings=groupings,
        base_dir=base_dir,
    )
