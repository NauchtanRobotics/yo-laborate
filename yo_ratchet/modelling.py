import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict

from yo_ratchet.dataset_versioning.tag import get_path_for_best_pretrained_model
from yo_ratchet.yo_wrangle.common import (
    get_config_items,
    save_output_to_text_file,
    get_yolo_detect_paths,
)
from yo_ratchet.yo_wrangle.stats import count_class_instances_in_datasets
from yo_ratchet.yo_wrangle.wrangle import collate_and_split

EPOCHS = 325
DETECT_IMAGE_SIZE = 800
TRAIN_IMAGE_SIZE = 640
IOU_THRES = 0.45


def prepare_dataset_and_train(
    classes_map: Dict[int, str],
    subsets_included: List[Path],
    dst_root: Path,
    every_n_th: int,
    keep_class_ids: Optional[List[int]],
    skip_class_ids: Optional[List[int]],
    base_dir: Path,
    recode_map: Optional[Dict[int, int]] = None,
    run_training: bool = True,
    cross_validation_index: int = 0,
    fine_tune_patience: int = 5,
    img_size: Optional[int] = TRAIN_IMAGE_SIZE,
    epochs: Optional[int] = EPOCHS,
    batch_size: Optional[int] = 62,
    cache: Optional[str] = "ram",
):
    if epochs is None:
        epochs = EPOCHS
    class_ids = list(classes_map.keys())
    output_str = count_class_instances_in_datasets(
        data_samples=subsets_included,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )
    collate_and_split(
        subsets_included=subsets_included,
        dst_root=dst_root,
        every_n_th=every_n_th,
        keep_class_ids=keep_class_ids,
        skip_class_ids=skip_class_ids,
        recode_map=recode_map,
        cross_validation_index=cross_validation_index,
    )
    """Add actual classes support after filtering"""
    final_subsets_included = [
        (dst_root / "train"),
        (dst_root / "val"),
    ]
    output_str += "\n"
    output_str += count_class_instances_in_datasets(
        data_samples=final_subsets_included,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )
    model_instance = dst_root.name
    file_name = f"{model_instance}_classes_support.txt"
    save_output_to_text_file(
        content=output_str,
        base_dir=base_dir,
        file_name=file_name,
        commit=False,
    )

    class_names = [classes_map[class_id] for class_id in class_ids]
    yaml_text = f"""train: {str(dst_root)}/train/images/
val: {str(dst_root)}/val/images/
nc: {len(class_ids)}
names: {class_names}"""

    """ Write dataset.yaml in DST folder."""
    dst_dataset_path = dst_root / "dataset.yaml"
    with open(f"{str(dst_dataset_path)}", "w") as f_out:
        f_out.write(yaml_text)

    (
        python_path,
        yolo_base_dir,
        cfg_path,
        _,
        hyp_path,
        _,
        _,
    ) = get_config_items(base_dir)
    weights_path, fine_tune = get_path_for_best_pretrained_model(base_dir=base_dir)
    if not fine_tune:
        patience = 50
    else:
        patience = fine_tune_patience  # Just use the default param value
    train_script = str(Path(yolo_base_dir) / "train.py")
    pytorch_cmd = [
        python_path,
        train_script,
        f"--img={img_size}",
        f"--batch={batch_size}",
        "--workers=4",
        "--device=0,1",
        f"--cfg={cfg_path}",
        f"--epochs={epochs}",
        f"--data={str(dst_dataset_path)}",
        f"--weights={weights_path}",
        f"--hyp={hyp_path}",
        f"--name={model_instance}",
        f"--patience={str(patience)}",
        f"--cache={cache}",
        "--freeze=3",
    ]
    if fine_tune:
        start_epoch = epochs - fine_tune_patience
        pytorch_cmd.append(f"--start-epoch={start_epoch}")
    else:
        pass

    train_cmd_str = " ".join(pytorch_cmd)
    file_name = f"{model_instance}_train_cmd.txt"
    save_output_to_text_file(
        content=train_cmd_str,
        base_dir=base_dir,
        file_name=file_name,
        commit=False,
    )

    if run_training:
        subprocess.check_call(
            pytorch_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=str(Path(yolo_base_dir).parent),
        )


def run_detections(
    images_path: Path,
    dataset_version: str,
    model_path: Path,
    model_version: str,
    base_dir: Path,
    conf_thres: float = 0.1,
    device: int = 1,
    img_size: Optional[int] = DETECT_IMAGE_SIZE,
):
    results_name = f"{dataset_version}__{model_version}_conf{int(conf_thres * 100)}pcnt"
    python_path, yolo_path = get_yolo_detect_paths(base_dir)
    detect_script = yolo_path / "detect.py"
    pytorch_cmd = [
        python_path,
        f"{str(detect_script)}",
        f"--source={str(images_path)}",
        f"--weights={model_path}",
        f"--img={img_size}",
        f"--device={device}",
        f"--name={results_name}",
        "--save-txt",
        "--save-conf",
        "--nosave",
        # "--agnostic-nms",
        f"--iou-thres={IOU_THRES}",
        f"--conf-thres={conf_thres}",
        "--half",
        "--augment",
    ]
    print(
        subprocess.check_output(
            pytorch_cmd,
            stderr=subprocess.STDOUT,
            cwd=str(yolo_path),
        )
    )
    return results_name


def run_detections_using_cv_ensemble(
    images_path: Path,
    detection_dataset_name: str,
    model_version: Optional[str],
    k_folds: int,
    base_dir: Path,
    conf_thres: float = 0.1,
    device: int = 0,
    img_size: Optional[int] = DETECT_IMAGE_SIZE,
    explicit_model_paths: Optional[List[Path]] = None,
) -> str:
    """
    explicit_model_paths is optional. If not provided, the models will be assumed to be
    located in:
    `<yolo_root> / "runs" / "train" / <model_version>.# / "weights" / "best.pt"`

    where the # is replaced by numbers 1 to 6.

    model_version must also be provided regardless of whether explicit_model_paths
    is provided - model_version is used to give a meaningful name to the detections'
    folder.

    """
    python_path, yolo_root, _, _, _, _, _ = get_config_items(base_dir)
    detect_script = Path(yolo_root) / "detect.py"

    if explicit_model_paths is not None:
        pass  # model_version must also be provided - used to label results folder.
    elif model_version is not None:
        models_root = Path(yolo_root) / "runs" / "train"
        model_paths = [
            models_root / f"{model_version}.{str(i + 1)}" / "weights" / "best.pt"
            for i in range(k_folds)
        ]
        explicit_model_paths = [str(model_path) for model_path in model_paths if model_path.exists()]
    elif explicit_model_paths is None and model_version is None:
        raise RuntimeError("You must provide one of these params: model_full_path or model_version.")

    results_name = (
        f"{detection_dataset_name}__{model_version}_conf{int(conf_thres * 100)}pcnt"
    )

    pytorch_cmd = [
        python_path,
        f"{str(detect_script)}",
        f"--source={str(images_path)}",
        f"--img={img_size}",
        f"--device={device}",
        f"--name={results_name}",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--agnostic-nms",
        f"--iou-thres={IOU_THRES}",
        f"--conf-thres={conf_thres}",
        "--augment",
        f"--weights",
    ]
    pytorch_cmd.extend(explicit_model_paths)
    print(
        subprocess.check_output(
            pytorch_cmd,
            stderr=subprocess.STDOUT,
            cwd=yolo_root,
        )
    )
    return results_name


def run_detections_using_cv_ensemble_given_paths(
    images_path: Path,
    detection_dataset_name: str,
    model_version: str,  # e.g. srd26.0
    k_folds: int,  # How many folder were used when cv modeling for <model_version>?
    python_path: Path,
    yolo_root: Path,
    conf_thres: float = 0.1,
    device: int = 0,
    img_size: Optional[int] = DETECT_IMAGE_SIZE,
) -> Path:
    """
    Returns pathlib.Path to inferences directory for this run (this folder will contain
    a directory called "labels".

    This version of the function works with the Google Cloud file watcher
    in defect_detection repo which is configured for calling
    this function from any python_path and yolo_root, which don't change anyway...

    Ahh, this function is almost completely redundant except is returns a full path
    to the 'detect' run inferences folder instead of just the folder name. This is useful
    as it alleviates the need to reconstruct the path elsewhere.

    """
    detections_folder_name = (
        f"{detection_dataset_name}__{model_version}_conf{int(conf_thres * 100)}pcnt"
    )
    detect_script = yolo_root / "detect.py"
    model_paths = get_paths_to_weights(
        yolo_root=yolo_root, k_folds=k_folds, model_version=model_version
    )
    pytorch_cmd = [
        python_path,
        f"{str(detect_script)}",
        f"--source={str(images_path)}",
        f"--img={img_size}",
        f"--device={device}",
        f"--name={detections_folder_name}",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--agnostic-nms",
        f"--iou-thres={IOU_THRES}",
        f"--conf-thres={conf_thres}",
        "--augment",
        f"--weights",
    ]
    pytorch_cmd.extend(model_paths)
    print(
        subprocess.check_output(
            pytorch_cmd,
            stderr=subprocess.STDOUT,
            cwd=str(yolo_root),
        )
    )
    inferences_path = yolo_root / "runs" / "detect" / detections_folder_name
    return inferences_path


def get_paths_to_weights(
    yolo_root: Path, k_folds: int, model_version: str
) -> List[str]:
    """
    Intelligently infers whether this is truely a K-Folds CV model, or just  a
    regular model.

    Returns model paths as a list of strings, whether that list have a length
    k-fold or 1 depends on whether k-folds models could be found.

    Raises a RuntimeError is neither a single model or k-folds models could be
    found.

    """
    models_root = yolo_root / "runs" / "train"
    model_paths = [
        models_root / f"{model_version}.{str(i + 1)}" / "weights" / "best.pt"
        for i in range(k_folds)
    ]

    model_paths = [str(model_path) for model_path in model_paths if model_path.exists()]
    if len(model_paths) == 0:  # This is not a CV modelling thing. Don't add .kfold
        model_path = models_root / model_version / "weights" / "best.pt"
        if not model_path.exists():  # The model simply isn't available
            raise RuntimeError("Model not found.")
        model_paths = [str(model_path)]

    return model_paths


def test_get_paths_to_weights():
    yolo_root = Path("/home/david/addn_repos/yolov5")
    k_folds = 6
    model_version = "srd31.1"
    model_paths = get_paths_to_weights(
        yolo_root=yolo_root, k_folds=k_folds, model_version=model_version
    )
    print(model_paths)
