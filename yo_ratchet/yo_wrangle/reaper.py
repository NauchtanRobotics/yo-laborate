from pathlib import Path
from typing import Optional

from google.cloud import storage

from yo_ratchet.modelling import run_detection_return_inferences_root_yolov8
from yo_ratchet.yo_wrangle.common import get_config_items, get_id_to_label_map, inferred_base_dir, get_yolo_detect_paths
from yo_ratchet.yo_wrangle.gcs_interface import download_all_blobs_in_bucket
from yo_ratchet.yo_wrangle.mine import extract_high_quality_training_data_from_raw_detections
from yo_ratchet.yo_wrangle.stats import count_class_instances_in_datasets


def harvest_training_data_from_images_in_the_cloud(
    storage_client,
    bucket_name: str,
    archive_root: Path,
    base_dir: Path,  # dir which contains config.ini having `[YOLO] YOLO_ROOT` and [DATASET] CLASSES_JSON data
    model_path: Path,
    tag: Optional[str] = None,  # will become part folder name @{YOLO_ROOT}/runs/detect/<results folder name>
    prefix: str = "Photo_2023",
    lower_probability_coefficient: float = 1,  # coeff of min_prob: adjust to control number/quality of hits
    conf_thresh: float = 0.10,  # an absolute minimum threshold that will be allowed. Conserves time when reviewing hits
    img_size: int = 1024  # resizing width/height spec for input to neural net.
):
    """
    Downloads from bucket and runs detections, then creates a folder with potential
    training data based on the images with object detection hits above a certain threshold
    based on the `min_prob` defined per class in `<base_dir>/classes.json` multiplied by
    <lower_probability_coefficient.

    Finally, a breakdown of the images and box count is printed to screen.

    This is going to be similar to the AI bot, except:
     * works through one bucket, and only one , to completion (no round-robin).
     * doesn't push results back up to a special results bucket in the cloud.
     * instead, a training data set is added to <base_dir> including images and annotations
       for positive hits above
    """
    dst_archive = archive_root / bucket_name
    download_all_blobs_in_bucket(
        storage_client=storage_client,
        bucket_name=bucket_name,
        prefix=prefix,
        dst_root=dst_archive
    )

    if tag is not None:
        results_folder_name = f"{bucket_name}_{tag}"
    else:
        results_folder_name = bucket_name

    model_label = model_path.parent.parent.name

    inferences_root = run_detection_return_inferences_root_yolov8(
        images_root=dst_archive,
        results_folder_name=results_folder_name,
        model_path=model_path,
        model_version=model_label,
        base_dir=base_dir,
        conf_thresh=conf_thresh,
        device=0,
        img_size=img_size
    )

    _, _, _, _, _, _, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    dst_training_data = base_dir / results_folder_name
    extract_high_quality_training_data_from_raw_detections(
        src_images_dir=dst_archive,
        annotations_dir=inferences_root,
        dst_images_dir=dst_training_data,
        classes_json_path=classes_json_path,
        lower_probability_coefficient=lower_probability_coefficient,
        upper_probability_coefficient=None,
        marginal_coefficient=None,
        marginal_classes=None,
        min_marginal_count=20,
        copy_all_src_images=False,
        outlier_config=None,
        move=False,
    )
    sample_folders = [dst_training_data]

    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    class_ids = list(classes_map.keys())
    print()
    output_str = count_class_instances_in_datasets(
        data_samples=sample_folders,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )


JSON_CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "GOOGLE_APPLICATION_CREDENTIALS.json"


def test_harvest_training_data_from_images_in_the_cloud():
    bucket_name = "example_bucket"
    model_version = "tsd8.1"
    base_dir = Path("/home/david/example_dataset")  # inferred_base_dir()
    archive_root = Path("/media/david/Samsung_T8/archive_original_format_2023")

    assert JSON_CREDENTIALS_PATH.exists()
    assert base_dir.exists() and (base_dir / "classes.json").exists()
    assert archive_root.parent.exists()
    archive_root.mkdir(exist_ok=True)

    cred_path = str(JSON_CREDENTIALS_PATH)
    storage_client = storage.Client.from_service_account_json(
        # project="sacred-bonus-274204",
        json_credentials_path=cred_path
    )

    unused_python_path, yolo_path = get_yolo_detect_paths(base_dir)

    model_path = (
        yolo_path / "runs/train" / model_version / "weights/best.pt"
    )
    assert model_path.exists()
    harvest_training_data_from_images_in_the_cloud(
        storage_client=storage_client,
        bucket_name=bucket_name,
        archive_root=archive_root,
        base_dir=base_dir,
        model_path=model_path,
        tag="2022",
        prefix="Photo_2022",
        conf_thresh=0.65
    )
