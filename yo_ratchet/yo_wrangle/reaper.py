from datetime import datetime
from pathlib import Path
from typing import Optional

from yo_ratchet.modelling import run_detection_return_inferences_root
from yo_ratchet.yo_wrangle.gcs_interface import download_all_blobs_in_bucket


def download_from_bucket_and_detect(
    storage_client,
    bucket_name: str,
    archive_root: Path,
    tag: Optional[str] = None,
    prefix: str = "Photo_2023"
):
    """
    Downloads from bucket and runs detections.

    This is going to be similar to the AI bot, except:
     * works through one bucket, and only one , to completion (no round-robin).
     * doesn't push results back up to a special results bucket in the cloud.
    """
    dst_root = archive_root / bucket_name
    download_all_blobs_in_bucket(
        storage_client=storage_client,
        bucket_name=bucket_name,
        prefix=prefix,
        dst_root=dst_root
    )

    if tag is not None:
        results_folder_name = f"{bucket_name}_{tag}"
    else:
        results_folder_name = bucket_name

    run_detection_return_inferences_root(
        images_root=dst_root,
        results_folder_name=results_folder_name,
        model_path,
        model_version,
        base_dir=,
        yolo_root=,
        conf_thres=,
        device=,
        img_size=
    )
