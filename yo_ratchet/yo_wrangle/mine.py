import shutil
from pathlib import Path
from typing import Optional, List

from common import YOLO_ANNOTATIONS_FOLDER_NAME
from wrangle import _get_hits_for_annotations_in_classes


def mine_annotations_for_target_classes(
    classes: List[int],
    src_images_dir: Path,
    dst_images_dir: Path,
    sample_size: Optional[int] = None,
):
    """
    NOT IN USE.

    Makes a sample folder of image and annotation files from a source directory
    wherein the sample is selected according to a criteria where an annotation
    file includes at least one annotation for a priority class defined by the
    `classes` parameter.

    Assumes images all have '.jpg' extension (lower case).

    NOTE::
        I find it easier to mine by running detect.py with --save-crops --save-txt mode.
        The --conf-thres param can be tweaked to control the sensitivity as required.
        For example, if the model only has 20 instances of a class, set conf-thres to 0.05,
        or if we have >500 instances, set conf-thres = 0.5 or higher.

    I then delete any folders containing defects that I am not seeking (e.g.
    runs/detect/<RUN NAME>/crops/D00).

    Then I use copy_images_recursive_inc_yolo_annotations_by_cropped_image_reference()
    to prepare the sample dataset.
    Set the runs/detect/<RUN NAME> as the reference folder, and set
    `annotations_folder` = "labels".

    """
    dst_annotations_dir = dst_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    if dst_annotations_dir.exists():
        raise Exception("Destination folder already exists. Exiting.")
    dst_annotations_dir.mkdir(parents=True)
    src_annotations_dir = src_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME

    hit_list: List[str] = _get_hits_for_annotations_in_classes(
        classes=classes,
        src_images_dir=src_images_dir,
    )
    if sample_size and len(hit_list) > sample_size:
        # Step 1. Wedge filtering.
        pass  # hit_list = get_confident_hits()  # do we really only want the clearest hits?

    for image_name_stem in hit_list:
        src_annotation_path = src_annotations_dir / f"{image_name_stem}.txt"
        dst_annotation_path = dst_annotations_dir / f"{image_name_stem}.txt"
        shutil.copy(src=str(src_annotation_path), dst=str(dst_annotation_path))

        src_image_path = src_images_dir / f"{image_name_stem}.jpg"
        dst_image_path = dst_images_dir / f"{image_name_stem}.jpg"
        shutil.copy(src=str(src_image_path), dst=str(dst_image_path))
