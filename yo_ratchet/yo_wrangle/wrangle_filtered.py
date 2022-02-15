import shutil
from pathlib import Path

import pandas

from yo_wrangle.common import YOLO_ANNOTATIONS_FOLDER_NAME


def copy_detections_and_images(
    src_images_dir: Path,
    filtered_annotations_file: Path,
    dst_images_dir: Path,
):
    """
    Copy images into a sample folder based on a single file containing all
    YOLO detections. These likely have been filtered based on probability
    thresholds, size and location filters.

    Copies a sample of original images from <src_images_dir> to::
        <dst_sample_dir>/*
    and annotations to::
        <dst_sample_dir>/YOLO_darknet/*

    Assumes content of annotations file conforms to:
    * No header row
    * Space separated data
    * Columns:
       Photo_Name class_id x_centre y_centre width height probability

    Processing steps:
    1. Reading annotations file into a panda.Dataframe, df
    2. Finds unique photo names from the first column of the annotations file,
    3. Looping for photo_name in unique_photo_names:
       - Copy image from src_images_dir to dst_sample_dir
       - Filter df_filtered = df.loc[df[0]==photo_name]
       - df_filtered.to_csv(df_filtered.loc[:,[1:]]) to create an individual
         annotation files which are saved in <dst_sample_dir>/YOLO_darknet

    """
    if not src_images_dir.exists():
        raise RuntimeError(f"Directory for src_images_dir not found: {src_images_dir}")
    if not filtered_annotations_file.exists():
        raise RuntimeError(
            f"Path to annotations_file not found: {filtered_annotations_file}"
        )
    if filtered_annotations_file.suffix == ".ai":
        raise RuntimeError(
            "File with .ai suffix may contain polygons.\n"
            "Change file suffix to .txt.\n"
            f"{filtered_annotations_file}"
        )
    dst_annotations_dir = dst_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    dst_annotations_dir.mkdir(parents=True, exist_ok=True)

    df = pandas.read_csv(filtered_annotations_file, sep=" ", header=None)
    unique_photo_names = df[0].unique().tolist()
    for photo_name in unique_photo_names:
        original_image_path = src_images_dir / photo_name
        if not original_image_path.exists():
            print(f"Image not found: {str(original_image_path)}")
            continue
        dst_image_path = dst_images_dir / original_image_path.name
        shutil.copy(src=original_image_path, dst=dst_image_path)
        df_filtered = df.loc[df[0] == photo_name, [1, 2, 3, 4, 5, 6]]
        dst_annotations_path = dst_annotations_dir / f"{original_image_path.stem}.txt"
        df_filtered.to_csv(dst_annotations_path, index=False, sep=" ", header=None)


def test_copy_detections_and_images():
    copy_detections_and_images(
        src_images_dir=Path("/home/david/RACAS/640_x_640/Scenic_Rim_2021_sealed"),
        filtered_annotations_file=Path(
            "/home/david/defect_detection/defect_detection/evaluate/Scenic_Rim_threshold_2b.txt"
        ),
        dst_images_dir=Path("/home/david/RACAS/640_x_640/Scenic_Rim_2021_mined"),
    )
