import shutil
from pathlib import Path
from typing import List, Optional

import pandas
from yo_ratchet.yo_filter.filter_central import calculate_wedge_envelop
from yo_ratchet.yo_filter.filters import apply_filters
from yo_ratchet.yo_wrangle.common import (
    YOLO_ANNOTATIONS_FOLDER_NAME,
    get_all_jpg_recursive,
)


def copy_images_and_annotations_listed_in_aggregated_annotations_file(
    src_images_dir: Path,
    filtered_annotations_file: Path,
    dst_images_dir: Path,
    copy_all_src_images: bool = False,
):
    """
    Copy images into a sample folder based on a single file containing all
    YOLO detections. These likely have been filtered based on probability
    thresholds, size and location filters. Only copies image if there is a
    corresponding annotations file; i.e. no detections, not mined. This
    keeps the dataset economically sized.

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
        if not copy_all_src_images:
            dst_image_path = dst_images_dir / original_image_path.name
            shutil.copy(src=str(original_image_path), dst=str(dst_image_path))
        df_filtered = df.loc[df[0] == photo_name, [1, 2, 3, 4, 5, 6]]
        dst_annotations_path = dst_annotations_dir / f"{original_image_path.stem}.txt"
        df_filtered.to_csv(dst_annotations_path, index=False, sep=" ", header=None)

    if copy_all_src_images:
        for image_path in get_all_jpg_recursive(img_root=src_images_dir):
            dst_image_path = dst_images_dir / image_path.name
            shutil.copy(src=str(image_path), dst=str(dst_image_path))


def filter_and_aggregated_annotations(
    annotations_dir: Path,
    classes_json_path: Path,
    output_path: Optional[Path] = Path.cwd() / "annotations.ai",
    filter_horizon: Optional[float] = None,
    y_wedge_apex: Optional[float] = None,
    classes_to_remove: Optional[List[int]] = None,
    global_object_width_threshold: float = 0.0,
    looseness: float = 1.0,
) -> List[str]:
    """
    Combines multiple multi-line annotation files into one big annotations file
    filtering out low probability defects according to parameters supplied.

    If the detections resulted from running the AI on un-transformed images, then set
    reverse_transform = False as the bounding boxes will already be positioned correctly.

    If detections were based on transformed images and you want to show bounding boxes
    on un-transformed images, the set reverse_transform = True.
    However, if you may want to show the bounding boxes on the transformed images;
    in such case set reverse_transform to true.

    If the detections resulted from running the AI on untransformed images, then set
    filter_horizon to 0.5 to remove detections above a certain horizon.

    Note::
        Horizon is measured from the top of the image. A bigger number applied to
        filter_horizon will potentially remove more defects.

    """
    if (annotations_dir / "labels").exists() and (annotations_dir / "labels").is_dir():
        annotations_dir = annotations_dir / "labels"
    else:
        pass  # All cool, user didn't make a mistake

    annotation_files: List = sorted(annotations_dir.rglob("*.txt"))
    if len(annotation_files) == 0:
        print(f"\nNo files found in {annotations_dir}")

    if y_wedge_apex:
        wedge_constants, wedge_gradients = calculate_wedge_envelop(
            x_apex=0.5,
            y_apex=y_wedge_apex,
        )
    else:
        wedge_constants, wedge_gradients = None, None

    aggregated_detections = []
    for file_path in annotation_files:
        with open(str(file_path), "r") as f:
            lines = f.readlines()

        lines = apply_filters(
            lines=lines,
            classes_json_path=classes_json_path,
            object_threshold_width=global_object_width_threshold,
            filter_horizon=filter_horizon,
            wedge_constants=wedge_constants,
            wedge_gradients=wedge_gradients,
            classes_to_remove=classes_to_remove,
            marginal_classes=None,
            min_count_marginal=None,
            looseness=looseness,
        )

        new_lines = []
        for line in lines:
            line = [str(element) for element in line]
            line = " ".join(line)
            new_lines.append(f"{file_path.stem}.jpg {line}\n")
        if output_path is not None:
            with open(output_path, "a") as f_out:
                f_out.writelines(new_lines)
        aggregated_detections.extend(new_lines)

    if output_path is not None:
        print(f"See results at {str(output_path)}")
    return aggregated_detections
