import os
import pandas as pd
import shutil
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, List, Tuple, Set

from yo_ratchet.yo_filter.filters import get_classes_info, get_lower_probability_thresholds
from yo_ratchet.yo_wrangle.common import YOLO_ANNOTATIONS_FOLDER_NAME, get_subsets_included
from yo_ratchet.yo_wrangle.gcs_interface import download_training_data_for_a_subset_from_a_single_yolo_file
from yo_ratchet.yo_wrangle.wrangle import _get_hits_for_annotations_in_classes, delete_redundant_samples, \
    cleanup_excess_annotations
from yo_ratchet.yo_wrangle.aggregated_annotations import (
    filter_and_aggregate_annotations,
    copy_training_data_listed_in_aggregated_annotations_file,
    copy_training_data_listed_in_aggregated_annotations_df,
)
from yo_ratchet.yo_filter.unsupervised import (
    OutlierParams,
    OutlierDetectionConfig,
)

PHOTO_NAME = "photo_name"
"""
This module is about filtering object detections that are already in one or more yolo files.

See the 'reaper' module for higher level functions that possibly download images and 
run detections before calling filtering functions in this module.

"""


def extract_high_confidence_training_data(
    src_folder: Path,
    target_dataset_root: Path,
    classes_json: Optional[Path] = None,
    move: bool = False
):
    """
    Use this functions to find a subset of high-confidence training data located in the
    folder <src_root> and copy that training data to folder f"{<dst_root>}/{<src_folder>.name}".
    Lower-confidence detections are retained for an image so long as there is at least one
    high-confidence object detection.

    Does not filter out object detection based on x,y co-ordinates as
    we wish to collect all high-confidence detections, whether they be a true-positive
    or a false-positive. False positives help the model learn to learn from the pre-labelling
    mistakes.

    Only copies image if not already existing anywhere within the overall dataset defined by
    the nominated <target_dataset> root directory or that inferred relative to the virtual environment
    in which yo-laborate has been installed.

    Threshold "min_prob" probabilities must be provided by a classes.json file which can be
    inferred based on <target_dataset>/classes.json. However, if the above file contains production
    thresholds, you may wish to explicitly provide a path to another classes.json file via
    <classes_json>.

    :return:
    """
    annotations_dir = src_folder / "YOLO_Darknet"
    dst_images_dir = target_dataset_root / src_folder.name
    existing_image_names = [image.name for image in target_dataset_root.rglob("*.jpg")]
    extract_high_quality_training_data_from_raw_detections_new_images_only(
        src_images_dir=src_folder,
        src_annotations_dir=annotations_dir,
        dst_images_dir=dst_images_dir,
        classes_json_path=classes_json,
        existing_image_names=existing_image_names,
        lower_probability_coefficient=1.2,
        upper_probability_coefficient=None,
        marginal_coefficient=1.2,
        filter_horizon=0.0,
        y_wedge_apex=None,
        marginal_classes=[5],
        min_marginal_count=5,
        copy_all_src_images=False,
        outlier_config=None,
        move=move
    )


def extract_high_quality_training_data_from_raw_detections_new_images_only(
    src_images_dir: Path,
    src_annotations_dir: Path,
    dst_images_dir: Path,
    classes_json_path: Path,
    existing_image_names: Optional[List] = None,
    lower_probability_coefficient: float = 0.7,
    upper_probability_coefficient: Optional[float] = None,
    marginal_coefficient: Optional[float] = None,
    filter_horizon: float = 0.0,
    y_wedge_apex: Optional[float] = -0.2,
    marginal_classes: Optional[List[int]] = None,
    min_marginal_count: Optional[int] = None,
    copy_all_src_images: bool = False,
    outlier_config: Optional[OutlierDetectionConfig] = None,
    move: bool = False,
):
    """
    Function to selectively copy images and pre-labelled yolo annotations. Reduces false positives where
    an image ONLY has low likelihood object detections with unlikely x, y position or low confidence level.
    However, all pre-labelling will be retained for an image if it has at least one associated highly
    quality object detection.

    Use this function if you want to minimize time to review training images as it will filtered out
    more false-positives as compared to the function extract_high_confidence_training_data().

    This function will copy images from::
        <src_images_dir>/
    to::
        <dst_images_dir>/

    and filtered annotation files from::
        <annotations_dir>/
    to::
        <dst_images_dir>/YOLO_darknet/*

    Filtering bounding box data is enabled according to the following parameters::

        * classes_json_path: Path to a classes information json file which contains
          minimum probability thresholds defined for production usage for each individual
          class. These thresholds are applied to filtering in combination with the parameters
          lower_probability_coefficient and upper_probability_coefficient defined below.

        * lower_probability_coefficient: The minimum probability threshold is this
          coefficient multiplied by the production probability threshold defined for
          each class in class. Choose a value in the range [0-inf]. Higher values
          result in more accurate training data, but too high will result in no training
          data being extracted.

        * upper_probability_coefficient: Set to None to not apply any filtering based
          on an upper limit on confidence, or for selectively mining difficult positives,
          set to a value  lower_probability_coefficient < upper_probability_coefficient < 1

        * filter_horizon: removes objects that have a centre above this normalised
          y-value (range[0-1]). i.e. Only keep objects in the image foreground.

        * wedge_apex: all objects are removed from top-left and top-right corners
          according to a wedge shape. You can choose the position of the apex.
          See docstring TODO: XXXX for more explanation.

        * marginal_classes: Remove any images that ONLY contain this selection of
          class_ids.

        * outlier_params: Removes outliers if this dict is provided.

        * global_object_width_threshold: TODO: Also, apply area and diagonal length filters?

    By default, only images associated with the filtered annotations will be copied to
    dst_images_dir. Optionally, all images in src_images_dir can be copied to dst_images_dir.
    Set copy_all_src_images = True when you wish to increase the weight of hard negatives.

    """
    classes_info = get_classes_info(classes_json_path=classes_json_path)
    if outlier_config:
        raise NotImplementedError
        # outlier_params = OutlierParams(
        #     classes_info=classes_info, outlier_config=outlier_config
        # )
    else:
        outlier_params = None

    filtered_detections = filter_and_aggregate_annotations(
        annotations_dir=src_annotations_dir,
        classes_info=classes_info,
        lower_probability_coefficient=lower_probability_coefficient,
        upper_probability_coefficient=upper_probability_coefficient,
        output_path=None,
        filter_horizon=filter_horizon,
        y_wedge_apex=y_wedge_apex,
        marginal_classes=marginal_classes,
        min_marginal_count=min_marginal_count,
        images_root=src_images_dir,
        outlier_params=outlier_params if marginal_coefficient is None else None,
        remove_probability=True,
    )
    if marginal_coefficient is not None:  # Only include marginal bounding boxes if at least one box > min_prob
        less_filtered = filter_and_aggregate_annotations(
            annotations_dir=src_annotations_dir,
            classes_info=classes_info,
            lower_probability_coefficient=marginal_coefficient,
            upper_probability_coefficient=upper_probability_coefficient,
            output_path=None,
            filter_horizon=filter_horizon,
            y_wedge_apex=y_wedge_apex,
            marginal_classes=marginal_classes,
            min_marginal_count=min_marginal_count,
            images_root=src_images_dir,
            outlier_params=outlier_params,
            remove_probability=True,
        )
        less_filtered_df = pd.DataFrame(less_filtered)  # , columns = [...]) ~ Risky as makes assumption about num cols
        # Assumes prob field: columns=[PHOTO_NAME, "class_id", "x_centre", "y_centre", "width", "height", "prob"])
        less_filtered_df = less_filtered_df.rename(columns={0: PHOTO_NAME})
        image_names_short_list = list({row.split(" ")[0] for row in filtered_detections})
        for name in image_names_short_list:
            if name in existing_image_names:
                print(f"{name} already exists in target dataset.")
        image_names_short_list = [name for name in image_names_short_list if name not in existing_image_names]
        plus_marginal_detections = less_filtered_df.loc[
            less_filtered_df[PHOTO_NAME].isin(image_names_short_list)
        ]
        copy_training_data_listed_in_aggregated_annotations_df(
            src_images_dir=src_images_dir,
            df_filtered_annotations=plus_marginal_detections,
            dst_images_dir=dst_images_dir,
            copy_all_src_images=copy_all_src_images,
            move=move
        )
    else:
        filtered_detections_df = pd.DataFrame(filtered_detections)
        if existing_image_names:
            filtered_detections_df = filtered_detections_df.loc[
                ~filtered_detections_df[PHOTO_NAME].isin(existing_image_names)
            ]
        copy_training_data_listed_in_aggregated_annotations_df(
            src_images_dir=src_images_dir,
            df_filtered_annotations=filtered_detections_df,
            dst_images_dir=dst_images_dir,
            copy_all_src_images=copy_all_src_images,
            move=move
        )


def extract_high_quality_training_data_from_raw_detections(
    src_images_dir: Path,
    annotations_dir: Path,
    dst_images_dir: Path,
    classes_json_path: Path,
    lower_probability_coefficient: float = 0.7,
    upper_probability_coefficient: Optional[float] = None,
    marginal_coefficient: Optional[float] = None,
    filter_horizon: float = 0.0,
    y_wedge_apex: Optional[float] = -0.2,
    marginal_classes: Optional[List[int]] = None,
    min_marginal_count: Optional[int] = None,
    copy_all_src_images: bool = False,
    outlier_config: Optional[OutlierDetectionConfig] = None,
    move: bool = False,
):
    """
    Function to selectively copy images and pre-labelled yolo annotations. Reduces false positives where
    an image ONLY has low likelihood object detections with unlikely x, y position or low confidence level.
    However, all pre-labelling will be retained for an image if it has at least one associated highly
    quality object detection.

    Use this function if you want to minimize time to review training images as it will filtered out
    more false-positives as compared to the function extract_high_confidence_training_data().

    This function will copy images from::
        <src_images_dir>/
    to::
        <dst_images_dir>/

    and filtered annotation files from::
        <annotations_dir>/
    to::
        <dst_images_dir>/YOLO_darknet/*

    Filtering bounding box data is enabled according to the following parameters::

        * classes_json_path: Path to a classes information json file which contains
          minimum probability thresholds defined for production usage for each individual
          class. These thresholds are applied to filtering in combination with the parameters
          lower_probability_coefficient and upper_probability_coefficient defined below.

        * lower_probability_coefficient: The minimum probability threshold is this
          coefficient multiplied by the production probability threshold defined for
          each class in class. Choose a value in the range [0-inf]. Higher values
          result in more accurate training data, but too high will result in no training
          data being extracted.

        * upper_probability_coefficient: Set to None to not apply any filtering based
          on an upper limit on confidence, or for selectively mining difficult positives,
          set to a value  lower_probability_coefficient < upper_probability_coefficient < 1

        * filter_horizon: removes objects that have a centre above this normalised
          y-value (range[0-1]). i.e. Only keep objects in the image foreground.

        * wedge_apex: all objects are removed from top-left and top-right corners
          according to a wedge shape. You can choose the position of the apex.
          See docstring TODO: XXXX for more explanation.

        * marginal_classes: Remove any images that ONLY contain this selection of
          class_ids.

        * outlier_params: Removes outliers if this dict is provided.

        * global_object_width_threshold: TODO: Also, apply area and diagonal length filters?

    By default, only images associated with the filtered annotations will be copied to
    dst_images_dir. Optionally, all images in src_images_dir can be copied to dst_images_dir.
    Set copy_all_src_images = True when you wish to increase the weight of hard negatives.

    """
    classes_info = get_classes_info(classes_json_path=classes_json_path)
    if outlier_config:
        raise NotImplementedError
        # outlier_params = OutlierParams(
        #     classes_info=classes_info, outlier_config=outlier_config
        # )
    else:
        outlier_params = None

    fd, filtered_annotations_path = mkstemp(suffix=".txt")   # move this code down to near first use of 'fd'
    filtered_detections = filter_and_aggregate_annotations(
        annotations_dir=annotations_dir,
        classes_info=classes_info,
        lower_probability_coefficient=lower_probability_coefficient,
        upper_probability_coefficient=upper_probability_coefficient,
        output_path=None,
        filter_horizon=filter_horizon,
        y_wedge_apex=y_wedge_apex,
        marginal_classes=marginal_classes,
        min_marginal_count=min_marginal_count,
        images_root=src_images_dir,
        outlier_params=outlier_params if marginal_coefficient is None else None,
        remove_probability=True,
    )
    if marginal_coefficient is not None:  # Only include marginal bounding boxes if at least one box > min_prob
        fd, less_filtered_path = mkstemp(suffix=".txt")
        _ = filter_and_aggregate_annotations(
            annotations_dir=annotations_dir,
            classes_info=classes_info,
            lower_probability_coefficient=marginal_coefficient,
            upper_probability_coefficient=upper_probability_coefficient,
            output_path=Path(less_filtered_path),
            filter_horizon=filter_horizon,
            y_wedge_apex=y_wedge_apex,
            marginal_classes=marginal_classes,
            min_marginal_count=min_marginal_count,
            images_root=src_images_dir,
            outlier_params=outlier_params,
            remove_probability=True,
        )
        less_filtered = pd.read_csv(less_filtered_path, sep=" ", header=None)
        less_filtered = less_filtered.rename(columns={0: PHOTO_NAME})
        image_names_short_list = list({row.split(" ")[0] for row in filtered_detections})
        plus_marginal_detections = less_filtered.loc[
            less_filtered[PHOTO_NAME].isin(image_names_short_list)
        ]
        plus_marginal_detections.to_csv(
            filtered_annotations_path, index=False, sep=" ", header=None
        )
        os.unlink(less_filtered_path)
    else:
        with open(filtered_annotations_path, "w") as fd:
            fd.writelines(filtered_detections)

    copy_training_data_listed_in_aggregated_annotations_file(
        src_images_dir=src_images_dir,
        filtered_annotations_file=Path(filtered_annotations_path),
        dst_images_dir=dst_images_dir,
        copy_all_src_images=copy_all_src_images,
        move=move,
    )
    os.unlink(filtered_annotations_path)


def selectively_copy_training_data_for_selected_classes(
    classes: List[int],
    src_images_dir: Path,
    dst_images_dir: Path,
    sample_size: Optional[int] = None,
):
    """
    Copies images and their annotations (assumed to be in the ./YOLO_darknet
    sub-folder) to a new location filtering for selected classes. Useful when
    you have to prioritise which classes you need to improve first.

    Not used often because it was easier to just select classes to detect in the
    command for yolov5/detect.py

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


def prepare_training_data_subset_from_reviewed_yolo_file(
    images_archive_dir: Path,
    yolo_file: Path,
    dst_images_dir: Path,
    classes_json_path: Path,
    copy_all_src_images: bool = False,
    move: bool = False,
    probability_thresh_coefficient: float = 0.9
):
    """
    Filters retains only images having confirmed or deleted annotations from reviewed bounding box data (.yolo file)
    and prepares the data into training data structure, then extracts all annotations relating to those images
    regardless of whether edited or raw. This saves time for the bounding box auditor in that they only
    have to confirm, deny or add one bounding box per image to ensure that the image is included in training
    data all with all the unedited pre-labelling.

    This version filters out low confidence boxes. Saves time when auditing.

    images_archive_dir: You can either provide a folder Path containing sub-folders of images,
                               or simply a folder Path which directly contains images.
    yolo_file: A path to a single text file containing aggregated bounding box annotations covering
               all objects defined for images in images_archive_dir.
    """
    classes_info = get_classes_info(classes_json_path=classes_json_path)
    lower_prob_thresholds = get_lower_probability_thresholds(
        classes_info=classes_info,
        lower_probability_coefficient=probability_thresh_coefficient,
    )

    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    hit_list = set()  # Identify photos that have had at least one defect confirmed or deleted
    for line in lines:
        line_split = line.split(" ")
        conf = float(line_split[6])
        if 0 < conf < 1:  # 1 and 0 are the probabilities for confirmed and denied annotations respectively.
            continue  # Only accept co
        # class_id = int(line_split[1])
        # if class_id not in [3, 4, 8, 9, 17, 19, 22, 29, 30, 33]:
        #     continue
        # else:
        #     pass
        photo_name = line_split[0]
        hit_list.add(photo_name)

    # Collect all annotations for photos in the hit-list, but remove probability (last field)
    # (regardless whether box was confirmed or not) except defects below the minimum confidence
    # Threshold
    filtered_detections = []
    for line in lines:
        line_split = line.split(" ")
        photo_name = line_split[0]
        if photo_name not in hit_list:
            continue
        else:
            pass
        conf = float(line_split[6])
        class_id = line_split[1]
        lower_threshold = lower_prob_thresholds.get(int(class_id), 0.1)
        if conf < lower_threshold:
            continue
        else:
            pass
        revised_line = " ".join(line_split[:6])
        filtered_detections.append(revised_line)

    fd, filtered_annotations_path = mkstemp(suffix=".txt")
    with open(filtered_annotations_path, "w") as fd:
        fd.write("\n".join(filtered_detections))

    sub_dir_paths = [x for x in images_archive_dir.iterdir() if x.is_dir()]
    sub_dir_paths = [x.name for x in sub_dir_paths if x.name.lower() != YOLO_ANNOTATIONS_FOLDER_NAME.lower()]
    if len(sub_dir_paths) == 0:
        sub_dir_paths = [images_archive_dir]
    else:
        pass  # A parent directory was passed in as a parameter as assumed in the first instance.

    for sub_dir_path in sub_dir_paths:
        copy_training_data_listed_in_aggregated_annotations_file(
            src_images_dir=sub_dir_path,
            filtered_annotations_file=Path(filtered_annotations_path),
            dst_images_dir=dst_images_dir,
            copy_all_src_images=copy_all_src_images,
            move=move,
        )
    os.unlink(filtered_annotations_path)


def prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file(
    images_archive_dir: Path,
    yolo_file: Path,
    dst_images_dir: Path,
    copy_all_src_images: bool = False,
    move: bool = False,
):
    """
    Filters retains only images having confirmed or deleted annotations from reviewed bounding box data (.yolo file)
    and prepares the data into training data structure, then extracts all annotations relating to those images
    regardless of whether edited or raw. This saves time for the bounding box auditor in that they only
    have to confirm, deny or add one bounding box per image to ensure that the image is included in training
    data all with all the unedited pre-labelling.

    images_archive_dir: You can either provide a folder Path containing sub-folders of images,
                               or simply a folder Path which directly contains images.
    yolo_file: A path to a single text file containing aggregated bounding box annotations covering
               all objects defined for images in images_archive_dir.

.
    """
    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    hit_list = set()  # Identify photos that have had at least one defect confirmed or deleted
    for line in lines:
        line_split = line.split(" ")
        conf = float(line_split[6])
        if 0 < conf < 1:  # 1 and 0 are the probabilities for confirmed and denied annotations respectively.
            continue  # Only accept co
        # class_id = int(line_split[1])
        # if class_id not in [3, 4, 8, 9, 17, 19, 22, 29, 30, 33]:
        #     continue
        # else:
        #     pass
        photo_name = line_split[0]
        hit_list.add(photo_name)

    # Collect all annotations for photos in the hit-list regardless of whether the
    # individual box was confirmed or not.
    # Removes probability (last field) if present.
    filtered_detections = []
    for line in lines:
        line_split = line.split(" ")
        photo_name = line_split[0]
        if photo_name not in hit_list:
            continue
        else:
            pass
        conf = float(line_split[6])
        if conf <= 0:  # Don't include denied boxes
            continue
        revised_line = " ".join(line_split[:6])
        filtered_detections.append(revised_line)

    fd, filtered_annotations_path = mkstemp(suffix=".txt")
    with open(filtered_annotations_path, "w") as fd:
        fd.write("\n".join(filtered_detections))

    sub_dir_paths = [x for x in images_archive_dir.iterdir() if x.is_dir()]
    sub_dir_paths = [x.name for x in sub_dir_paths if x.name.lower() != YOLO_ANNOTATIONS_FOLDER_NAME.lower()]
    if len(sub_dir_paths) == 0:
        sub_dir_paths = [images_archive_dir]
    else:
        pass  # A parent directory was passed in as a parameter as assumed in the first instance.

    for sub_dir_path in sub_dir_paths:
        copy_training_data_listed_in_aggregated_annotations_file(
            src_images_dir=sub_dir_path,
            filtered_annotations_file=Path(filtered_annotations_path),
            dst_images_dir=dst_images_dir,
            copy_all_src_images=copy_all_src_images,
            move=move,
        )
    os.unlink(filtered_annotations_path)


def join_multiple_yolo_files_without_duplication_or_overwrite(
    src_dir: Path,
    include_unedited: Optional[bool] = False
):
    """
    Combines the confirmed and denied bounding boxes content first from multiple
    yolo files in a way that prevents overwriting edited bounding boxes
    when the same bounding box was not edited in one of the other yolo files. The
    confidence (7th field) is retained in the output allowing for review of bounding
    auditing for quality purposes.

    Adding bounding boxes that have not been edited in any of the input yolo files
    is optional.

    Future: Adds in unique bounding boxes from un-edited content from yolo files.

    """
    master_filtered_detections = []
    master_hit_list = []
    file_paths = [x for x in src_dir.iterdir()]
    for yolo_file in file_paths:
        filtered_detections, hit_list = data_from_images_with_at_least_one_confirmed_or_denied_box(
            yolo_file=yolo_file
        )
        master_filtered_detections.extend(filtered_detections)
        master_hit_list.extend(hit_list)

    if include_unedited:
        for yolo_file in file_paths:
            unedited_detections = yolo_data_from_unedited_images(
                yolo_file=yolo_file, exclusion_list=master_hit_list
            )
            master_filtered_detections.extend(unedited_detections)
            break  # for Isaac 2023, the same yolo starting file was used consistently

    print(f"total hits before enforcing unique: {len(master_filtered_detections)}")
    master_filtered_detections = set(master_filtered_detections)
    print(f"total hits after enforcing unique: {len(master_filtered_detections)}")
    joined_file_path = src_dir / "joined.yolo"
    with open(joined_file_path, "w") as fd:
        fd.write("\n".join(master_filtered_detections))


def data_from_images_with_at_least_one_confirmed_or_denied_box(yolo_file: Path) -> Tuple[List[str], Set[str]]:
    """
    Returns condensed yolo data (as a list) containing only

    The confidence/probability (7th and last field) been retained to allow for
    review of bounding box edits via Virtual RACAS software.

    """
    filtered_detections = []
    hit_list = set()  # Identify photos that have had at least one defect confirmed or deleted

    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.split(" ")
        conf = float(line_split[6])
        if 0 < conf < 1:  # 1 and 0 are the probabilities for confirmed and denied annotations respectively.
            continue  # Only accept confirmed or denied
        photo_name = line_split[0]
        hit_list.add(photo_name)

    # Collect all annotations for photos in the hit-list regardless of whether the
    # individual box was confirmed or not.
    for line in lines:
        line_split = line.split(" ")
        photo_name = line_split[0]
        if photo_name not in hit_list:
            continue
        else:
            pass
        filtered_detections.append(line)
    print(f"{yolo_file.name} has {len(hit_list)} hits.")
    return filtered_detections, hit_list


def yolo_data_from_unedited_images(yolo_file: Path, exclusion_list: List[str]) -> List[str]:
    """
    Return yolo data as a list with confidence/probability (last field) data stripped out.

    """
    filtered_detections = []

    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    # Collect all annotations for photos except for those in the hit-list
    for line in lines:
        line_split = line.split(" ")
        photo_name = line_split[0]
        if photo_name in exclusion_list:
            continue
        else:
            pass
        # revised_line = " ".join(line_split[:6])
        filtered_detections.append(line)

    return filtered_detections


def download_images_and_prepare_unique_training_data(
    storage_client,
    yolo_file: Path,
    bucket_name: str,
    images_prefix: str,
    download_dst: Path,
    final_dst: Path
):
    """
    This is a bit slow because it downloads all images in yolo file, not just the images
    which correspond to confirmed or denied bounding boxes.

    First downloads all images referenced in a yolo file, then prepares a folder of
    unique images and their corresponding text files containing annotations.

    The location of existing training data subsets is inferred from final_dst.
    To ensure that only unique images are added to your training data set, make sure that
    final_dst is a path to be created within your training data repository.

    """
    assert yolo_file.exists()
    assert not final_dst.exists(), "Final destination must not already exist for uniqueness. Merge afterwards."
    download_training_data_for_a_subset_from_a_single_yolo_file(
        bucket_name=bucket_name,
        storage_client=storage_client,
        yolo_file=yolo_file,
        dst_folder=download_dst,
        images_prefix=images_prefix
    )

    assert download_dst.exists()
    len_images = len(list(download_dst.rglob("*.jpg")))
    assert len_images > 0

    base_dir = final_dst.parent
    other_sample_folders = get_subsets_included(base_dir=base_dir)  # do prior to adding final_dst
    for folder in other_sample_folders:
        cleanup_excess_annotations(subset_folder=folder)

    prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file(
        images_archive_dir=download_dst,
        yolo_file=yolo_file,
        dst_images_dir=final_dst,
        copy_all_src_images=False,
        move=True  # Do dry run before changing this parameter to True
    )

    delete_redundant_samples(
        sample_folder_to_clean=final_dst,
        other_sample_folders=other_sample_folders,
    )
