import os
import pandas as pd
import shutil
from pathlib import Path
from tempfile import mkstemp
from typing import Optional, List

from yo_ratchet.yo_filter.filters import get_classes_info, get_lower_probability_thresholds
from yo_ratchet.yo_wrangle.common import YOLO_ANNOTATIONS_FOLDER_NAME
from yo_ratchet.yo_wrangle.wrangle import _get_hits_for_annotations_in_classes
from yo_ratchet.yo_wrangle.aggregated_annotations import (
    filter_and_aggregate_annotations,
    copy_training_data_listed_in_aggregated_annotations_file,
)
from yo_ratchet.yo_filter.unsupervised import (
    OutlierParams,
    OutlierDetectionConfig,
)


def extract_high_quality_training_data_from_yolo_runs_detect(
    src_images_dir: Path,
    annotations_dir: Path,
    dst_images_dir: Path,
    classes_json_path: Path,
    lower_probability_coefficient: float = 0.7,
    upper_probability_coefficient: Optional[float] = None,
    marginal_coefficient: Optional[float] = None,
    filter_horizon: float = 0.0,
    y_wedge_apex: float = -0.2,
    marginal_classes: Optional[List[int]] = None,
    min_marginal_count: Optional[int] = None,
    copy_all_src_images: bool = False,
    outlier_config: Optional[OutlierDetectionConfig] = None,
    move: bool = False,
):
    """
    Function to selectively copy images and annotations from a yolov5 detection run.

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
        outlier_params = OutlierParams(
            classes_info=classes_info, outlier_config=outlier_config
        )
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
        less_filtered = less_filtered.rename(columns={0: "photo_name"})
        image_names_short_list = list({row.split(" ")[0] for row in filtered_detections})
        plus_marginal_detections = less_filtered.loc[
            less_filtered["photo_name"].isin(image_names_short_list)
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


def prepare_training_data_subset_from_reviewed_annotations(
    images_archive_dir: Path,
    yolo_file: Path,
    dst_images_dir: Path,
    classes_json_path: Path,
    copy_all_src_images: bool = False,
    move: bool = False,
):
    """
    Filters for confirmed and deleted annotations from reviewed bounding box data (.yolo file)
    and prepares the data into training data structure.

    images_archive_dir: You can either provide a folder Path containing sub-folders of images,
                               or simply a folder Path which directly contains images.
    yolo_file: A path to a single text file containing aggregated bounding box annotations covering
               all objects defined for images in images_archive_dir.
    """
    classes_info = get_classes_info(classes_json_path=classes_json_path)
    lower_prob_thresholds = get_lower_probability_thresholds(
        classes_info=classes_info,
        lower_probability_coefficient=1.0,
    )

    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    hit_list = set()
    for line in lines:
        line_split = line.split(" ")

        conf = float(line_split[6])
        if conf not in [0, 1]:
            continue

        class_id = int(line_split[1])
        # if class_id not in [3, 4, 8, 9, 17, 19, 22, 29, 30, 33]:
        #     continue
        # else:
        #     pass

        photo_name = line_split[0]
        hit_list.add(photo_name)

    # copy line if photo_name is in hit-list, but remove probability (last field)
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


def test_prepare_training_data_subset_from_reviewed_annotations():
    prepare_training_data_subset_from_reviewed_annotations(
        images_archive_dir=Path("/media/david/Samsung_T8/bot_archives/western_downs_regional_council"),
        yolo_file=Path("/media/david/Carol_sexy/Defects_western_downs_regional_council_1664227892_edited.yolo"),
        dst_images_dir=Path("/home/david/RACAS/sealed_roads_dataset/WDRC_2022_Sep_Risk"),
        classes_json_path=Path("/home/david/RACAS/sealed_roads_dataset/classes.json"),
        copy_all_src_images=False,
        move=False  # Do dry run before changing this parameter to True
    )
