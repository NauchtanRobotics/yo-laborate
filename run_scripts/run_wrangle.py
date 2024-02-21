"""
Standard dataset building steps::

    1. Flatten YOLO 'detect' folder structure and merge annotations with images.

        -> test_copy_recursive_images_and_yolo_annotations_by_cropped_image_reference()

    2. Uniformly sample positive YOLO detections from a project

        -> test_subsample_a_directory()

    3. Extract all unique samples for specific classes from confident detections
       to complement previous samples taken from YOLO detections (perhaps at lower confidence).

        -> test_prepare_unique_dataset_from_detections()

    4. Collate multiple samples from one project into a common directory

        -> test_collate_additional_sample(), adding new dataset to the list.

    4. Collate samples taken from various projects and split into train and validation
       data sets.

          -> test_collate_image_and_annotation_subsets()

"""
import shutil
import tempfile
from pathlib import Path

from yo_ratchet.yo_wrangle.common import get_classes_json_path, get_id_to_label_map
from yo_ratchet.yo_wrangle.stats import count_class_instances_in_datasets
from yo_ratchet.yo_wrangle.wrangle import (
    copy_images_recursive_inc_yolo_annotations_by_reference_dir,
    subsample_a_directory,
    collate_image_and_annotation_subsets,
    split_yolo_train_dataset_every_nth,
    prepare_unique_dataset_from_detections,
    copy_detect_folder_recursively_as_reference_then_subsample,
    delete_redundant_samples,
    collate_additional_sample,
    add_subset_folder_unique_images_only,
    flatten_images_dir,
    cleanup_excess_annotations,
)


def test_flatten_images_dir():
    src_root = "/media/david/Samsung_T8/Scenic_Rim_2022"
    dst_root = "/media/david/Samsung_T8/Scenic_Rim_2022_flat"  # "/home/david/RACAS/boosted/600_x_600/unmasked/RACAS_Gladstone_2020_mined"
    flatten_images_dir(
        src_root=src_root, dst_root=dst_root, subfolders_to_skip=[], every_n_th=1
    )


def test_copy_recursive_images_and_yolo_annotations_by_cropped_image_reference():
    """
    1. Flatten YOLO 'detect' folder structure and merge annotations with images.

    Merge images from <src_images> folder with detected "labels" according to reference
    images in "cropped". Source images are flattened into <dst_images> folder and
    annotations end up in <dst_images>/YOLO_darknet folder as required for editing
    bounding boxes when using OpenLabeling library

    Note::
        The cropped image reference doesn't need to be cropped, and
        doesn't even need to be a different reference to the original images dir.


    """
    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=Path(
            "/home/david/RACAS/640_x_640/Scenic_test_clustering"
            # "/home/david/addn_repos/yolov5/runs/detect/Charters_Towers_potholes__srd18.3_conf5pcnt",
        ),
        original_images_dir=Path(
            "/home/david/RACAS/640_x_640/Scenic_Rim_2021_mined_19.1"
        ),
        dst_sample_dir=Path(
            "/home/david/RACAS/sealed_roads_dataset/Scenic_clustering_stp2"
        ),
        num=None,
        move=False,
        annotations_location="yolo",  # "ref_yolo",
    )


def test_subsample_a_directory():
    dataset_name = "Whitsunday_2018_40pcnt"
    subsample_a_directory(
        src_images_root=Path(
            f"/home/david/RACAS/boosted/600_x_600/unmasked/{dataset_name}_all"
        ),
        dst_images_root=Path(
            f"/home/david/RACAS/boosted/600_x_600/unmasked/{dataset_name}_random_sample"
        ),
        every_n_th=6,
    )


def test_copy_recursive_by_reference_then_subsample():
    copy_detect_folder_recursively_as_reference_then_subsample(
        reference_dir=Path(
            "/home/david/RACAS/boosted/600_x_600/unmasked/Charters_Towers_2021_defects"
            # "/home/david/addn_repos/yolov5/runs/detect/ChartersTowers__Coll_8a_unweighted_65conf",
        ),
        original_images_dir=Path(
            "/media/david/Samsung_T8/640_x_640/RACAS_CTRC_2021"
        ),
        dst_sample_dir=Path(
            "/home/david/RACAS/boosted/600_x_600/unmasked/Charters_Towers_2021_defects2"
        ),
        annotations_location="ref_yolo",
        every_n_th=1,
    )


def test_collate_image_and_annotation_subsets():
    keep_class_ids = None  # None actually means keep all classes
    skip_class_ids = [13, 14, 15, 22]
    every_n_th = 4  # for the validation subset
    dst_root = Path(
        "/home/david/production/test/split_cabonne"
    )
    srd_root = Path("/home/david/production/sealed_roads_dataset")
    samples_required = [
        (srd_root / "Cabonne_2020"),
    ]
    temp_dir = tempfile.mkdtemp()
    collate_image_and_annotation_subsets(
        samples_required=samples_required,
        dst_folder=Path(temp_dir),
        skip_class_ids=skip_class_ids,
        keep_class_ids=keep_class_ids,
    )
    split_yolo_train_dataset_every_nth(
        src_images_root=Path(temp_dir),
        dst_dataset_root=dst_root,
        every_n_th=every_n_th,
        cross_validation_index=3
    )
    shutil.rmtree(temp_dir)
    final_subsets_included = [
        (dst_root / "train"),
        (dst_root / "val"),
    ]
    classes_json_path = get_classes_json_path(base_dir=srd_root)
    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    class_ids = [int(class_id) for class_id in classes_map.keys()]
    print("\n")
    count_class_instances_in_datasets(
        data_samples=final_subsets_included,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )


def test_prepare_unique_dataset_from_detections():
    existing_sample_dir = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Whitsunday_2018_sample_0"
    )
    new_sample_dir = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Whitsunday_2018_sample_1"
    )
    reference_dir = Path(
        "/home/david/addn_repos/yolov5/runs/detect/Whitsunday_2018_Trans_Collation_7__60conf"
    )
    original_images_dir = Path(
        "/media/david/Samsung_T8/640_x_640/RACAS_Whitsunday_2018"
    )
    selected_classes = ["D40", "EB", "RK"]

    prepare_unique_dataset_from_detections(
        reference_dir=reference_dir,
        original_images_dir=original_images_dir,
        dst_sample_dir=new_sample_dir,
        class_names=selected_classes,
        other_sample_folders=[existing_sample_dir],
    )


def test_collate_additional_sample():
    """
    Use this function when you want to amalgamate existing and new images into the
    same subfolder. See also add_unique().

    Cannot be done within prepare_unique_dataset_from_detections() as the bounding
    boxes need to be manually established.

    """
    existing_sample_dir = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Charters_Towers_2021_subsample"
    )
    additional_sample_dir = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Charter_Towers_2021_defects2"
    )
    dst_folder = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Charter_Towers_2021_combined_samples"
    )
    collate_additional_sample(
        existing_sample_dir=existing_sample_dir,
        additional_sample_dir=additional_sample_dir,
        dst_folder=dst_folder,
    )


def test_delete_redundant_samples():
    delete_redundant_samples(
        sample_folder_to_clean=Path(
            "/home/david/RACAS/boosted/600_x_600/unmasked/CentralCoast_35pcnt_EB"
        ),
        other_sample_folders=[
            Path(
                "/home/david/RACAS/boosted/600_x_600/unmasked/CentralCoast_10pcnt_L0_LG_WS"
            ),
            Path(
                "/home/david/RACAS/boosted/600_x_600/unmasked/CentralCoast_25pcnt_AP_D10_D20"
            ),
        ],
    )


def test_signs():
    sample_folders = [
        (
            Path("/home/david/RACAS/boosted/600_x_600/unmasked/Signs_Hobart_2021"),
            72,
        ),
        (
            Path(
                "/home/david/RACAS/boosted/600_x_600/unmasked/Signs_Central_Coast_2021"
            ),
            22,
        ),
    ]
    temp_images_root = (
        "/home/david/RACAS/boosted/600_x_600/unmasked/bbox_collation_signs"
    )
    collate_image_and_annotation_subsets(
        samples_required=sample_folders,
        dst_folder=Path(temp_images_root),
        keep_class_ids=None,  # [3,4]
    )
    dst_root = Path("/home/david/RACAS/boosted/600_x_600/unmasked/Signs_Only_Split")
    split_yolo_train_dataset_every_nth(
        src_images_root=Path(temp_images_root),
        dst_dataset_root=Path(dst_root),
        every_n_th=3,
    )


def test_add_unique():
    add_subset_folder_unique_images_only(
        existing_dataset_root=Path("/home/david/RACAS/sealed_roads_dataset"),
        src_new_images=Path("/home/david/RACAS/640_x_640/CT_D40_SD")
    )


def test_cleanup_excess_annotations():
    cleanup_excess_annotations(subset_folder=Path(
        "/home/david/RACAS/sealed_roads_dataset/Sunshine_Coast_2022_1")
    )


def test_collate_samples():
    sample_folders = [
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654062225"),
            None,
        ),
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654048134"),
            None,
        ),
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654050846"),
            None,
        ),
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654060521"),
            None,
        ),
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654064371"),
            None,
        ),
        (
            Path("/home/david/RACAS/Sunshine_Coast_1654065888"),
            None,
        ),
    ]
    temp_images_root = (
        "/home/david/RACAS/sealed_roads_dataset/Sunshine_Coast_2022_1"
    )
    collate_image_and_annotation_subsets(
        samples_required=sample_folders,
        dst_folder=Path(temp_images_root),
        keep_class_ids=None,  # [3,4]
    )
