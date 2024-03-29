"""
Standard dataset building steps::

    1. Merge images from <src_images> folder with detected "labels" according to reference
       images in "cropped". Source images are flattened into <dst_images> folder and
       annotations end up in <dst_images>/YOLO_darknet folder as required for editing
       bounding boxes when using OpenLabeling library. Use::

        -> copy_recursive_images_and_yolo_annotations_by_cropped_image_reference()

    2. Uniformly sample positive YOLO detections from a project. Use::

        -> subsample_a_directory()

    3. Extract all unique samples for specific classes from confident detections
       to complement previous samples taken from YOLO detections (perhaps at lower confidence).
       Use::

        -> prepare_unique_dataset_from_detections()

    4. Collate multiple samples from one project into a common directory. Use::

        -> collate_additional_sample()

    5. Collate samples taken from various projects and split into train and validation
       data sets. Use::

        -> collate_image_and_annotation_subsets()

"""
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from yo_ratchet.yo_wrangle.common import (
    get_all_jpg_recursive,
    get_all_txt_recursive,
    YOLO_ANNOTATIONS_FOLDER_NAME,
    LABELS_FOLDER_NAME,
)
from yo_ratchet.yo_wrangle.recode import recode_using_class_mapping


def flatten_images_dir(
    src_root: str,
    dst_root: str,
    subfolders_to_skip: Optional[List[str]] = None,
    every_n_th: int = 1,
):
    """
    Takes a collection of folders containing images recursively nested within <src_dir>
    and copies all the images into <dst_root> which is a flat structure that is suitable
    to use as the --source parameter for calls to yolov5.

    """
    imgs_root_dir = Path(src_root)
    destination_dir = Path(dst_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for i, path_image in enumerate(
        sorted(get_all_jpg_recursive(img_root=imgs_root_dir))
    ):
        if subfolders_to_skip and path_image.parent.name in subfolders_to_skip:
            continue
        if i % every_n_th != 0:
            continue
        new_image_path = destination_dir / path_image.name
        shutil.copyfile(str(path_image), str(new_image_path))


def copy_images_recursive_inc_yolo_annotations_by_reference_dir(
    reference_dir: Path,  # Can be the same as the original_images_dir when moving imgs
    original_images_dir: Path,
    dst_sample_dir: Path,
    num: Optional[int] = None,
    move: bool = False,
    annotations_location: str = "yolo",  # "yolo" | "ref_yolo" | "labels"
):
    """
    Used mainly to copy images into a sample folder based on cropped YOLO
    detections.

    Copies a sample of original images and their corresponding annotation files.
    Moves images to <dst_sample_dir>/ and annotations to
    <dst_sample_dir>/YOLO_darknet/

    The images folder structure is the destination is FLATTENED.

    Reference dir can be different to the original_images_dir when a reference dir
    has been cleaned up (e.g. images with unsealed roads or certain defect types
    have been deleted) but the reference dir does not contain the annotations so
    this directory is not directly helpful to move.

    Optional params that are particularly useful for splitting into train, val
    datasets::
        * :param num: int, Breaks after num images
        * :param move: bool, Move instead of copy.

    n.b. This function assumes by default that there are annotations in
    <original_images_dir>/YOLO_darknet (compatible with OpenLabel). However,
    using the annotations_location param this assumption can be changed to
    be compatible with the folder structure produce by YOLOv5 detections,
    i.e.:
        <reference_dir>/"cropped"
        <reference_dir>/"labels"

    """
    if (
        annotations_location == "yolo"
    ):  # YOLO_darknet dir nested in the original_images dir
        src_annotations_dir = original_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    elif (
        annotations_location == "ref_yolo"
    ):  # YOLO_darknet dir nested in withing the reference_dir directory
        src_annotations_dir = reference_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    elif (
        annotations_location == "labels"
    ):  # E.g. In a "labels" directory which is beside
        src_annotations_dir = reference_dir / annotations_location
    else:
        raise Exception(
            f"Annotations dir structure not allowed: {str(reference_dir / annotations_location)}"
        )

    dst_annotations_dir = dst_sample_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    dst_annotations_dir.mkdir(parents=True, exist_ok=True)
    for i, reference_image_path in enumerate(
        sorted(get_all_jpg_recursive(img_root=reference_dir))
    ):
        if num and i >= num:
            break
        original_image_path = original_images_dir / reference_image_path.name
        if (
            annotations_location == "labels" and not original_image_path.exists()
        ):  # Image is associated with > 1 crops.
            original_image_path_stem = original_image_path.stem[:-1]
            original_image_path = (
                original_image_path.parent / f"{original_image_path_stem}.jpg"
            )
            if (
                not original_image_path.exists()
            ):  # There were 10 or more crops, so trim stem by one more char.
                original_image_path_stem = original_image_path.stem[:-1]
                original_image_path = (
                    original_image_path.parent / f"{original_image_path_stem}.jpg"
                )
                if not original_image_path.exists():
                    print(f"Does not exist: {str(original_image_path)}")
                    continue

        dst_image_path = dst_sample_dir / original_image_path.name
        if move:
            shutil.move(src=str(original_image_path), dst=str(dst_image_path))
        else:
            shutil.copy(src=str(original_image_path), dst=str(dst_image_path))

        src_annotations_path = src_annotations_dir / f"{original_image_path.stem}.txt"
        if not src_annotations_path.exists():
            continue
        dst_annotations_path = (
            # dst_sample_dir
            # / YOLO_ANNOTATIONS_FOLDER_NAME
            dst_annotations_dir
            / f"{original_image_path.stem}.txt"
        )
        if move:
            shutil.move(src=str(src_annotations_path), dst=str(dst_annotations_path))
        else:
            shutil.copy(src=str(src_annotations_path), dst=str(dst_annotations_path))

    if annotations_location == "labels":
        for annotations_path in get_all_txt_recursive(root_dir=dst_annotations_dir):
            new_lines = []
            with open(str(annotations_path), mode="r+") as f:
                lines = f.readlines()
                for line in lines:
                    line_split = line.split(" ")
                    num_boxes = (len(line_split) - 1) // 4
                    if num_boxes > 1:
                        raise Exception("Take a looky here")
                    if len(line_split) > 5:
                        new_line = " ".join(line_split[0:-1])
                        new_lines.append(f"{new_line}\n")

            if len(line_split) > 5:
                with open(str(annotations_path), mode="w") as f:
                    f.writelines(new_lines)


def subsample_a_directory(
    src_images_root: Path,
    dst_images_root: Path,
    every_n_th: int = 6,
    move: bool = False,
):
    """
    Only intended to work on directory after annotations have been moved to "YOLO_darknet"
    folder (won't work directly on detection results where annotations are in "labels".
    Best to run copy_images_recursive_inc_yolo_annotations_by_reference_dir() on yolo
    detect results first.

    Similar to importer.subsample_a_directory() but additionally copies annotations.

    """
    dst_images_root.mkdir(parents=True)
    dst_annotations_root = dst_images_root / YOLO_ANNOTATIONS_FOLDER_NAME
    dst_annotations_root.mkdir(parents=True)
    for i, image_path in enumerate(get_all_jpg_recursive(img_root=src_images_root)):
        if i % every_n_th != 0:
            continue
        outfile = dst_images_root / image_path.name
        if move:  # Move the image
            shutil.move(src=str(image_path), dst=str(outfile))
        else:
            shutil.copyfile(src=str(image_path), dst=str(outfile))

        src_annotations_file = (
            src_images_root / YOLO_ANNOTATIONS_FOLDER_NAME / f"{image_path.stem}.txt"
        )
        dst_annotations_file = dst_annotations_root / f"{image_path.stem}.txt"
        if move:  # Move the annotation
            shutil.move(src=str(src_annotations_file), dst=str(dst_annotations_file))
        else:
            shutil.copyfile(
                src=str(src_annotations_file), dst=str(dst_annotations_file)
            )


def copy_detect_folder_recursively_as_reference_then_subsample(
    reference_dir: Path,
    original_images_dir: Path,
    dst_sample_dir: Path,
    annotations_location="labels",
    every_n_th=6,
):
    temp_file = tempfile.mkdtemp()
    temp_path = Path(temp_file)
    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=reference_dir,
        original_images_dir=original_images_dir,
        dst_sample_dir=temp_path,
        num=None,
        move=False,
        annotations_location=annotations_location,
    )
    subsample_a_directory(
        src_images_root=temp_path,
        dst_images_root=dst_sample_dir,
        every_n_th=every_n_th,
    )
    shutil.rmtree(str(temp_path))


def collate_image_and_annotation_subsets(
    samples_required: List[Path],
    dst_folder: Path,
    keep_class_ids: Optional[List[int]] = None,
    skip_class_ids: Optional[List[int]] = None,
):
    """
    Copies a sample of original images and their corresponding annotation files
    to <dst_folder>/ and <dst_folder>/YOLO_darknet/ respectively.

    This function assumes that there are annotations in original_images_dir/YOLO_darknet
    and will replicate this structure in the destination folder.

    :param samples_required: List of tuples of Path to folder to be sampled, qty images to sample
    :param dst_folder:       Destination for collated sample images/annotations.
    :param keep_class_ids:   List of class ids to for which annotations should be removed.To keep all
                             class ids, simply set to None (default).
    :param skip_class_ids:   List of class ids to for which annotations should be kept.

    """
    dst_folder.mkdir(exist_ok=True)
    for original_images_dir in samples_required:
        assert (
            original_images_dir.exists()
        ), f"Subset path not found: {str(original_images_dir)}"
        src_annotations_dir = original_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
        dst_annotations_folder = dst_folder / YOLO_ANNOTATIONS_FOLDER_NAME
        dst_annotations_folder.mkdir(exist_ok=True)
        original_image_paths = sorted(
            get_all_jpg_recursive(img_root=original_images_dir)
        )
        assert (
            len(original_image_paths) > 0
        ), f"Subset path has no jpg files: {str(original_images_dir)}"
        for i, original_image_path in enumerate(original_image_paths):
            dst_image_path = dst_folder / original_image_path.name
            # if dst_image_path.exists():
            #     dst_image_path = dst_folder / f"{original_image_path.stem}zzz{original_image_path.suffix}"
            if dst_image_path.exists():
                print(f"File name is not unique, skipping {str(dst_image_path.name)}")
                continue
            shutil.copy(src=str(original_image_path), dst=str(dst_image_path))

            src_annotations_path = (
                src_annotations_dir / f"{original_image_path.stem}.txt"
            )
            dst_annotations_path = (
                dst_folder
                / YOLO_ANNOTATIONS_FOLDER_NAME
                / f"{original_image_path.stem}.txt"
            )
            if src_annotations_path.exists():
                shutil.copy(
                    src=str(src_annotations_path), dst=str(dst_annotations_path)
                )
            else:
                print(f"Annotation file not found: {str(src_annotations_path)}")

        if keep_class_ids or skip_class_ids:
            filter_dataset_for_classes(
                annotations_dir=dst_annotations_folder,
                keep_class_ids=keep_class_ids,
                skip_class_ids=skip_class_ids,
            )


def split_yolo_train_dataset_every_nth(
    src_images_root: Path,
    dst_dataset_root: Path,
    every_n_th: int = 5,
    cross_validation_index: int = 0,
):
    """
    Copies images and annotations from <src_images_root> and <src_images_root>/YOLO_darknet
    respectively to a structure expected by yolov5/, as follows::

        <dst_dataset_root>/images/train/
        <dst_dataset_root>/images/val/
        <dst_dataset_root>/labels/train/
        <dst_dataset_root>/labels/val/

    """
    dst_dataset_root.mkdir(exist_ok=True, parents=True)
    dst_train_root = dst_dataset_root / "train"
    dst_train_root.mkdir()
    dst_val_root = dst_dataset_root / "val"
    dst_val_root.mkdir()

    dst_train_images_root = dst_train_root / "images"
    dst_train_images_root.mkdir()
    dst_train_labels_root = dst_train_root / "labels"
    dst_train_labels_root.mkdir()

    dst_val_images_root = dst_val_root / "images"
    dst_val_images_root.mkdir()
    dst_val_labels_root = dst_val_root / "labels"
    dst_val_labels_root.mkdir()

    for i, image_path in enumerate(get_all_jpg_recursive(img_root=src_images_root)):
        if i < cross_validation_index:
            continue
        src_annotations_file = (
            src_images_root / YOLO_ANNOTATIONS_FOLDER_NAME / f"{image_path.stem}.txt"
        )

        if (i - cross_validation_index) % every_n_th != 0:
            dst = dst_train_root
        else:
            dst = dst_val_root

        dst_image_path = dst / "images" / image_path.name
        shutil.copy(src=str(image_path), dst=str(dst_image_path))

        dst_annotations_file = dst / "labels" / src_annotations_file.name
        if src_annotations_file.exists():
            shutil.copy(src=str(src_annotations_file), dst=str(dst_annotations_file))
        else:
            print(f"Annotation file not found: {str(src_annotations_file)}")


def check_train_val_are_unique(dataset_path: Path):
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    train_image_names = [x.name for x in get_all_jpg_recursive(img_root=train_path)]
    val_image_names = [x.name for x in get_all_jpg_recursive(img_root=val_path)]
    assert len(set(val_image_names) & set(train_image_names)) == 0


def collate_and_split(
    subsets_included: List[Path],
    dst_root: Path,
    every_n_th: int = 1,  # for the validation subset.
    keep_class_ids: Optional[List] = None,  # None actually means keep all classes
    skip_class_ids: Optional[List] = None,
    recode_map: Optional[Dict[int, int]] = None,
    cross_validation_index: int = 0,
):
    if dst_root.exists():
        shutil.rmtree(str(dst_root))
    temp_dir = tempfile.mkdtemp()
    collate_image_and_annotation_subsets(
        samples_required=subsets_included,
        dst_folder=Path(temp_dir),
        skip_class_ids=skip_class_ids,
        keep_class_ids=keep_class_ids,
    )
    if recode_map is not None:
        annotations_root = Path(temp_dir) / YOLO_ANNOTATIONS_FOLDER_NAME
        recode_using_class_mapping(
            annotations_dir=annotations_root,
            recode_map=recode_map,
            only_retain_mapped_keys=False,
        )
    split_yolo_train_dataset_every_nth(
        src_images_root=Path(temp_dir),
        dst_dataset_root=dst_root,
        every_n_th=every_n_th,
        cross_validation_index=cross_validation_index,
    )
    check_train_val_are_unique(dataset_path=dst_root)
    shutil.rmtree(temp_dir)


def _get_hits_for_annotations_in_classes(
    classes: List[int],
    src_images_dir: Path,
) -> List[str]:
    """
    Returns list of image filename stems for annotation files which contain
    at least one of the classes specified in the param `classes`.

    """
    src_annotations_dir = src_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    hit_list: List[str] = []
    for src_annotation_path in get_all_txt_recursive(root_dir=src_annotations_dir):
        with open(str(src_annotation_path)) as f:
            lines = f.readlines()
        for line in lines:
            class_idx = int(line.strip().split(" ")[0])
            if class_idx in classes:
                hit_list.append(src_annotation_path.stem)
                break
    return hit_list


def delete_redundant_samples(
    sample_folder_to_clean: Path,
    other_sample_folders: List[Path],
):
    reference_image_paths = []
    for sample_folder in other_sample_folders:
        for image_path in get_all_jpg_recursive(sample_folder):
            reference_image_paths.append(image_path.name)
    unique_images = []
    for image_path in get_all_jpg_recursive(sample_folder_to_clean):
        image_name = image_path.name
        if image_name in reference_image_paths:
            image_path.unlink()
        if image_name in unique_images:
            image_path.unlink()
        else:
            unique_images.append(image_name)


def prepare_unique_dataset_from_detections(
    reference_dir: Path,
    original_images_dir: Path,
    dst_sample_dir: Path,
    class_names: List[str],
    other_sample_folders: List[Path],
):
    """
    1. Copies a detection directory to temporary location filtering for selected class folders.
    2. Run copy_by_reference to another destination directory
    3. Run delete_redundant_samples() applied to the destination directory.

    """
    temp_file = tempfile.mkdtemp()
    temp_path = Path(temp_file)
    src_folder = reference_dir / "labels"
    dst_folder = Path(temp_path) / "labels"
    shutil.copytree(src=str(src_folder), dst=str(dst_folder))
    src_crops_dir = reference_dir / "crops"
    dst_crops_dir = temp_path / "crops"
    dst_crops_dir.mkdir()
    for folder in src_crops_dir.iterdir():
        if folder.is_dir() and folder.name in class_names:
            dst_folder = temp_path / "crops" / folder.name
            shutil.copytree(src=str(folder), dst=str(dst_folder))

    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=Path(temp_path),
        original_images_dir=original_images_dir,
        dst_sample_dir=Path(dst_sample_dir),
        num=None,
        move=False,
        annotations_location="labels",
    )
    shutil.rmtree(temp_path)

    delete_redundant_samples(
        sample_folder_to_clean=dst_sample_dir,
        other_sample_folders=other_sample_folders,
    )


def filter_dataset_for_classes(
    annotations_dir: Path,
    keep_class_ids: Optional[List[int]] = None,
    skip_class_ids: Optional[List[int]] = None,
):
    """
    WARNING::
        Designed to be used on copies - permanently deletes annotations.

    """

    for annotations_path in get_all_txt_recursive(root_dir=annotations_dir):
        with open(str(annotations_path), "r") as file:
            annotation_lines = file.readlines()

        new_lines = []
        for annotation in annotation_lines:
            class_id = int(annotation.strip().split(" ")[0])
            if skip_class_ids and class_id in skip_class_ids:
                continue
            elif keep_class_ids and class_id not in keep_class_ids:
                continue
            new_lines.append(annotation)

        with open(str(annotations_path), "w") as file:
            file.writelines(new_lines)


def collate_additional_sample(
    existing_sample_dir: Path,
    additional_sample_dir: Path,
    dst_folder: Path,
):
    """
    Cannot be done within prepare_unique_dataset_from_detections() as the bounding
    boxes need to be manually established.

    """
    sample_folders = [
        existing_sample_dir,
        additional_sample_dir,
    ]
    collate_image_and_annotation_subsets(
        samples_required=sample_folders,
        dst_folder=dst_folder,
        keep_class_ids=None,
    )


def add_subset_folder_unique_images_only(
    existing_dataset_root: Path,
    src_new_images: Path,
):
    """
    Adds images and embedded YOLO_darknet annotations folder to a dataset where
    the images are a unique offering to the dataset.

    Does this by establishing a list of image names that already exist and
    then copies in additional images where the file name is not in the existing
    set.

    To keep this function simple, it assumes that all image names will be unique
    across all subset folders. This function should be modified to create a unique
    name if duplicate names are expected.

    The destination subset folder retains the folder name as per src_new_images.

    """
    dst_root: Path = existing_dataset_root / src_new_images.name
    (dst_root / YOLO_ANNOTATIONS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    existing_image_paths = get_all_jpg_recursive(img_root=existing_dataset_root)
    existing_image_names = [image_path.name for image_path in existing_image_paths]

    fresh_images_available_paths = list(get_all_jpg_recursive(img_root=src_new_images))

    for image_path in fresh_images_available_paths:
        if image_path.name in existing_image_names:
            continue
        new_image_path = dst_root / image_path.name
        annotation_name = f"{image_path.stem}.txt"
        src_annotation_path = (
            image_path.parent / YOLO_ANNOTATIONS_FOLDER_NAME / annotation_name
        )
        dst_annotation_path = dst_root / YOLO_ANNOTATIONS_FOLDER_NAME / annotation_name
        shutil.copy(src=str(image_path), dst=str(new_image_path))
        if src_annotation_path.exists():
            shutil.copy(src=str(src_annotation_path), dst=str(dst_annotation_path))


def _infer_annotations_folder_path(subset_folder: Path):
    """
    Infers path to annotations_folder assuming, in order of precedence, the
    path to be formulated as::

        * <subset_folder>/YOLO_darknet
        * <subset_folder>/labels
        * <subset_folder.parent>/labels

    :raises RuntimeError: if not of the above paths exist.

    """
    if (subset_folder / YOLO_ANNOTATIONS_FOLDER_NAME).exists():
        annotations_root = subset_folder / YOLO_ANNOTATIONS_FOLDER_NAME
    elif (subset_folder / LABELS_FOLDER_NAME).exists():
        annotations_root = subset_folder / LABELS_FOLDER_NAME
    elif (subset_folder.parent / LABELS_FOLDER_NAME).exists():
        annotations_root = subset_folder.parent / LABELS_FOLDER_NAME
    else:
        raise RuntimeError("Could not infer annotations_root path.")
    return annotations_root


def cleanup_excess_annotations(subset_folder: Path):
    """
    Removes any annotations file for which there is no matching image found
    in subset_folder.

    Infers path to annotations_folder assuming, in order of precedence, the
    path to be formulated as::

        * <subset_folder>/YOLO_darknet
        * <subset_folder>/labels
        * <subset_folder.parent>/labels

    """
    annotations_folder = _infer_annotations_folder_path(subset_folder=subset_folder)
    annotation_paths = get_all_txt_recursive(root_dir=annotations_folder)
    for annotation_path in annotation_paths:
        image_path = subset_folder / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            annotation_path.unlink()
